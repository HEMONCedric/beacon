"""Adapter between BEACON's voxel grids and the released OncoRep interactive API."""

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import nibabel as nib
import numpy as np
import oncorep
import SimpleITK as sitk
import torch
from oncorep import prompts
from scipy.ndimage import affine_transform
from scipy.optimize import linear_sum_assignment

from .mask_refinement import MaskSnapshot
from .native_grid import grid_metadata


REVERSE = np.eye(4)[[2, 1, 0, 3]]
LPS_TO_RAS = np.diag([-1., -1., 1., 1.])
MAX_BLOCK_VOXELS = 128 * 256 * 256


@dataclass(frozen=True)
class UndoState:
    model_mask: MaskSnapshot | None
    initial_mask: tuple[np.ndarray, MaskSnapshot] | None
    accepted: list
    pending: list
    pending_count: int


def image_affine(grid):
    """SimpleITK xyz/LPS grid to NIfTI xyz/RAS."""
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(grid["direction_xyz_lps"]).reshape(3, 3) @ np.diag(grid["spacing_xyz_mm"])
    affine[:3, 3] = grid["origin_xyz_lps_mm"]
    return LPS_TO_RAS @ affine


def resolve_modality(override, metadata, data):
    if override != "Auto":
        return override
    declared = {"CT": "CT", "MR": "MR", "MRI": "MR"}
    modality = declared.get(str(metadata).strip().upper())
    if modality:
        return modality
    sample = data[tuple(slice(None, None, max(1, count // 48)) for count in data.shape)]
    if np.mean((sample >= -1100) & (sample <= -850)) >= .005 and np.mean(sample > 20) >= .01:
        return "CT"
    raise ValueError("Set the OncoRep modality to CT, MR or PET (SUV) in Settings.")


def coordinates(matrix, indices):
    return np.asarray(indices) @ matrix[:3, :3].T + matrix[:3, 3]


def box_corners(lo, hi):
    return np.asarray(list(product(*zip(lo, hi))))


def output_region(mask, start, destination_to_source, destination_shape):
    """Resample only the destination region touched by a source mask block."""
    stop = np.asarray(start) + mask.shape
    source_to_destination = np.linalg.inv(destination_to_source)
    corners = coordinates(source_to_destination, box_corners(np.asarray(start) - .5, stop - .5))
    lo = np.maximum(np.floor(corners.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(corners.max(0)).astype(int) + 1, destination_shape)
    if np.any(lo >= hi):
        raise ValueError("The model prediction is outside the selected image")
    offset = coordinates(destination_to_source, lo) - start
    resampled = affine_transform(
        mask, destination_to_source[:3, :3], offset, output_shape=tuple(hi - lo),
        order=0, mode="grid-constant", prefilter=False,
    )
    return tuple(slice(a, b) for a, b in zip(lo, hi)), resampled


def foreground_crop(mask):
    extent = [np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != axis))) for axis in range(3)]
    if not extent[0].size:
        return None
    start = np.array([indices[0] for indices in extent])
    section = tuple(slice(indices[0], indices[-1] + 1) for indices in extent)
    return mask[section], start


class LassoSession(oncorep.InteractiveSession):
    """Use OncoRep's native triplet and retain extra contours on their own slices."""

    def reset(self):
        super().reset()
        self.extra_lassos = []

    def extent(self, with_mask=False):
        bounds = super().extent(with_mask)
        if not self.extra_lassos:
            return bounds
        boxes = [bbox for _, _, _, bbox in self.extra_lassos]
        if bounds is not None:
            boxes.append(bounds)
        return np.min([box[0] for box in boxes], axis=0), np.max([box[1] for box in boxes], axis=0)

    def _encode(self, corner, size):
        if np.prod(size) > MAX_BLOCK_VOXELS:
            raise ValueError("These prompts span too large an area for interactive OncoRep. Outline one lesion at a time.")
        super()._encode(corner, size)

    def channels(self):
        channels = super().channels()
        for axis, index, plane, _ in self.extra_lassos:
            position = index - self.corner[axis]
            section = tuple(position if i == axis else slice(None) for i in range(3))
            origin = tuple(self.corner[i] for i in range(3) if i != axis)
            sampled = oncorep.preprocessing.crop3d(plane, origin, channels[(0, *section)].shape, False)
            np.maximum(channels[(0, *section)], sampled, out=channels[(0, *section)])
            channels[(1, *section)] = 1
        return channels


class OncoRepSession:
    segmentation_engine = "oncorep"
    preferred_scribble_thickness = (1, 1, 1)
    supported_interactions = {"points": True, "bbox2d": True, "scribble": True, "lasso": True}
    supports_initial_label = True
    supports_undo = True
    license = None

    def __init__(self, weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        spec = weights or "interactive"
        # Release weights contain only tensors and metadata. Keep the same
        # weights-only policy for offline files selected through BEACON.
        path, _ = oncorep.weights.resolve(spec)
        state = oncorep.weights.state_dict_of(torch.load(path, map_location="cpu", weights_only=True))
        architecture = oncorep.weights.architecture(state)
        if architecture["dim"] != 3 or architecture["prompt_channels"] != len(prompts.PROMPT_CHANNELS):
            raise ValueError("Select the OncoRep 3D interactive checkpoint (interactive.pt).")
        network = oncorep.FusionUNet(**architecture)
        network.load_state_dict(state)
        self.network = network.to(self.device).eval()
        self.target = None

    def set_volume(self, native_data, native_grid, displayed, modality, progress):
        started = perf_counter()
        progress(f"Preparing {modality} for IMPACT-OncoRep…")
        # The released loader applies the body crop, spacing, antialiasing and
        # modality-specific normalization. The temporary NIfTI is uncompressed.
        with TemporaryDirectory(prefix="beacon-oncorep-") as folder:
            path = Path(folder) / "input.nii"
            affine = image_affine(native_grid)
            image = nib.Nifti1Image(np.asarray(native_data).transpose(2, 1, 0), affine, dtype=native_data.dtype)
            image.set_sform(affine, code="scanner")
            image.set_qform(affine, code="scanner")
            nib.save(image, path)
            # Reference coverage only affects extra sequences. BEACON uses one.
            volume = oncorep.load([path], [modality], ref_view=False)
        self.engine = LassoSession(volume, self.network, device=self.device)
        self.modality = modality
        self.shape = displayed.GetSize()[::-1]
        self.display_to_grid = np.linalg.solve(
            volume.affine @ REVERSE, image_affine(grid_metadata(displayed)) @ REVERSE,
        )
        self.grid_to_display = np.linalg.inv(self.display_to_grid)
        normals = self.grid_to_display[:3, :3]
        normals = np.abs(normals / np.linalg.norm(normals, axis=1, keepdims=True))
        self.lasso_axes = dict(zip(*linear_sum_assignment(-normals)))
        self.accepted, self.pending = [], []
        self.last_undo = None
        self._initial_mask = None
        self.last_prediction_s = self.last_projection_s = 0.
        self.preprocess_s = perf_counter() - started

    def set_target_buffer(self, target):
        if target.shape != self.shape:
            raise ValueError("Mask buffer and selected image grid differ")
        self.target = target

    def set_do_autozoom(self, _enabled):
        pass  # OncoRep chooses its patch from the physical prompt extent.

    def _source_points(self, crop, bbox):
        points = np.argwhere(crop) + np.asarray(bbox)[:, 0]
        if not len(points):
            raise ValueError("Draw a non-empty prompt.")
        return points

    def _point_indices(self, points):
        return np.rint(coordinates(self.display_to_grid, points)).astype(int)

    def _queue(self, kind, payload, run):
        self.pending.append((kind, payload))
        if run:
            try:
                return self._predict()
            except Exception:
                self.pending.pop()
                raise

    def add_point_interaction(self, point, include_interaction, run_prediction):
        index = self.engine._check(self._point_indices(point))[0]
        return self._queue("click", (index, include_interaction), run_prediction)

    def add_bbox_interaction(self, bbox, include_interaction, run_prediction):
        if not include_interaction:
            raise ValueError("Use a negative point or scribble to remove tissue with OncoRep.")
        lo, hi = np.asarray(bbox, dtype=float).T
        corners = self._point_indices(box_corners(lo, np.maximum(lo, hi - 1)))
        return self._queue("box", (corners.min(0), corners.max(0) + 1), run_prediction)

    def add_scribble_interaction(self, mask, include_interaction, run_prediction, interaction_bbox):
        points = self._point_indices(self._source_points(mask, interaction_bbox))
        points = points[((points >= 0) & (points < self.engine.volume.shape)).all(1)]
        if not len(points):
            raise ValueError("The scribble is outside the OncoRep body crop.")
        return self._queue("scribble", (np.unique(points, axis=0), include_interaction), run_prediction)

    def add_lasso_interaction(self, mask, include_interaction, run_prediction, interaction_bbox, axis=None):
        if not include_interaction:
            raise ValueError("Use a negative point or scribble to remove tissue with OncoRep.")
        if axis is None:
            axis = next((i for i, size in enumerate(mask.shape) if size == 1), None)
        if axis is None or mask.shape[axis] != 1:
            raise ValueError("Draw the lasso on one 2D slice.")

        fixed = self.lasso_axes[axis]
        source_axes = [i for i in range(3) if i != axis]
        plane_axes = [i for i in range(3) if i != fixed]
        source = self._source_points(mask, interaction_bbox)
        centre = coordinates(self.display_to_grid, source.mean(0))
        index = int(np.rint(centre[fixed]))
        shape = self.engine.volume.shape
        if not 0 <= index < shape[fixed]:
            raise ValueError("The lasso plane is outside the OncoRep body crop.")

        # Project the filled outline onto the nearest model plane. Inverse
        # sampling preserves a solid contour even when the scan is anisotropic.
        matrix = self.display_to_grid[:3, :3]
        projected = matrix[np.ix_(plane_axes, source_axes)]
        inverse = np.linalg.inv(projected)
        start = np.asarray(interaction_bbox)[:, 0]
        origin = self.display_to_grid[plane_axes, 3] + matrix[plane_axes, axis] * start[axis]
        plane = affine_transform(
            np.squeeze(mask, axis=axis).astype(bool), inverse, -inverse @ origin - start[source_axes],
            output_shape=tuple(shape[i] for i in plane_axes), order=0, mode="grid-constant", prefilter=False,
        )
        if not plane.any():
            ij = np.rint(centre[plane_axes]).astype(int)
            if np.any(ij < 0) or np.any(ij >= plane.shape):
                raise ValueError("The lasso is outside the OncoRep body crop.")
            plane[tuple(ij)] = True
        bounds = np.argwhere(plane)
        bbox = (np.insert(bounds.min(0), fixed, index), np.insert(bounds.max(0) + 1, fixed, index + 1))
        return self._queue("lasso", (fixed, index, plane, bbox), run_prediction)

    def add_beacon_lasso(self, prompt, predict):
        return self.add_lasso_interaction(prompt.crop, True, predict, prompt.bbox, prompt.axis)

    def _apply(self, events):
        engine = self.engine
        engine.lasso = engine.box = None
        engine.positives.clear()
        engine.negatives.clear()
        engine.scribbles.clear()
        engine.negative_scribbles.clear()
        lassos = []
        for kind, payload in events:
            if kind == "click":
                engine.add_click(*payload)
            elif kind == "box":
                engine.set_box(*payload)
            elif kind == "scribble":
                pixels, positive = payload
                (engine.scribbles if positive else engine.negative_scribbles).append(pixels)
            else:
                lassos.append(payload)
        first = {plane[0]: plane for plane in lassos[:3]}
        if len(first) == 3:
            engine.set_lassos(oncorep.lasso_from_masks(
                tuple(first[i][1] for i in range(3)), *(first[i][2] for i in range(3)),
            ))
            lassos = lassos[3:]
        engine.extra_lassos = lassos

    def _mask_projection(self, mask, corner):
        lo = np.maximum(corner, 0)
        hi = np.minimum(np.asarray(corner) + mask.shape, self.engine.volume.shape)
        region = tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, corner))
        return output_region(mask[region], lo, self.display_to_grid, self.shape)

    def _snapshot(self, pending, pending_count=0):
        native = self.engine.mask
        return UndoState(
            MaskSnapshot.capture(native, 0) if native.any() else None,
            self._initial_mask,
            self.accepted.copy(), list(pending), pending_count,
        )

    def _restore_snapshot(self, state):
        self.accepted, self.pending = state.accepted, state.pending
        self.engine.mask.fill(False)
        if state.model_mask is not None:
            np.copyto(self.engine.mask, state.model_mask.restore())
        self.target.fill(0)
        self._initial_mask = state.initial_mask
        if self._initial_mask is not None:
            # Some input contours lose voxels when projected to the model grid.
            start, snapshot = self._initial_mask
            section = tuple(slice(a, a + n) for a, n in zip(start, snapshot.shape))
            self.target[section] = snapshot.restore()
        elif state.model_mask is not None:
            cropped, start = foreground_crop(self.engine.mask)
            section, mask = output_region(cropped, start, self.display_to_grid, self.shape)
            self.target[section] = mask
        self._apply(self.accepted)

    def _predict(self):
        if not self.pending:
            return None
        started = perf_counter()
        engine = self.engine
        before = self._snapshot(self.pending[:-1])
        undo = before
        if self.last_undo is not None and self.last_undo.pending_count == len(self.pending):
            # Running prompts already queued before an import belongs to the
            # import action. The latest marker in Classic is still that import.
            undo = replace(self.last_undo, pending_count=0)
        saved = engine.corner, engine.size, engine._skips, engine.last
        try:
            self._apply(self.accepted + self.pending)
            prediction = engine.predict()
            projected_at = perf_counter()
            section, mask = self._mask_projection(prediction.mask, prediction.corner)
        except Exception:
            engine.mask.fill(False)
            if before.model_mask is not None:
                np.copyto(engine.mask, before.model_mask.restore())
            engine.corner, engine.size, engine._skips, engine.last = saved
            self._apply(self.accepted)
            raise
        self.target.fill(0)
        self.target[section] = mask
        self._initial_mask = None
        self.last_undo = undo
        self.accepted.extend(self.pending)
        self.pending.clear()
        self.last_prediction_s = projected_at - started
        self.last_projection_s = perf_counter() - projected_at
        return None  # The previous result may extend beyond the new crop.

    def add_initial_seg_interaction(self, mask, run_prediction=False):
        if mask.shape != self.shape:
            raise ValueError("The initial mask must be on the selected image grid.")
        cropped = foreground_crop(mask)
        projected = np.zeros(self.engine.volume.shape, bool)
        initial_mask = None
        if cropped is not None:
            data, start = cropped
            data = data.astype(bool, copy=False)
            region, binary = output_region(
                data, start, self.grid_to_display, self.engine.volume.shape,
            )
            projected[region] = binary
            initial_mask = start, MaskSnapshot.capture(data, 0)
        previous_undo = self.last_undo
        undo = self._snapshot(self.pending, pending_count=len(self.pending))
        self.engine.set_mask(projected)
        np.not_equal(mask, 0, out=self.target)
        self._initial_mask = initial_mask
        self.last_undo = undo
        if run_prediction and self.pending:
            try:
                self._predict()
            except Exception:
                self._restore_snapshot(undo)
                self.last_undo = previous_undo
                raise

    def undo(self):
        if self.pending and (self.last_undo is None or len(self.pending) > self.last_undo.pending_count):
            self.pending.pop()
            return True
        if self.last_undo is None:
            return False
        self._restore_snapshot(self.last_undo)
        self.last_undo = None
        return True

    def reset_interactions(self):
        self.engine.reset()
        self.accepted.clear()
        self.pending.clear()
        self.last_undo = None
        self._initial_mask = None
        self.target.fill(0)

    def reset_for_refinement(self):
        engine = self.engine
        encoding = engine.corner, engine.size, engine._skips
        self.reset_interactions()
        engine.corner, engine.size, engine._skips = encoding
