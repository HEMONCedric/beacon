"""Physical-space dilation and replayable positive-lasso refinement."""

from dataclasses import dataclass
import zlib

import numpy as np
from scipy.ndimage import binary_dilation


def binary_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 3 or array.dtype.kind not in "biu" or array.min() < 0 or array.max() > 1:
        raise ValueError("This workflow requires a binary 3D mask (labels 0 and 1)")
    return array


@dataclass(frozen=True)
class MaskSnapshot:
    shape: tuple
    dtype: str
    payload: bytes
    extra_count: int

    @classmethod
    def capture(cls, mask, extra_count):
        array = np.ascontiguousarray(mask)
        return cls(array.shape, array.dtype.str, zlib.compress(memoryview(array).cast("B"), 1), extra_count)

    def restore(self):
        return np.frombuffer(zlib.decompress(self.payload), dtype=self.dtype).reshape(self.shape)


def dilate_mask_mm(mask, spacing_zyx, radius_mm=2.0):
    array = binary_mask(mask)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Dilation requires valid physical voxel spacing")
    if not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError("Dilation radius must be finite and non-negative")
    radii = np.ceil(radius_mm / spacing).astype(int)
    axes = np.ogrid[tuple(slice(-n, n + 1) for n in radii)]
    distance_squared = sum((axis * step) ** 2 for axis, step in zip(axes, spacing))
    kernel = distance_squared <= radius_mm ** 2 + 1e-9
    # Unlike distance_transform_edt, this needs no full-volume float64 map.
    return binary_dilation(array, structure=kernel, iterations=1, border_value=0).astype(np.uint8)


def dilate_with_snapshot(mask, spacing_zyx, extra_count):
    return dilate_mask_mm(mask, spacing_zyx), MaskSnapshot.capture(mask, extra_count)


def refine_with_lassos(session, mask, prompts, extra_count):
    """Rebuild backend state from the displayed mask and retained lassos.

    The image is NOT uploaded again. Replay also makes the next refinement safe
    after a client-side dilation or Undo, independently of server undo depth.
    """
    array = binary_mask(mask)
    if not getattr(session, "supports_initial_label", False):
        raise ValueError("This checkpoint cannot refine an existing mask")
    if not prompts:
        raise ValueError("A refinement requires a lasso")
    for prompt in prompts:
        prompt.validate(array.shape)
    snapshot = MaskSnapshot.capture(array, extra_count)
    target = np.zeros_like(array, dtype=np.uint8)
    session.set_target_buffer(target)
    # Reconnect can leave the backend pointing at the live napari array.
    # Detach it before reset (which clears its target in-place), so a failed
    # request cannot erase the displayed mask from the worker thread.
    getattr(session, "reset_for_refinement", session.reset_interactions)()
    if np.any(array):
        # Initial-seg seeding resets the backend's interactions, so it must
        # precede replay of the three initial AND all additional lassos.
        session.add_initial_seg_interaction(array, run_prediction=False)
    from .volume_geometry import submit_lasso

    for index, prompt in enumerate(prompts):
        submit_lasso(session, prompt, index == len(prompts) - 1)
    return target, snapshot
