"""Isotropic working volumes and label export on the acquisition grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from napari_beacon_dicom.volume_reader import geometry_metadata, physical_matrix_zyx


MAX_WORKING_VOXELS = 512 * 1024**2


@dataclass
class PreparedVolume:
    """Working pixels and geometry references for both voxel grids."""
    name: str
    original: sitk.Image
    working: sitk.Image
    data: np.ndarray
    display_matrix_zyx: np.ndarray
    plane_names: tuple[str, str, str]

    def layer_transform(self) -> dict[str, Any]:
        """Viewer coordinates are independent of scanner coordinates."""
        linear = self.display_matrix_zyx[:3, :3]
        scale = np.linalg.norm(linear, axis=0)
        return {
            "scale": tuple(float(x) for x in scale),
            "translate": tuple(float(x) for x in self.display_matrix_zyx[:3, 3]),
            "rotate": linear / scale,
            "shear": (0.0, 0.0, 0.0),
            "affine": np.eye(4),
        }

    def mask_in_original_space(self, mask: np.ndarray) -> sitk.Image:
        if tuple(mask.shape[::-1]) != self.working.GetSize():
            raise ValueError("The segmentation does not match the working volume")
        labels = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8))
        labels.CopyInformation(self.working)
        return sitk.Resample(
            labels, self.original, sitk.Transform(), sitk.sitkNearestNeighbor,
            0, sitk.sitkUInt8,
        )

    def original_display_reference(self) -> dict[str, Any]:
        matrix = self.display_matrix_zyx @ np.linalg.solve(
            physical_matrix_zyx(self.working), physical_matrix_zyx(self.original)
        )
        return {"geometry": geometry_metadata(self.original),
                "display_matrix_zyx": matrix.tolist()}


def image_from_layer(data: Any, matrix_zyx: np.ndarray, metadata: dict) -> sitk.Image:
    """Capture voxel geometry without reading a layer from a worker thread."""
    array = np.asarray(data)
    if array.ndim != 3 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("Choose a scalar 3D image")
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    dicom = metadata.get("dicom", metadata.get("beacon_volume", {}))
    if all(key in dicom for key in (
        "spacing_xyz_mm", "origin_xyz_lps_mm", "direction_xyz_lps"
    )):
        image.SetSpacing(tuple(dicom["spacing_xyz_mm"]))
        image.SetOrigin(tuple(dicom["origin_xyz_lps_mm"]))
        image.SetDirection(tuple(dicom["direction_xyz_lps"]))
    else:
        # napari uses z/y/x; SimpleITK uses x/y/z physical coordinates.
        reverse = np.eye(3)[::-1]
        matrix = np.asarray(matrix_zyx, dtype=float)
        linear = reverse @ matrix[:3, :3] @ reverse
        spacing = np.linalg.norm(linear, axis=0)
        if not np.isfinite(matrix).all() or np.any(spacing <= 0):
            raise ValueError("The image has invalid voxel geometry")
        direction = linear / spacing
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
            raise ValueError("Sheared image grids are not supported; use the original image file")
        image.SetSpacing(tuple(float(x) for x in spacing))
        image.SetOrigin(tuple(float(x) for x in reverse @ matrix[:3, 3]))
        image.SetDirection(tuple(float(x) for x in direction.ravel()))
    return image


def prepare_volume(
    image: sitk.Image, name: str, display_matrix_zyx: np.ndarray | None = None,
    export_reference: sitk.Image | None = None,
) -> PreparedVolume:
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Choose a scalar 3D image")
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("The image has invalid voxel spacing")
    if min(image.GetSize()) < 2:
        raise ValueError("Three-plane lasso requires a volume with at least two voxels per axis")
    # Change only sampling density. Retain acquisition axes and voxel-zero
    # location, including oblique/sagittal directions and reflected axes.
    size = np.ceil((np.asarray(image.GetSize()) - 1) * spacing - 1e-7).astype(int) + 1
    if int(np.prod(size)) > MAX_WORKING_VOXELS:
        raise ValueError("The 1 mm volume is too large (more than 512 million voxels)")
    export_reference = image if export_reference is None else export_reference
    original = sitk.Image(export_reference.GetSize(), sitk.sitkUInt8)
    original.CopyInformation(export_reference)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(x) for x in size])
    resampler.SetOutputSpacing((1.0, 1.0, 1.0))
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetUseNearestNeighborExtrapolator(True)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    working = resampler.Execute(image)
    array = sitk.GetArrayFromImage(working)
    reference = sitk.Image(working.GetSize(), sitk.sitkUInt8)
    reference.CopyInformation(working)
    if display_matrix_zyx is None:
        # Match the DICOM reader's native display frame, not patient-space
        # translation/direction. Scanner geometry is retained in the references.
        display_matrix_zyx = np.diag((*spacing[::-1], 1.0))
    display_matrix = np.asarray(display_matrix_zyx, dtype=float) @ np.diag(
        (*tuple(1 / spacing[::-1]), 1.0)
    )
    orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(image.GetDirection())
    names = {"L": "Sagittal", "R": "Sagittal", "P": "Coronal",
             "A": "Coronal", "S": "Axial", "I": "Axial"}
    plane_names = tuple(names[letter] for letter in orientation[::-1])
    return PreparedVolume(name, original, reference, array, display_matrix, plane_names)


def _same_voxels(array: np.ndarray, candidate: np.ndarray) -> bool:
    if array.shape != candidate.shape:
        return False
    # Bound temporary comparison arrays even for a full CT scan.
    return all(np.allclose(array[start:start + 16], candidate[start:start + 16],
                           rtol=1e-6, atol=1e-5, equal_nan=True)
               for start in range(0, array.shape[0], 16))


def prepare_layer(
    data: Any, matrix_zyx: np.ndarray, metadata: dict, name: str,
    source_path: str | None = None,
) -> PreparedVolume:
    # Some napari NIfTI readers retain a unit layer scale and keep the header
    # geometry only in the source file. Recover it when the layer still contains
    # that file's voxels (in either z/y/x or x/y/z array order).
    if source_path and not any(key in metadata for key in ("dicom", "beacon_volume")):
        path = Path(source_path)
        if path.is_file() and any(str(path).lower().endswith(ext) for ext in (
            ".nii", ".nii.gz", ".mha", ".mhd", ".nrrd",
        )):
            try:
                source = sitk.ReadImage(str(path))
            except RuntimeError:
                source = None
            if source is not None and source.GetDimension() == 3 and source.GetNumberOfComponentsPerPixel() == 1:
                original_data = sitk.GetArrayFromImage(source)
                array = np.asarray(data)
                for transposed, candidate in enumerate((original_data, original_data.transpose(2, 1, 0))):
                    if _same_voxels(array, candidate):
                        if transposed:
                            # Keep the active layer's array axes, but map them
                            # correctly to patient space and export on the file grid.
                            layer_image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
                            layer_image.SetSpacing(source.GetSpacing()[::-1])
                            layer_image.SetOrigin(source.GetOrigin())
                            direction = np.array(source.GetDirection()).reshape(3, 3) @ np.eye(3)[::-1]
                            layer_image.SetDirection(tuple(direction.ravel()))
                            return prepare_volume(layer_image, name, matrix_zyx, source)
                        return prepare_volume(source, name, matrix_zyx)
    return prepare_volume(image_from_layer(data, matrix_zyx, metadata), name, matrix_zyx)


def prepare_path(path: str) -> PreparedVolume:
    source = Path(path).expanduser()
    if source.is_dir() or source.suffix.lower() == ".dcm":
        from napari_beacon_dicom import read_dicom_image

        image = read_dicom_image(source)
    else:
        header = sitk.ImageFileReader()
        header.SetFileName(str(source))
        header.ReadImageInformation()
        if header.HasMetaDataKey("0020|000e"):
            # DICOM exports also use extensionless or CT.<UID> filenames.
            from napari_beacon_dicom import read_dicom_image

            image = read_dicom_image(source)
        else:
            image = sitk.ReadImage(str(source))
    return prepare_volume(image, source.name)


@dataclass
class LassoPrompt:
    axis: int
    crop: np.ndarray
    bbox: list[list[int]]

    def validate(self, shape: tuple[int, ...]) -> None:
        if self.axis not in (0, 1, 2) or len(self.bbox) != 3:
            raise ValueError("A lasso must belong to one of the three planes")
        expected = []
        for axis, (lo, hi) in enumerate(self.bbox):
            if not 0 <= lo < hi <= shape[axis]:
                raise ValueError("The lasso extends outside the working image")
            expected.append(hi - lo)
        if expected[self.axis] != 1 or tuple(expected) != self.crop.shape:
            raise ValueError("A lasso must be a single 2D slice")
        if not np.any(self.crop):
            raise ValueError("The lasso is empty")


def predict_three_lassos(session, prompts: list[LassoPrompt], shape: tuple[int, ...]) -> np.ndarray:
    if {prompt.axis for prompt in prompts} != {0, 1, 2} or len(prompts) != 3:
        raise ValueError("Draw one lasso in each of the three planes before generating")
    for prompt in prompts:
        prompt.validate(shape)
    target = np.zeros(shape, dtype=np.uint8)
    session.reset_interactions()
    session.set_target_buffer(target)
    for index, prompt in enumerate(prompts):
        session.add_lasso_interaction(
            prompt.crop, True, run_prediction=index == 2,
            interaction_bbox=prompt.bbox,
        )
    return target
