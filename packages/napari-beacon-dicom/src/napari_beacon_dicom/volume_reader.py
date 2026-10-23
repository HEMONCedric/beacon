"""Read medical volumes in BEACON's native-axis display frame."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import SimpleITK as sitk


VOLUME_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")
SEGMENTATION_MARKER = "BEACON segmentation"


def geometry_metadata(image: sitk.Image) -> dict[str, Any]:
    return {
        "size_xyz": tuple(image.GetSize()),
        "spacing_xyz_mm": tuple(image.GetSpacing()),
        "origin_xyz_lps_mm": tuple(image.GetOrigin()),
        "direction_xyz_lps": tuple(image.GetDirection()),
    }


def physical_matrix_zyx(image: sitk.Image) -> np.ndarray:
    reverse = np.eye(3)[::-1]
    result = np.eye(4)
    result[:3, :3] = reverse @ (
        np.array(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    ) @ reverse
    result[:3, 3] = reverse @ np.array(image.GetOrigin())
    return result


def _same_geometry(candidate: dict, expected: dict) -> bool:
    try:
        return all(
            np.asarray(candidate[key]).shape == np.asarray(values).shape
            and np.allclose(candidate[key], values, rtol=1e-6, atol=1e-4)
            for key, values in expected.items()
        )
    except (KeyError, TypeError, ValueError):
        return False


def _reference_display_matrix(image: sitk.Image, viewer) -> np.ndarray | None:
    if viewer is None:
        return None
    expected = geometry_metadata(image)
    candidates = []
    for layer in viewer.layers:
        reference = layer.metadata.get("beacon_export_reference", {})
        if _same_geometry(reference.get("geometry", {}), expected):
            matrix = np.asarray(reference.get("display_matrix_zyx"), dtype=float)
            if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                candidates.append((layer, matrix))
        geometry = layer.metadata.get("dicom", layer.metadata.get("beacon_volume", {}))
        if geometry:
            geometry = dict(geometry)
            shape = getattr(layer.data, "shape", ())
            geometry.setdefault("size_xyz", tuple(shape[::-1]))
            if _same_geometry(geometry, expected):
                candidates.append((layer, layer._data_to_world.affine_matrix.copy()))
    if not candidates:
        return None
    for layer, matrix in candidates:
        if layer is viewer.layers.selection.active:
            return matrix
    first = candidates[0][1]
    # Never choose arbitrarily between identical grids placed differently.
    return first if all(np.allclose(first, matrix) for _, matrix in candidates) else None


def _is_labels(image: sitk.Image, array: np.ndarray, path: Path) -> bool:
    for key in ("descrip", "ITK_FileNotes", "beacon_segmentation"):
        if image.HasMetaDataKey(key):
            value = image.GetMetaData(key)
            if SEGMENTATION_MARKER in value or (key == "beacon_segmentation" and value == "1"):
                return True
    if image.HasMetaDataKey("intent_code"):
        try:
            if int(float(image.GetMetaData("intent_code"))) == 1002:
                return True
        except ValueError:
            pass
    # Compatibility with earlier unmarked, binary BEACON exports. Require
    # both a mask filename and binary integer data, not just an 8-bit image.
    name = path.name.lower()
    for suffix in VOLUME_SUFFIXES:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return bool(
        re.search(r"(^|[_.-])(mask|masks|seg|segmentation|labels)([_.-]|$)", name)
        and np.issubdtype(array.dtype, np.integer)
        and array.min() >= 0 and array.max() <= 1
    )


def read_medical_volume(path, *, viewer=None):
    if viewer is None:
        from napari import current_viewer

        viewer = current_viewer()
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    result = []
    for item in paths:
        item = Path(item)
        image = sitk.ReadImage(str(item))
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("BEACON medical-volume import requires a scalar 3D image")
        array = sitk.GetArrayFromImage(image)
        matrix = _reference_display_matrix(image, viewer)
        if matrix is None:
            # Same convention as the DICOM reader: native array axes, physical
            # spacing, local origin. Patient geometry remains available below.
            matrix = np.diag((*image.GetSpacing()[::-1], 1.0))
        linear = matrix[:3, :3]
        scale = np.linalg.norm(linear, axis=0)
        attributes = {
            "name": item.name.removesuffix(".gz").rsplit(".", 1)[0],
            "scale": tuple(scale),
            "translate": tuple(matrix[:3, 3]),
            "rotate": linear / scale,
            "metadata": {"beacon_volume": geometry_metadata(image)},
        }
        labels = _is_labels(image, array, item)
        if labels:
            attributes["opacity"] = 0.5
        else:
            attributes["colormap"] = "gray"
        result.append((array, attributes, "labels" if labels else "image"))
    return result
