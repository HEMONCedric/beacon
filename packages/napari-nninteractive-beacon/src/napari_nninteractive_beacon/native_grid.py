"""Versioned, GUI-free native-volume transport and exact-grid resampling."""

from math import prod

import numpy as np
import SimpleITK as sitk

PROTOCOL_VERSION = 1
NATIVE_PATH = "/beacon/set_native_image"
CAPABILITIES_PATH = "/beacon/native_upload"
MAX_VOXELS = 512 * 1024**2


def grid_metadata(image):
    return {"size_xyz": list(image.GetSize()), "spacing_xyz_mm": list(image.GetSpacing()),
            "origin_xyz_lps_mm": list(image.GetOrigin()),
            "direction_xyz_lps": list(image.GetDirection())}


def validate_grid(grid):
    if not isinstance(grid, dict):
        raise ValueError("Missing image grid")
    size = grid.get("size_xyz")
    if not isinstance(size, (list, tuple)) or len(size) != 3 or any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in size
    ) or prod(size) > MAX_VOXELS:
        raise ValueError("Invalid or oversized image grid")
    for key, length in (("spacing_xyz_mm", 3), ("origin_xyz_lps_mm", 3), ("direction_xyz_lps", 9)):
        values = np.asarray(grid.get(key), dtype=float)
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid grid {key}")
    if np.any(np.asarray(grid["spacing_xyz_mm"]) <= 0):
        raise ValueError("Grid spacing must be positive")
    direction = np.asarray(grid["direction_xyz_lps"]).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Grid direction must be orthonormal")
    return grid


def copy_grid(image, grid):
    validate_grid(grid)
    if tuple(image.GetSize()) != tuple(grid["size_xyz"]):
        raise ValueError("Image array does not match its declared grid")
    image.SetSpacing(tuple(grid["spacing_xyz_mm"]))
    image.SetOrigin(tuple(grid["origin_xyz_lps_mm"]))
    image.SetDirection(tuple(grid["direction_xyz_lps"]))


def same_grid(first, second):
    try:
        validate_grid(first)
        validate_grid(second)
        return all(np.asarray(first[key]).shape == np.asarray(second[key]).shape
                   and np.allclose(first[key], second[key], rtol=0, atol=1e-7)
                   for key in first)
    except (ValueError, TypeError, KeyError):
        return False


def resample_exact(image, target_grid):
    """Shared by the workstation and server, with no dimension rounding here."""
    validate_grid(target_grid)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(x) for x in target_grid["size_xyz"]])
    resampler.SetOutputSpacing(tuple(target_grid["spacing_xyz_mm"]))
    resampler.SetOutputOrigin(tuple(target_grid["origin_xyz_lps_mm"]))
    resampler.SetOutputDirection(tuple(target_grid["direction_xyz_lps"]))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetUseNearestNeighborExtrapolator(True)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(image)


def resample_native(array, source_grid, target_grid):
    validate_grid(source_grid)
    validate_grid(target_grid)
    if array.ndim != 4 or array.shape[0] != 1 or array.dtype.kind not in "iuf":
        raise ValueError("Expected a real, single-channel [1,z,y,x] image")
    if tuple(array.shape[1:][::-1]) != tuple(source_grid["size_xyz"]):
        raise ValueError("Native image shape does not match source grid")
    if not np.allclose(target_grid["spacing_xyz_mm"], (1, 1, 1), rtol=0, atol=1e-8):
        raise ValueError("BEACON working grid must have 1 mm spacing")
    for key in ("origin_xyz_lps_mm", "direction_xyz_lps"):
        if not np.allclose(source_grid[key], target_grid[key], rtol=0, atol=1e-7):
            raise ValueError("Working grid must retain source origin and direction")
    image = sitk.GetImageFromArray(np.ascontiguousarray(array[0]))
    copy_grid(image, source_grid)
    return sitk.GetArrayFromImage(resample_exact(image, target_grid))[None]


def sample_values(array):
    flat = array.reshape(-1)
    indices = np.linspace(0, flat.size - 1, min(257, flat.size), dtype=np.int64)
    return flat[indices].astype(float).tolist()
