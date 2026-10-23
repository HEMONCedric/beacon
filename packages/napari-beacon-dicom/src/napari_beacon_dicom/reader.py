"""Discover and load DICOM image series with physical geometry intact."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import SimpleITK as sitk

PathOrPaths = str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
_DICOM_SUFFIXES = {".dcm", ".dicom", ".ima"}
_DICOM_READER_PATTERNS = ("*.dcm", "*.dicom", "*.ima")
_PLUGIN_NAME = "napari-beacon-dicom"

_SAFE_TAGS = {
    "modality": "0008|0060",
    "study_description": "0008|1030",
    "series_description": "0008|103e",
    "series_number": "0020|0011",
    "series_instance_uid": "0020|000e",
    "study_instance_uid": "0020|000d",
    "frame_of_reference_uid": "0020|0052",
    "window_center": "0028|1050",
    "window_width": "0028|1051",
}


@dataclass(frozen=True)
class DicomSeries:
    """Non-identifying description of one spatially sortable DICOM series."""

    directory: Path
    series_uid: str
    files: tuple[Path, ...]
    modality: str = ""
    study_description: str = ""
    series_description: str = ""
    series_number: str = ""
    study_instance_uid: str = ""
    frame_of_reference_uid: str = ""
    window_center: float | None = None
    window_width: float | None = None

    @property
    def display_name(self) -> str:
        description = self.series_description or self.study_description
        prefix = f"{self.modality} " if self.modality else ""
        number = f"{self.series_number} - " if self.series_number else ""
        return f"{number}{prefix}{description or 'DICOM series'}".strip()


@dataclass(frozen=True)
class LoadedDicomVolume:
    """A DICOM volume prepared as one napari Image layer."""

    data: np.ndarray
    affine_zyx: np.ndarray
    name: str
    metadata: dict[str, Any]
    contrast_limits: tuple[float, float] | None

    @property
    def scale_zyx(self) -> tuple[float, float, float]:
        """Voxel spacing in the native array axes used for display."""
        linear = np.asarray(self.affine_zyx[:3, :3], dtype=float)
        return tuple(float(value) for value in np.linalg.norm(linear, axis=0))

    def layer_data(self) -> tuple[np.ndarray, dict[str, Any], str]:
        # napari 0.6 cannot reliably slice a 3-D layer when a permutation or
        # oblique patient orientation is encoded in ``affine``.  A sagittal
        # acquisition can consequently appear as a thin reformatted slab
        # instead of its native stack.  Display the native acquisition axes
        # with physically correct voxel spacing.  The complete DICOM geometry
        # remains in metadata for inference/export.
        attributes: dict[str, Any] = {
            "name": self.name,
            "scale": self.scale_zyx,
            "metadata": self.metadata,
            "colormap": "gray",
        }
        if self.contrast_limits is not None:
            attributes["contrast_limits"] = self.contrast_limits
        return self.data, attributes, "image"


def _text(reader: sitk.ImageFileReader, tag: str) -> str:
    if not reader.HasMetaDataKey(tag):
        return ""
    return reader.GetMetaData(tag).replace("\x00", "").strip()


def _first_number(value: str) -> float | None:
    if not value:
        return None
    try:
        number = float(value.split("\\", 1)[0].strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _header(path: Path) -> dict[str, str]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOff()
    reader.ReadImageInformation()
    return {name: _text(reader, tag) for name, tag in _SAFE_TAGS.items()}


def _directories(root: Path, recursive: bool) -> list[Path]:
    if not recursive:
        return [root]
    result: list[Path] = []
    for directory, _subdirectories, filenames in os.walk(root):
        if filenames:
            result.append(Path(directory))
    return result or [root]


def discover_dicom_series(
    directory: str | os.PathLike[str], *, recursive: bool = True
) -> list[DicomSeries]:
    """Find DICOM image series below ``directory`` without reading pixels."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"DICOM directory does not exist: {root}")

    series: list[DicomSeries] = []
    for candidate in _directories(root, recursive):
        try:
            identifiers = (
                sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(candidate)) or ()
            )
        except RuntimeError:
            continue
        for identifier in identifiers:
            try:
                filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                    str(candidate), identifier, False, False, False
                )
            except RuntimeError:
                continue
            files = tuple(Path(filename).resolve() for filename in filenames)
            if not files:
                continue
            try:
                tags = _header(files[0])
            except RuntimeError:
                continue
            series.append(
                DicomSeries(
                    directory=candidate.resolve(),
                    series_uid=tags["series_instance_uid"] or str(identifier),
                    files=files,
                    modality=tags["modality"],
                    study_description=tags["study_description"],
                    series_description=tags["series_description"],
                    series_number=tags["series_number"],
                    study_instance_uid=tags["study_instance_uid"],
                    frame_of_reference_uid=tags["frame_of_reference_uid"],
                    window_center=_first_number(tags["window_center"]),
                    window_width=_first_number(tags["window_width"]),
                )
            )

    def sort_key(item: DicomSeries) -> tuple[int, str, str, str]:
        try:
            number = int(item.series_number)
        except ValueError:
            number = 2**31 - 1
        relative = str(item.directory.relative_to(root))
        return number, item.modality, item.series_description, relative

    series.sort(key=sort_key)
    return series


def _affine_zyx(image: sitk.Image) -> np.ndarray:
    if image.GetDimension() != 3:
        raise ValueError(
            f"Only scalar 3D DICOM image series are supported, got {image.GetDimension()}D"
        )
    direction_xyz = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    spacing_xyz = np.asarray(image.GetSpacing(), dtype=float)
    origin_xyz = np.asarray(image.GetOrigin(), dtype=float)
    if not (
        np.all(np.isfinite(direction_xyz))
        and np.all(np.isfinite(spacing_xyz))
        and np.all(np.isfinite(origin_xyz))
        and np.all(spacing_xyz > 0)
    ):
        raise ValueError("The DICOM series has invalid physical geometry")
    reverse = np.eye(3)[::-1]
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = reverse @ (direction_xyz @ np.diag(spacing_xyz)) @ reverse
    affine[:3, 3] = reverse @ origin_xyz
    return affine


def _contrast_limits(
    data: np.ndarray, series: DicomSeries
) -> tuple[float, float] | None:
    center, width = series.window_center, series.window_width
    if center is not None and width is not None and width > 0:
        return center - width / 2.0, center + width / 2.0
    if not data.size:
        return None
    low, high = np.percentile(data, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.min(data)), float(np.max(data))
    return (float(low), float(high)) if low < high else None


def read_dicom_series_image(series: DicomSeries) -> sitk.Image:
    """Read one descriptor as a 3D SimpleITK image."""
    if not series.files:
        raise ValueError("The selected DICOM series contains no files")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in series.files])
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    image = reader.Execute()
    # GDCM exposes enhanced/multiframe objects as one file. Reading that file
    # directly preserves its native 3-D frame stack, whereas a series reader
    # may introduce an extra singleton series axis depending on the transfer
    # syntax and SimpleITK build.
    if len(series.files) == 1:
        single_file_image = sitk.ReadImage(str(series.files[0]))
        if single_file_image.GetDimension() == 3:
            image = single_file_image
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Only scalar DICOM image series are supported")
    if image.GetDimension() != 3:
        raise ValueError(
            f"Only 3D DICOM image series are supported, got {image.GetDimension()}D"
        )
    return image


def read_dicom_series(series: DicomSeries) -> LoadedDicomVolume:
    """Read one descriptor into a geometry-aware napari volume."""
    image = read_dicom_series_image(series)
    data = sitk.GetArrayFromImage(image)
    if data.ndim != 3:
        raise ValueError(f"Only 3D DICOM image series are supported, got shape {data.shape}")

    safe_metadata = {
        "modality": series.modality,
        "study_description": series.study_description,
        "series_description": series.series_description,
        "series_number": series.series_number,
        "series_instance_uid": series.series_uid,
        "study_instance_uid": series.study_instance_uid,
        "frame_of_reference_uid": series.frame_of_reference_uid,
        "instance_count": len(series.files),
        "spacing_xyz_mm": tuple(float(value) for value in image.GetSpacing()),
        "origin_xyz_lps_mm": tuple(float(value) for value in image.GetOrigin()),
        "direction_xyz_lps": tuple(float(value) for value in image.GetDirection()),
    }
    return LoadedDicomVolume(
        data=data,
        affine_zyx=_affine_zyx(image),
        name=series.display_name,
        metadata={"dicom": safe_metadata},
        contrast_limits=_contrast_limits(data, series),
    )


def _normalise_path(path: PathOrPaths) -> Path:
    if isinstance(path, (str, os.PathLike)):
        return Path(path).expanduser().resolve()
    paths = [Path(item).expanduser().resolve() for item in path]
    if not paths:
        raise ValueError("No DICOM path was provided")
    if len(paths) == 1:
        return paths[0]
    parents = {item.parent for item in paths}
    if len(parents) != 1:
        raise ValueError("Selected DICOM files must belong to the same directory")
    return paths[0]


def _series_for_path(path: Path) -> DicomSeries:
    if path.is_dir():
        discovered = discover_dicom_series(path)
        if not discovered:
            raise ValueError(f"No DICOM image series was found in {path}")
        if len(discovered) != 1:
            raise ValueError(
                f"Found {len(discovered)} DICOM series. Use Plugins > BEACON DICOM > "
                "Open DICOM series to select one."
            )
        return discovered[0]
    if not path.is_file():
        raise ValueError(f"DICOM path does not exist: {path}")

    try:
        tags = _header(path)
    except RuntimeError as exc:
        raise ValueError(f"Not a readable DICOM image: {path.name}") from exc
    uid = tags["series_instance_uid"]
    discovered = discover_dicom_series(path.parent, recursive=False)
    for item in discovered:
        if uid and item.series_uid == uid:
            return item
        if path in item.files:
            return item
    raise ValueError(f"No spatial DICOM series contains {path.name}")


def read_dicom(path: PathOrPaths) -> list[tuple[np.ndarray, dict[str, Any], str]]:
    """npe2 reader entry point for a DICOM file or single-series folder."""
    return [read_dicom_series(_series_for_path(_normalise_path(path))).layer_data()]


def read_dicom_image(path: PathOrPaths) -> sitk.Image:
    """Read the complete series containing one selected DICOM file.

    This is intended for BEACON workflows that need a SimpleITK reference
    image for later mask export rather than napari layer data.
    """
    return read_dicom_series_image(_series_for_path(_normalise_path(path)))


def get_reader(path: PathOrPaths):
    """Return the DICOM reader callable required by the npe2 reader contract."""
    try:
        candidate = _normalise_path(path)
    except ValueError:
        return None
    from .volume_reader import VOLUME_SUFFIXES, read_medical_volume

    if str(candidate).lower().endswith(VOLUME_SUFFIXES):
        return read_medical_volume
    if candidate.is_dir() or candidate.suffix.lower() in _DICOM_SUFFIXES:
        return read_dicom
    return None


def configure_dicom_reader_preference(settings=None) -> None:
    """Prefer BEACON's geometry-aware DICOM and medical-volume readers.

    Napari's built-in image reader also advertises ``*.dcm``. Without an
    explicit preference, drag-and-drop can therefore open individual 2D
    instances rather than asking this plugin to assemble their complete
    SeriesInstanceUID.
    """
    if settings is None:
        from napari.settings import get_settings

        settings = get_settings()
    from .volume_reader import VOLUME_SUFFIXES

    preferences = {
        pattern: _PLUGIN_NAME
        for pattern in (*_DICOM_READER_PATTERNS, *(f"*{suffix}" for suffix in VOLUME_SUFFIXES))
    }
    # napari 0.6 gives '*.gz' and '*.nii.gz' the same specificity score.
    # Ties follow insertion order, so appending our rules leaves an existing
    # '*.gz -> napari-nifti' rule in charge of NIfTI imports. Put our rules
    # first, retaining unrelated rules (including '*.gz' for other formats).
    settings.plugins.extension2reader = {
        **preferences,
        **{
            pattern: reader
            for pattern, reader in settings.plugins.extension2reader.items()
            if pattern not in preferences
        },
    }


__all__ = [
    "DicomSeries",
    "LoadedDicomVolume",
    "discover_dicom_series",
    "configure_dicom_reader_preference",
    "get_reader",
    "read_dicom",
    "read_dicom_image",
    "read_dicom_series",
    "read_dicom_series_image",
]
