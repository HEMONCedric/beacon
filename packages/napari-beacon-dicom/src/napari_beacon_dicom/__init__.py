"""DICOM image-series support for BEACON."""

from __future__ import annotations

from .reader import (
    DicomSeries,
    LoadedDicomVolume,
    configure_dicom_reader_preference,
    discover_dicom_series,
    get_reader,
    read_dicom,
    read_dicom_image,
    read_dicom_series,
    read_dicom_series_image,
)

__all__ = [
    "DicomSeries",
    "DicomSeriesWidget",
    "LoadedDicomVolume",
    "configure_dicom_reader_preference",
    "discover_dicom_series",
    "get_reader",
    "read_dicom",
    "read_dicom_image",
    "read_dicom_series",
    "read_dicom_series_image",
]


def __getattr__(name: str):
    """Keep reader helpers usable without importing Qt."""
    if name == "DicomSeriesWidget":
        from .widget import DicomSeriesWidget

        return DicomSeriesWidget
    raise AttributeError(name)
