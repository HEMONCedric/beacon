"""Qt series selector for folders containing one or more DICOM acquisitions."""

from __future__ import annotations

from pathlib import Path

from napari.qt.threading import create_worker
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .reader import (
    DicomSeries,
    LoadedDicomVolume,
    configure_dicom_reader_preference,
    discover_dicom_series,
    read_dicom_series,
)


class DicomSeriesWidget(QWidget):
    """Discover and load a selected DICOM image series without blocking Qt."""

    def __init__(self, napari_viewer) -> None:
        super().__init__()
        configure_dicom_reader_preference()
        self.viewer = napari_viewer
        self._worker = None
        self._series: list[DicomSeries] = []
        self.setMinimumWidth(520)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Folder containing a DICOM image series")
        self.path_edit.returnPressed.connect(self._scan)
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse_button)
        root.addLayout(path_row)

        controls = QHBoxLayout()
        self.recursive_check = QCheckBox("Include subfolders")
        self.recursive_check.setChecked(True)
        self.scan_button = QPushButton("Scan")
        self.scan_button.clicked.connect(self._scan)
        controls.addWidget(self.recursive_check)
        controls.addStretch(1)
        controls.addWidget(self.scan_button)
        root.addLayout(controls)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Series", "Modality", "Description", "Slices", "Folder"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.doubleClicked.connect(self._load_selected)
        root.addWidget(self.table, 1)

        self.load_button = QPushButton("Load selected series")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self._load_selected)
        root.addWidget(self.load_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.status = QLabel("Choose a DICOM folder.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.status)

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select DICOM directory", self.path_edit.text()
        )
        if directory:
            self.path_edit.setText(directory)
            self._scan()

    def _scan(self) -> None:
        if self._worker is not None:
            return
        directory = Path(self.path_edit.text().strip()).expanduser()
        if not directory.is_dir():
            self._fail(ValueError("Select an existing DICOM directory"))
            return
        self._set_busy(True, "Scanning DICOM headers…")
        worker = create_worker(
            discover_dicom_series,
            directory,
            recursive=self.recursive_check.isChecked(),
        )
        self._worker = worker
        worker.returned.connect(self._on_discovered)
        worker.errored.connect(self._fail)
        worker.finished.connect(self._finished)
        worker.start()

    def _on_discovered(self, series: list[DicomSeries]) -> None:
        self._series = series
        self.table.setRowCount(len(series))
        root = Path(self.path_edit.text().strip()).expanduser().resolve()
        for row, item in enumerate(series):
            try:
                folder = str(item.directory.relative_to(root)) or "."
            except ValueError:
                folder = item.directory.name
            values = (
                item.series_number or "—",
                item.modality or "—",
                item.series_description or item.study_description or "—",
                str(len(item.files)),
                folder,
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        if series:
            self.table.selectRow(0)
            self.load_button.setEnabled(True)
            self.status.setText(
                f"Found {len(series)} DICOM series. Select one to load."
            )
        else:
            self.load_button.setEnabled(False)
            self.status.setText("No DICOM image series found in this folder.")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

    def _load_selected(self, *_args) -> None:
        if self._worker is not None:
            return
        row = self.table.currentRow()
        if row < 0 or row >= len(self._series):
            self._fail(ValueError("Select a DICOM series first"))
            return
        selected = self._series[row]
        self._set_busy(True, f"Loading {len(selected.files)} DICOM instances…")
        worker = create_worker(read_dicom_series, selected)
        self._worker = worker
        worker.returned.connect(self._on_loaded)
        worker.errored.connect(self._fail)
        worker.finished.connect(self._finished)
        worker.start()

    def _on_loaded(self, volume: LoadedDicomVolume) -> None:
        data, attributes, _layer_type = volume.layer_data()
        layer = self.viewer.add_image(data, **attributes)
        self.viewer.layers.selection.active = layer
        self.status.setText(
            f"Loaded {volume.name}: {data.shape[0]} slices, shape {tuple(data.shape)}."
        )
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        show_info(f"DICOM series loaded: {volume.name}")

    def _fail(self, error) -> None:
        message = str(error) or type(error).__name__
        self.status.setText(message)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        show_error(f"DICOM: {message}")

    def _finished(self) -> None:
        self._worker = None
        self.scan_button.setEnabled(True)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

    def _set_busy(self, busy: bool, message: str) -> None:
        self.scan_button.setEnabled(not busy)
        self.load_button.setEnabled(not busy and bool(self._series))
        self.progress.setRange(0, 0 if busy else 100)
        self.status.setText(message)


__all__ = ["DicomSeriesWidget"]
