"""Display-only window/level presets shared by all BEACON image layers."""

from math import prod

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox

from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls
from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase, QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Image
from napari.utils.notifications import show_warning


# (level, window) in HU, not raw stored DICOM values. The four core CT
# presets use Slicer's VolumeDisplayPresets.json defaults:
# https://github.com/Slicer/Slicer/blob/main/Modules/Loadable/Volumes/Resources/VolumeDisplayPresets.json
CT_WINDOWS = {
    "CT: Abdomen": (40.0, 350.0),
    "CT: Bone": (400.0, 1000.0),
    "CT: Brain": (50.0, 100.0),
    "CT: Lung": (-500.0, 1400.0),
    "CT: Mediastinum": (40.0, 400.0),
    "CT: Soft tissue": (40.0, 400.0),
    # Liver uses the OHIF CT preset; the remaining extended presets retain
    # the values previously provided by BEACON's FixedImageLayer controls.
    "CT: Liver": (90.0, 150.0),
    "CT: Dental": (400.0, 2000.0),
    "CT: Inner ear": (700.0, 4000.0),
    "CT: Larynx": (40.0, 250.0),
    "CT: Pelvis": (250.0, 1000.0),
    "CT: Spine": (35.0, 300.0),
    "CT: Vertebrae": (350.0, 2000.0),
}
MR_PERCENTILES = {
    "MR: Standard (1-99%)": (1.0, 99.0),
    "MR: High contrast (5-95%)": (5.0, 95.0),
    "MR: Wide (0.1-99.9%)": (0.1, 99.9),
}
MAX_WINDOW_SAMPLES = 262_144


def sampled_intensities(layer: Image) -> np.ndarray:
    """Read a bounded spatial sample, including from lazy/multiscale data."""
    source = layer.data[-1] if layer.multiscale else layer.data
    shape = tuple(int(n) for n in source.shape)
    steps = [1] * len(shape)
    sampled_shape = list(shape)
    while prod(sampled_shape) > MAX_WINDOW_SAMPLES:
        axis = int(np.argmax(sampled_shape))
        steps[axis] *= 2
        sampled_shape[axis] = (shape[axis] + steps[axis] - 1) // steps[axis]
    values = np.asarray(source[tuple(slice(None, None, step) for step in steps)])
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("Window presets require a real-valued scalar image")
    values = values[np.isfinite(values)].astype(np.float64, copy=False)
    if not values.size:
        raise ValueError("The image sample contains no finite intensities")
    return values


def percentile_window(layer: Image, percentiles, *, ignore_zero=False) -> tuple[float, float]:
    values = sampled_intensities(layer)
    if ignore_zero and np.any(values != 0):
        values = values[values != 0]
    low, high = np.percentile(values, percentiles)
    if low == high:
        low, high = values.min(), values.max()
    if low == high:
        # A constant/empty-background image still needs a valid slider range.
        margin = max(abs(float(low)) * 1e-6, 0.5)
        low, high = low - margin, high + margin
    return float(low), float(high)


class ImageWindowPresetControl(QtWidgetControlsBase):
    def __init__(self, parent, layer):
        super().__init__(parent, layer)
        self._applying = False
        self._original_limits = tuple(layer.contrast_limits)
        self._preset_limits = self._original_limits
        self.combo = QComboBox(parent)
        self.combo.setObjectName("beacon_window_preset")
        self.combo.addItems(["Manual", "Original", "Auto (0.5-99.5%)"])
        self.combo.insertSeparator(self.combo.count())
        for name, (level, window) in CT_WINDOWS.items():
            self.combo.addItem(name)
            self.combo.setItemData(self.combo.count() - 1,
                                   f"CT in HU: W {window:g} / L {level:g}",
                                   Qt.ItemDataRole.ToolTipRole)
        self.combo.insertSeparator(self.combo.count())
        for name in MR_PERCENTILES:
            self.combo.addItem(name)
            self.combo.setItemData(
                self.combo.count() - 1,
                "Relative to finite, non-zero intensities in a bounded volume sample. "
                "Not a fixed T1/T2 calibration.", Qt.ItemDataRole.ToolTipRole,
            )
        self.combo.setCurrentText("Original")
        self.combo.setToolTip(
            "Display only. CT presets require HU. MR presets use image percentiles "
            "and exclude zero background. Auto includes background. "
            "Manual keeps the current window; Original restores its initial limits. "
            "No voxel values or geometry are changed."
        )
        self.combo.setEnabled(not layer.rgb)
        self.label = QtWrappedLabel("window preset:")
        # activated also fires when reselecting the same item (useful for Auto).
        self.combo.activated.connect(self._apply)
        layer.events.contrast_limits.connect(self._on_limits_change)
        layer.events.data.connect(self._on_data_change)

    def get_widget_controls(self):
        return [(self.label, self.combo)]

    def _manual(self):
        with qt_signals_blocked(self.combo):
            self.combo.setCurrentText("Manual")

    def _on_limits_change(self):
        if not self._applying and not np.allclose(
            self._layer.contrast_limits, self._preset_limits, rtol=1e-9, atol=1e-9,
        ):
            self._manual()

    def _on_data_change(self):
        self._original_limits = tuple(self._layer.contrast_limits)
        self.combo.setEnabled(not self._layer.rgb)
        self._manual()

    def _apply(self, _index=None):
        name = self.combo.currentText()
        if self._layer.rgb or name == "Manual":
            return
        try:
            if name in CT_WINDOWS:
                level, window = CT_WINDOWS[name]
                limits = (level - window / 2, level + window / 2)
            elif name in MR_PERCENTILES:
                limits = percentile_window(self._layer, MR_PERCENTILES[name], ignore_zero=True)
            elif name == "Auto (0.5-99.5%)":
                limits = percentile_window(self._layer, (0.5, 99.5))
            elif name == "Original":
                limits = self._original_limits
            else:
                return
        except (TypeError, ValueError, RuntimeError) as error:
            self._manual()
            show_warning(f"Cannot apply window preset: {error}")
            return
        self._applying = True
        try:
            # Otherwise napari's continuous auto-contrast overwrites a preset
            # on the next slice. Keep its existing button in sync as well.
            self.parent()._contrast_limits_control.auto_scale_bar._auto_btn.setChecked(False)
            self._layer._keep_auto_contrast = False
            self._layer.contrast_limits = limits
            self._preset_limits = tuple(self._layer.contrast_limits)
        finally:
            self._applying = False


class QtBeaconImageControls(QtImageControls):
    """Native image controls plus one normally styled window preset row."""

    def __init__(self, layer):
        super().__init__(layer)
        self._window_preset_control = ImageWindowPresetControl(self, layer)
        form = self.layout()
        contrast_row, _ = form.getWidgetPosition(
            self._contrast_limits_control.contrast_limits_slider,
        )
        form.insertRow(contrast_row + 1, *self._window_preset_control.get_widget_controls()[0])


# The package already registers controls for its custom layers. Register the
# scalar Image base too, so DICOM, file imports and working images share this UI.
layer_to_controls[Image] = QtBeaconImageControls
