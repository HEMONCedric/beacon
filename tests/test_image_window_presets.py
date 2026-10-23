import unittest
from math import prod
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from napari.components import ViewerModel
from napari.layers import Image, Labels
from napari._qt.layer_controls.qt_layer_controls_container import create_qt_layer_controls
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QComboBox

from napari_beacon_layers import FixedImageLayer
from napari_beacon_layers.controls.image_window_controls import (
    CT_WINDOWS, MAX_WINDOW_SAMPLES, MR_PERCENTILES, QtBeaconImageControls,
    percentile_window, sampled_intensities,
)


class WindowPresetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def controls(self, layer):
        control = create_qt_layer_controls(layer)
        self.addCleanup(control.close)
        return control

    def select(self, controls, name):
        combo = controls._window_preset_control.combo
        index = combo.findText(name)
        self.assertGreaterEqual(index, 0)
        combo.setCurrentIndex(index)
        combo.activated.emit(index)

    def test_registered_for_normal_and_fixed_images_without_duplicate_rows(self):
        for layer in (Image(np.zeros((4, 5, 6))), FixedImageLayer(np.zeros((4, 5, 6)))):
            controls = self.controls(layer)
            self.assertIsInstance(controls, QtBeaconImageControls)
            combo = controls._window_preset_control.combo
            self.assertEqual(len(controls.findChildren(QComboBox, "beacon_window_preset")), 1)
            contrast_row, _ = controls.layout().getWidgetPosition(
                controls._contrast_limits_control.contrast_limits_slider,
            )
            preset_row, _ = controls.layout().getWidgetPosition(combo)
            self.assertEqual(preset_row, contrast_row + 1)
            self.assertEqual(combo.currentText(), "Original")
        labels = self.controls(Labels(np.zeros((4, 5, 6), dtype=np.uint8)))
        self.assertFalse(hasattr(labels, "_window_preset_control"))

    def test_all_ct_presets_change_only_display_and_do_not_clamp_slider(self):
        data = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
        original = data.copy()
        layer = Image(data, scale=(3, 0.8, 0.8), translate=(10, 20, 30),
                      contrast_limits=(5, 95), gamma=1.2, colormap="magma")
        controls = self.controls(layer)
        initial_range = tuple(layer.contrast_limits_range)
        initial_matrix = layer._data_to_world.affine_matrix.copy()
        for name, (level, window) in CT_WINDOWS.items():
            self.select(controls, name)
            np.testing.assert_allclose(layer.contrast_limits,
                                       (level - window / 2, level + window / 2))
            self.assertEqual(controls._window_preset_control.combo.currentText(), name)
            self.assertLessEqual(layer.contrast_limits_range[0], initial_range[0])
            self.assertGreaterEqual(layer.contrast_limits_range[1], initial_range[1])
        self.assertIs(layer.data, data)
        np.testing.assert_array_equal(layer.data, original)
        np.testing.assert_allclose(layer._data_to_world.affine_matrix, initial_matrix)
        self.assertEqual(layer.gamma, 1.2)
        self.assertEqual(layer.colormap.name, "magma")
        self.assertEqual(CT_WINDOWS["CT: Lung"], (-500, 1400))

    def test_mr_presets_follow_intensity_scale_and_exclude_zero_background(self):
        tissue = np.arange(1, 101, dtype=float)
        data = np.zeros((10, 10, 10))
        data[0] = tissue.reshape(10, 10)
        for factor in (1, 100):
            layer = Image(data * factor)
            controls = self.controls(layer)
            for name, percentiles in MR_PERCENTILES.items():
                self.select(controls, name)
                np.testing.assert_allclose(layer.contrast_limits,
                                           np.percentile(tissue * factor, percentiles))

    def test_auto_original_manual_and_external_contrast_changes(self):
        layer = Image(np.arange(1000.).reshape(10, 10, 10), contrast_limits=(10, 200))
        controls = self.controls(layer)
        self.select(controls, "Auto (0.5-99.5%)")
        np.testing.assert_allclose(layer.contrast_limits, np.percentile(layer.data, [0.5, 99.5]))
        self.select(controls, "CT: Brain")
        self.select(controls, "Manual")
        np.testing.assert_allclose(layer.contrast_limits, (0, 100))
        self.select(controls, "Original")
        np.testing.assert_allclose(layer.contrast_limits, (10, 200))
        self.select(controls, "CT: Lung")
        layer.contrast_limits = (25, 75)
        self.assertEqual(controls._window_preset_control.combo.currentText(), "Manual")
        np.testing.assert_allclose(layer.contrast_limits, (25, 75))
        self.select(controls, "CT: Lung")
        layer.data = layer.data + 10
        self.assertEqual(controls._window_preset_control.combo.currentText(), "Manual")

    def test_preset_stops_continuous_autocontrast(self):
        layer = Image(np.arange(120.).reshape(4, 5, 6))
        controls = self.controls(layer)
        button = controls._contrast_limits_control.auto_scale_bar._auto_btn
        button.setChecked(True)
        self.assertTrue(layer._keep_auto_contrast)
        self.select(controls, "CT: Bone")
        self.assertFalse(button.isChecked())
        self.assertFalse(layer._keep_auto_contrast)

    def test_layer_selection_keeps_independent_presets(self):
        viewer = ViewerModel()
        first = viewer.add_image(np.zeros((4, 5, 6)))
        second = viewer.add_image(np.ones((4, 5, 6)))
        a, b = self.controls(first), self.controls(second)
        self.select(a, "CT: Lung")
        self.select(b, "CT: Brain")
        viewer.layers.selection.active = first
        self.assertEqual(a._window_preset_control.combo.currentText(), "CT: Lung")
        viewer.layers.selection.active = second
        self.assertEqual(b._window_preset_control.combo.currentText(), "CT: Brain")
        np.testing.assert_allclose(first.contrast_limits, (-1200, 200))
        np.testing.assert_allclose(second.contrast_limits, (0, 100))

    def test_empty_constant_and_nonfinite_images(self):
        for value in (0, 3, -3):
            data = np.full((4, 5, 6), value, dtype=float)
            layer = Image(data)
            low, high = percentile_window(layer, (1, 99), ignore_zero=True)
            self.assertTrue(np.isfinite([low, high]).all())
            self.assertLess(low, value)
            self.assertGreater(high, value)
        layer = Image(np.arange(120.).reshape(4, 5, 6))
        layer.data[0] = np.nan
        layer.data[1, 0] = np.inf
        limits = percentile_window(layer, (1, 99), ignore_zero=True)
        self.assertTrue(np.isfinite(limits).all())
        controls = self.controls(layer)
        before = tuple(layer.contrast_limits)
        layer.data[:] = np.nan
        with patch("napari_beacon_layers.controls.image_window_controls.show_warning") as warning:
            self.select(controls, "MR: Standard (1-99%)")
        warning.assert_called_once()
        self.assertEqual(tuple(layer.contrast_limits), before)
        self.assertEqual(controls._window_preset_control.combo.currentText(), "Manual")

    def test_lazy_volume_is_sampled_before_conversion_and_is_bounded(self):
        class LazyArray:
            shape = (1, 1, 100_000_000)

            def __array__(self, *args, **kwargs):
                raise AssertionError("Must not materialize the entire volume")

            def __getitem__(self, slices):
                self.sample_shape = tuple(len(range(*part.indices(size)))
                                          for size, part in zip(self.shape, slices))
                return np.ones(self.sample_shape, dtype=np.float32)

        source = LazyArray()
        values = sampled_intensities(SimpleNamespace(data=source, multiscale=False))
        self.assertLessEqual(values.size, MAX_WINDOW_SAMPLES)
        self.assertEqual(values.size, prod(source.sample_shape))

    def test_rgb_disabled_and_multiscale_supported(self):
        rgb = Image(np.zeros((5, 6, 3)), rgb=True)
        controls = self.controls(rgb)
        self.assertFalse(controls._window_preset_control.combo.isEnabled())
        initial = tuple(rgb.contrast_limits)
        self.select(controls, "CT: Lung")
        self.assertEqual(tuple(rgb.contrast_limits), initial)
        levels = [np.arange(512.).reshape(8, 8, 8), np.arange(64.).reshape(4, 4, 4)]
        layer = Image(levels, multiscale=True, rgb=False)
        np.testing.assert_array_equal(np.sort(sampled_intensities(layer)), np.arange(64.))

    def test_deleting_controls_disconnects_its_layer_callbacks(self):
        layer = Image(np.zeros((4, 5, 6)))
        controls = create_qt_layer_controls(layer)
        wrapper = controls._window_preset_control
        def connected(emitter):
            return any(isinstance(callback, tuple) and callback[0]() is wrapper
                       for callback in emitter.callbacks)
        self.assertTrue(connected(layer.events.contrast_limits))
        controls.deleteLater()
        self.assertFalse(connected(layer.events.contrast_limits))
        self.assertFalse(connected(layer.events.data))
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


if __name__ == "__main__":
    unittest.main()
