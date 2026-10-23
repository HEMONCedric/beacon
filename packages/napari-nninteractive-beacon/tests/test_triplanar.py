from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import SimpleITK as sitk
from napari.components import ViewerModel
from napari_beacon_dicom.volume_reader import read_medical_volume
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication

from napari_nninteractive_beacon.volume_geometry import (
    LassoPrompt, image_from_layer, predict_three_lassos, prepare_layer, prepare_volume,
)
from napari_nninteractive_beacon.widget_triplanar import nnInteractiveWidgetTriplanar


class FakeSession:
    supported_interactions = {"lasso": True, "points": True, "bbox2d": True, "scribble": True}
    preferred_scribble_thickness = (1, 1, 1)
    license = "test"

    def __init__(self):
        self.calls = []
        self.target = None

    def set_image(self, image, properties):
        self.image = image
        self.properties = properties

    def set_target_buffer(self, target):
        self.target = target

    def set_do_autozoom(self, enabled):
        pass

    def reset_interactions(self):
        self.calls.clear()

    def add_lasso_interaction(self, mask, positive, run_prediction=True, interaction_bbox=None):
        self.calls.append((mask.copy(), positive, run_prediction, interaction_bbox))
        if run_prediction:
            self.target[1:-1, 1:-1, 1:-1] = 1


class GeometryTests(unittest.TestCase):
    def test_isotropic_working_grid_and_export_preserve_anatomy(self):
        image = sitk.GetImageFromArray(np.ones((5, 7, 9), dtype=np.float32))
        image.SetSpacing((0.5, 1.0, 3.0))
        image.SetOrigin((21.0, -13.0, 8.0))
        # Sagittal/flipped acquisition axes must survive resampling unchanged.
        image.SetDirection((0, 0, -1, 0, 1, 0, 1, 0, 0))
        prepared = prepare_volume(image, "sagittal")
        self.assertEqual(prepared.working.GetSpacing(), (1, 1, 1))
        np.testing.assert_allclose(prepared.working.GetDirection(), image.GetDirection())
        self.assertEqual(prepared.working.GetOrigin(), image.GetOrigin())
        self.assertEqual(prepared.working.GetSize(), (5, 7, 13))
        self.assertEqual(prepared.plane_names, ("Sagittal", "Coronal", "Axial"))
        np.testing.assert_allclose(prepared.display_matrix_zyx, np.eye(4))
        mask = np.zeros(prepared.data.shape, dtype=np.uint8)
        # Mark one physical point and verify its voxel on the original grid.
        original_index = (4, 3, 2)
        physical = image.TransformIndexToPhysicalPoint(original_index)
        work_index = prepared.working.TransformPhysicalPointToIndex(physical)
        mask[tuple(work_index[::-1])] = 1
        exported = prepared.mask_in_original_space(mask)
        self.assertEqual(exported.GetSize(), image.GetSize())
        self.assertEqual(exported.GetSpacing(), image.GetSpacing())
        self.assertEqual(exported.GetOrigin(), image.GetOrigin())
        self.assertEqual(exported.GetDirection(), image.GetDirection())
        self.assertEqual(exported[original_index], 1)

    def test_active_dicom_layer_uses_scanner_geometry(self):
        metadata = {"dicom": {
            "spacing_xyz_mm": (0.7, 0.8, 2.5),
            "origin_xyz_lps_mm": (10, 20, 30),
            "direction_xyz_lps": tuple(np.eye(3).ravel()),
        }}
        image = image_from_layer(np.zeros((3, 4, 5)), np.eye(4), metadata)
        self.assertEqual(image.GetSpacing(), (0.7, 0.8, 2.5))
        self.assertEqual(image.GetOrigin(), (10, 20, 30))

    def test_active_dicom_retains_display_frame_separately_from_patient_frame(self):
        spacing_xyz = (1.074, 1.074, 3.0)
        matrix = np.diag((*spacing_xyz[::-1], 1.0))
        matrix[:3, 3] = (12, 30, 40)
        metadata = {"dicom": {
            "spacing_xyz_mm": spacing_xyz,
            "origin_xyz_lps_mm": (-275, -275, -1572.5),
            "direction_xyz_lps": tuple(np.eye(3).ravel()),
        }}
        volume = prepare_layer(np.ones((5, 7, 9)), matrix, metadata, "test CT")
        self.assertEqual(volume.working.GetOrigin(), (-275, -275, -1572.5))
        for index in ((0, 0, 0), (2, 3, 4), (4, 6, 8)):
            source = np.array((*index, 1.0))
            work = np.array((*(np.array(index) * np.array(spacing_xyz[::-1])), 1.0))
            np.testing.assert_allclose(matrix @ source, volume.display_matrix_zyx @ work)
            np.testing.assert_allclose(
                volume.original.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in index[::-1])),
                volume.working.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in work[:3][::-1])),
            )

    def test_oblique_resampling_retains_directions_and_linear_voxel_values(self):
        z, y, x = np.indices((5, 7, 9))
        image = sitk.GetImageFromArray((100 * z + 10 * y + x).astype(np.float32))
        image.SetSpacing((0.5, 1, 3))
        image.SetOrigin((123, -50, 99))
        angle = np.deg2rad(20)
        direction = (1, 0, 0, 0, np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle))
        image.SetDirection(direction)
        volume = prepare_volume(image, "oblique")
        self.assertEqual(volume.working.GetSize(), (5, 7, 13))
        np.testing.assert_allclose(volume.working.GetDirection(), direction)
        self.assertEqual(volume.working.GetOrigin(), image.GetOrigin())
        # Working xyz=(2,3,6) corresponds to native xyz=(4,3,2).
        self.assertAlmostEqual(float(volume.data[6, 3, 2]), 234, places=4)

    def test_layer_transform_including_negative_spacing_is_preserved(self):
        matrix = np.diag((-2.5, 0.8, 0.7, 1.0))
        matrix[:3, 3] = (5, 10, 20)
        image = image_from_layer(np.zeros((3, 4, 5)), matrix, {})
        np.testing.assert_allclose(image.TransformIndexToPhysicalPoint((2, 1, 1)), (21.4, 10.8, 2.5))

    def test_active_nifti_recovers_header_geometry_in_both_array_orders(self):
        data = np.arange(5 * 7 * 9, dtype=np.float32).reshape((5, 7, 9))
        image = sitk.GetImageFromArray(data)
        image.SetSpacing((0.5, 0.8, 3))
        image.SetOrigin((3, 4, 5))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "image.nii.gz")
            sitk.WriteImage(image, path)
            for array in (data, data.transpose(2, 1, 0)):
                volume = prepare_layer(array, np.eye(4), {}, "active", path)
                np.testing.assert_allclose(volume.original.GetSpacing(), image.GetSpacing())
                self.assertEqual(volume.original.GetSize(), image.GetSize())
                physical = image.TransformIndexToPhysicalPoint((4, 3, 2))
                index = volume.working.TransformPhysicalPointToIndex(physical)
                mask = np.zeros(volume.data.shape, dtype=np.uint8)
                mask[tuple(slice(max(0, x - 1), x + 2) for x in index[::-1])] = 1
                self.assertEqual(volume.mask_in_original_space(mask)[4, 3, 2], 1)

    def test_three_prompts_are_accumulated_before_a_single_prediction(self):
        session = FakeSession()
        prompts = [
            LassoPrompt(0, np.ones((1, 2, 2), np.uint8), [[2, 3], [1, 3], [1, 3]]),
            LassoPrompt(1, np.ones((2, 1, 2), np.uint8), [[1, 3], [2, 3], [1, 3]]),
            LassoPrompt(2, np.ones((2, 2, 1), np.uint8), [[1, 3], [1, 3], [2, 3]]),
        ]
        with self.assertRaises(ValueError):
            predict_three_lassos(session, prompts[:2], (5, 5, 5))
        result = predict_three_lassos(session, prompts, (5, 5, 5))
        self.assertEqual([call[2] for call in session.calls], [False, False, True])
        self.assertTrue(all(call[1] for call in session.calls))
        self.assertTrue(np.any(result))


def wait_worker(widget, attribute):
    worker = getattr(widget, attribute)
    if worker is None:
        return
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    worker.finished.connect(loop.quit)
    timer.start(5000)
    loop.exec()
    timer.stop()
    if getattr(widget, attribute) is not None and attribute == "_task_worker":
        raise AssertionError("Worker did not finish")


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.patches = [
            patch("nnInteractive.model_management.list_models", return_value=[]),
            patch("nnInteractive.model_management.get_default_model_id", return_value="test"),
            patch("napari_nninteractive.widget_gui.VersionChecker.start"),
            patch("napari_nninteractive.widget_controls.LayerControls._refresh_prompt_cursor"),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.viewer = ViewerModel()
        object.__setattr__(self.viewer, "window", SimpleNamespace(
            _qt_viewer=MagicMock(), _qt_window=MagicMock(),
        ))
        self.widget = nnInteractiveWidgetTriplanar(self.viewer)
        self.widget._pending_original_layer = None
        image = sitk.GetImageFromArray(np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11))
        image.SetSpacing((0.8, 0.9, 2.0))
        image.SetOrigin((10, 20, 30))
        self.volume = prepare_volume(image, "test")
        self.widget._on_prepared(self.volume)
        self.widget._task_finished()

    def initialize(self, remote):
        widget = self.widget
        widget._remote_mode = remote
        widget._remote_connected = remote
        self.session = FakeSession()
        if remote:
            widget.session = self.session
            widget.on_init()
            wait_worker(widget, "_worker")
        else:
            with tempfile.TemporaryDirectory() as checkpoint:
                widget.model_selection_local.blockSignals(True)
                widget.model_selection_local.setText(checkpoint)
                widget.model_selection_local.blockSignals(False)
                with patch.object(widget, "_construct_local_session", side_effect=lambda: setattr(widget, "session", self.session)):
                    widget.on_init()
        self.assertTrue(widget._session_ready)
        np.testing.assert_allclose(self.session.properties["spacing"], (1, 1, 1))

    def draw(self, axis):
        widget = self.widget
        widget._select_plane(axis)
        shape = widget._prepared.data.shape
        index = shape[axis] // 2
        widget._set_slice(axis, index)
        others = [i for i in range(3) if i != axis]
        vertices = []
        for a, b in ((1, 1), (shape[others[0]] - 2, 1),
                     (shape[others[0]] - 2, shape[others[1]] - 2),
                     (1, shape[others[1]] - 2)):
            vertex = [0., 0., 0.]
            vertex[axis], vertex[others[0]], vertex[others[1]] = index, a, b
            vertices.append(vertex)
        layer = self.viewer.layers[widget.lasso_layer_name]
        layer.add(np.array(vertices), shape_type="polygon")
        QApplication.processEvents()
        self.assertIn(axis, widget._prompts)

    def test_mouse_drawing_finishes_before_plane_switch(self):
        from napari.layers.shapes._shapes_mouse_bindings import initiate_polygon_draw

        self.initialize(remote=False)
        layer = self.viewer.layers[self.widget.lasso_layer_name]
        self.widget._select_plane(0)
        self.viewer.dims.set_point(0, self.widget._working_layer.translate[0] + 3)
        vertices = np.array([[3, 1, 1], [3, 5, 1], [3, 5, 5], [3, 1, 5], [3, 1, 1]], dtype=float)
        initiate_polygon_draw(layer, vertices[0])
        layer._data_view.edit(0, vertices)
        layer._moving_value = (0, len(vertices) - 1)
        layer._finish_drawing()
        self.assertFalse(layer._is_creating)
        self.assertNotIn(0, self.widget._prompts)
        QApplication.processEvents()
        self.assertIn(0, self.widget._prompts)
        self.assertEqual(len(layer.data), 0)
        self.assertFalse(layer.selected_data)
        self.assertEqual(self.viewer.dims.not_displayed, (1,))

    def test_reference_traces_are_bright_red_and_single_slice(self):
        self.initialize(remote=False)
        overlay = self.widget._prompt_layer
        np.testing.assert_allclose(overlay.colormap.map(np.array([0, 1])),
                                   [[0, 0, 0, 0], [1, 0.12, 0.12, 1]])
        self.assertAlmostEqual(overlay.opacity, 0.65)
        expected = np.zeros_like(overlay.data)
        for axis in range(3):
            self.draw(axis)
            prompt = self.widget._prompts[axis]
            lo, hi = prompt.bbox[axis]
            self.assertEqual(hi - lo, 1)
            self.assertEqual(prompt.crop.shape[axis], 1)
            slices = tuple(slice(a, b) for a, b in prompt.bbox)
            expected[slices] |= prompt.crop
            np.testing.assert_array_equal(overlay.data, expected)
        prompts = [prompt.crop.copy() for prompt in self.widget._prompts.values()]
        self.widget._generate()
        wait_worker(self.widget, "_task_worker")
        for call, original in zip(self.session.calls, prompts, strict=True):
            np.testing.assert_array_equal(call[0], original)

    def test_reference_traces_stay_single_slice_at_both_volume_edges(self):
        self.initialize(remote=False)
        shape = self.widget._prepared.data.shape
        for axis in range(3):
            for index in (0, shape[axis] - 1):
                bbox = [[1, 3], [1, 3], [1, 3]]
                bbox[axis] = [index, index + 1]
                crop = np.ones(tuple(hi - lo for lo, hi in bbox), dtype=np.uint8)
                self.widget._prompts.clear()
                self.widget._prompts[axis] = LassoPrompt(axis, crop, bbox)
                self.widget._refresh_prompt_masks()
                self.assertEqual(np.count_nonzero(self.widget._prompt_layer.data), 4)
                self.assertEqual(np.count_nonzero(crop), 4)
                self.assertEqual(np.count_nonzero(np.take(self.widget._prompt_layer.data, index, axis=axis)), 4)
                adjacent = 1 if index == 0 else index - 1
                self.assertFalse(np.any(np.take(self.widget._prompt_layer.data, adjacent, axis=axis)))
                opposite = -1 if index == 0 else 0
                self.assertFalse(np.any(np.take(self.widget._prompt_layer.data, opposite, axis=axis)))

    def test_local_lassos_generate_undo_refine_and_export(self):
        self.initialize(remote=False)
        for axis in range(3):
            self.draw(axis)
            self.assertEqual(self.widget.generate_button.isEnabled(), axis == 2)
        self.assertEqual(len(self.session.calls), 0)
        self.widget._generate()
        wait_worker(self.widget, "_task_worker")
        self.assertTrue(self.widget._generated)
        self.assertEqual([call[2] for call in self.session.calls], [False, False, True])
        result = self.widget._result_layer()
        self.assertTrue(np.any(result.data))
        self.widget.on_undo()
        self.assertFalse(self.widget._generated)
        self.assertFalse(np.any(result.data))
        self.assertEqual(len(self.widget._prompts), 3)
        self.widget._generate()
        wait_worker(self.widget, "_task_worker")
        self.widget.on_refine_manually()
        self.assertTrue(result.editable)
        self.assertEqual(result.mode, "paint")
        self.assertFalse(result.autofill)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "mask.nii.gz")
            self.widget._write_mask(self.volume, result.data, path)
            exported = sitk.ReadImage(path)
            self.assertEqual(exported.GetSize(), self.volume.original.GetSize())
            np.testing.assert_allclose(exported.GetSpacing(), self.volume.original.GetSpacing())
            np.testing.assert_allclose(exported.GetOrigin(), self.volume.original.GetOrigin())
            data, attributes, kind = read_medical_volume(path, viewer=self.viewer)[0]
            imported = self.viewer.add_labels(data, **attributes)
            self.assertEqual(kind, "labels")
            source_index = (2, 3, 4)
            physical = self.volume.original.TransformIndexToPhysicalPoint(source_index[::-1])
            work_xyz = self.volume.working.TransformPhysicalPointToContinuousIndex(physical)
            np.testing.assert_allclose(
                imported.data_to_world(source_index),
                self.widget._working_layer.data_to_world(work_xyz[::-1]),
            )

    def test_remote_upload_then_lasso_replacement_and_undo(self):
        self.initialize(remote=True)
        self.draw(0)
        self.draw(1)
        self.draw(0)
        self.assertEqual(len(self.widget._prompts), 2)
        self.widget.on_undo()
        self.assertNotIn(0, self.widget._prompts)
        self.assertIn(1, self.widget._prompts)
        self.draw(0)
        self.draw(2)
        self.widget._generate()
        wait_worker(self.widget, "_task_worker")
        self.assertTrue(self.widget._generated)
        self.assertTrue(self.widget.reset_button.isHidden())
        self.assertTrue(self.widget._group(self.widget.prompt_button).isHidden())
        for name in (self.widget.point_layer_name, self.widget.bbox_layer_name, self.widget.scribble_layer_name):
            self.assertNotIn(name, self.viewer.layers)

    def test_new_image_reuses_layers_with_new_geometry(self):
        self.initialize(remote=True)
        for axis in range(3):
            self.draw(axis)
        self.widget._generate()
        wait_worker(self.widget, "_task_worker")
        result = self.widget._result_layer()
        lasso = self.viewer.layers[self.widget.lasso_layer_name]
        image = sitk.GetImageFromArray(np.ones((9, 11, 13), dtype=np.float32))
        image.SetOrigin((-20, 40, 100))
        volume = prepare_volume(image, "new image")
        self.widget._on_prepared(volume)
        self.widget._task_finished()
        self.assertFalse(self.widget._generated)
        self.assertEqual(len(self.widget._prompts), 0)
        self.initialize(remote=True)
        self.assertIs(result, self.widget._result_layer())
        self.assertIs(lasso, self.viewer.layers[self.widget.lasso_layer_name])
        self.assertEqual(lasso._shape, volume.data.shape)
        np.testing.assert_allclose(lasso._data_to_world.affine_matrix, volume.display_matrix_zyx)
        self.assertFalse(np.any(result.data))
        for axis in range(3):
            self.draw(axis)
        self.widget._generate()
        wait_worker(self.widget, "_task_worker")
        self.assertTrue(self.widget._generated)

    def test_active_sagittal_dicom_navigation_uses_the_working_grid(self):
        original_data = np.ones((5, 7, 9), dtype=np.float32)
        source = self.viewer.add_image(
            original_data, scale=(3, 1, 0.5), metadata={"dicom": {
                "spacing_xyz_mm": (0.5, 1, 3),
                "origin_xyz_lps_mm": (-130, 210, -300),
                "direction_xyz_lps": (0, 0, -1, 0, 1, 0, 1, 0, 0),
            }},
        )
        self.viewer.layers.selection.active = source
        self.widget._use_active_image()
        wait_worker(self.widget, "_task_worker")
        volume = self.widget._prepared
        origin = volume.layer_transform()["translate"]
        expected = [(start, start + size - 1, 1)
                    for start, size in zip(origin, volume.data.shape)]
        np.testing.assert_allclose(self.viewer.dims.range, expected)
        self.assertFalse(source.visible)
        self.assertIs(source.data, original_data)

    def test_reflected_active_layer_stays_aligned_and_backend_receives_one_mm(self):
        source = self.viewer.add_image(
            np.ones((5, 7, 9), dtype=np.float32),
            scale=(-3, 0.8, 0.7), translate=(100, 20, 30),
        )
        self.viewer.layers.selection.active = source
        self.widget._use_active_image()
        wait_worker(self.widget, "_task_worker")
        self.initialize(remote=True)
        work = self.widget._working_layer
        np.testing.assert_allclose(source.data_to_world((2, 3, 4)), work.data_to_world((6, 2.4, 2.8)))
        np.testing.assert_allclose(self.widget._result_layer()._data_to_world.affine_matrix,
                                   work._data_to_world.affine_matrix)
        for axis in range(3):
            self.draw(axis)
        self.widget._generate()
        wait_worker(self.widget, "_task_worker")
        self.assertTrue(self.widget._generated)

    def test_backend_spacing_is_physical_even_when_source_viewer_units_are_pixels(self):
        data = np.arange(5 * 7 * 9, dtype=np.float32).reshape((5, 7, 9))
        image = sitk.GetImageFromArray(data)
        image.SetSpacing((0.5, 0.8, 3))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "source.nii.gz")
            sitk.WriteImage(image, path)
            volume = prepare_layer(data, np.eye(4), {}, "active", path)
        self.widget._on_prepared(volume)
        self.widget._task_finished()
        self.initialize(remote=False)
        np.testing.assert_allclose(self.widget._working_layer.scale, (1 / 3, 1 / 0.8, 1 / 0.5))
        # initialize() also asserts that the actual backend receives (1,1,1).
        np.testing.assert_allclose(self.session.properties["spacing"], (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
