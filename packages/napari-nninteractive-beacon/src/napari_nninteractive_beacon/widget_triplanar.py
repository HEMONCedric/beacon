"""One volume, three orthogonal lasso masks, one nnInteractive segmentation."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from napari.layers import Image, Labels
from napari.layers.base._base_constants import ActionType
from napari.qt.threading import create_worker
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.notifications import show_error, show_warning
from napari_beacon_dicom.volume_reader import SEGMENTATION_MARKER
from napari_nninteractive.widget_main import _SESSION_LOST_ERRORS
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QShortcut, QVBoxLayout, QWidget,
)

from .volume_geometry import (
    LassoPrompt, PreparedVolume, predict_three_lassos, prepare_layer, prepare_path,
)
from .widget_nninteractive import nnInteractiveWidgetBeacon


PLANES = ("Axial", "Coronal", "Sagittal")


class nnInteractiveWidgetTriplanar(nnInteractiveWidgetBeacon):
    """Classic's local/remote backend with a constrained three-plane workflow."""

    def __init__(self, napari_viewer, **kwargs: Any):
        self._prepared: PreparedVolume | None = None
        self._working_layer: Image | None = None
        self._original_layer: Image | None = None
        self._original_visible = True
        self._prompt_layer: Labels | None = None
        self._prompts: OrderedDict[int, LassoPrompt] = OrderedDict()
        self._task_worker = None
        self._initializing = False
        self._connecting = False
        self._session_ready = False
        self._generated = False
        self._ui_ready = False
        self._workflow_active = False
        self._restricting_ranges = False
        self._capture_pending = False
        self._plane_names = PLANES
        super().__init__(napari_viewer, **kwargs)

        self.label_layer_name = "BEACON 3-plane - Segmentation"
        self.lasso_layer_name = "BEACON 3-plane - Lasso"
        self.layer_dict = {3: self.lasso_layer_name}
        self.interaction_type = 3
        self.auto_run_ckbx.blockSignals(True)
        self.auto_run_ckbx.setChecked(False)
        self.auto_run_ckbx.blockSignals(False)

        # Keep Classic's Local/Remote, checkpoint, SSH and optional key controls.
        # Remove both the controls and shortcuts of the unsupported workflows.
        for widget in (
            self.prompt_button, self.interaction_button, self.label_for_init,
            self.auto_run_ckbx, self.aggregation_output_combo,
        ):
            self._hide_group(widget)
        for button in (self.reset_button, self.reset_interaction_button):
            button.hide()
            for shortcut in button.findChildren(QShortcut):
                shortcut.setEnabled(False)
        self.refine_manual_button.setToolTip("Edit the generated mask with Paint/Erase")
        self.export_button.setText("Export mask (original grid)")

        self._build_image_selection()
        self._build_lasso_workflow()
        self._ui_ready = True
        self._viewer.dims.events.range.connect(self._restrict_navigation)
        self._update_workflow()

    @staticmethod
    def _group(widget: QWidget) -> QGroupBox:
        parent = widget.parentWidget()
        while parent is not None and not isinstance(parent, QGroupBox):
            parent = parent.parentWidget()
        if parent is None:
            raise RuntimeError("The installed nnInteractive UI has changed")
        return parent

    def _hide_group(self, widget: QWidget) -> None:
        group = self._group(widget)
        group.hide()
        for shortcut in group.findChildren(QShortcut):
            shortcut.setEnabled(False)

    def _build_image_selection(self) -> None:
        group = self._group(self.image_selection)
        self.image_selection.hide()
        layout = group.layout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Image file path…")
        self.path_edit.returnPressed.connect(self._load_entered_path)
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._browse_image)
        row = QHBoxLayout()
        row.addWidget(self.path_edit, 1)
        row.addWidget(self.browse_button)
        layout.addLayout(row)
        self.active_image_button = QPushButton("Use active image layer")
        self.active_image_button.clicked.connect(self._use_active_image)
        layout.addWidget(self.active_image_button)
        self.image_info = QLabel("Choose a 3D image.")
        self.image_info.setWordWrap(True)
        layout.addWidget(self.image_info)

    def _build_lasso_workflow(self) -> None:
        group = QGroupBox("Three-plane lasso")
        layout = QVBoxLayout(group)
        instructions = QLabel(
            "Initialize, then draw one closed lasso around the same lesion in each plane. "
            "Drawing again in a plane replaces its mask."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        row = QHBoxLayout()
        self.plane_buttons = []
        for axis, name in enumerate(PLANES):
            button = QPushButton(name)
            button.clicked.connect(lambda _checked=False, axis=axis: self._select_plane(axis))
            self.plane_buttons.append(button)
            row.addWidget(button)
        layout.addLayout(row)
        self.lasso_status = QLabel("0 / 3 planes")
        self.lasso_status.setWordWrap(True)
        layout.addWidget(self.lasso_status)
        self.generate_button = QPushButton("Generate segmentation")
        self.generate_button.clicked.connect(self._generate)
        layout.addWidget(self.generate_button)
        self.task_status = QLabel("")
        self.task_status.setWordWrap(True)
        layout.addWidget(self.task_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)
        image_group = self._group(self.image_selection)
        workflow_layout = image_group.parentWidget().layout()
        model_group = self._group(self.mode_switch)
        model_group.setTitle("Local / Remote")
        workflow_layout.insertWidget(1, group)

    def _update_workflow(self) -> None:
        if not self._ui_ready:
            return
        busy = self._task_worker is not None or self._initializing or self._connecting
        drawing = self._session_ready and not self._generated and not busy
        for axis, button in enumerate(self.plane_buttons):
            button.setEnabled(drawing)
            button.setText(f"{self._plane_names[axis]} {'✓' if axis in self._prompts else ''}".strip())
        self.lasso_status.setText(
            f"{len(self._prompts)} / 3 planes — "
            + ", ".join(f"{name}: {'ready' if axis in self._prompts else 'missing'}"
                        for axis, name in enumerate(self._plane_names))
        )
        for widget in (self.path_edit, self.browse_button, self.active_image_button):
            widget.setEnabled(not busy)
        self.init_button.setEnabled(self._prepared is not None and not busy)
        self.generate_button.setEnabled(drawing and len(self._prompts) == 3)
        self.undo_button.setEnabled(not busy and (self._generated or bool(self._prompts)))
        self.refine_manual_button.setEnabled(
            self._generated and not self._manual_refinement_active and not busy
        )
        self.export_button.setEnabled(self._generated and not busy)
        for widget in (
            self.mode_switch, self.server_url_edit, self.api_key_edit,
            self.ssh_target_edit, self.connect_btn, self.model_selection,
            self.model_selection_local, self.propagate_ckbx,
            self.use_torch_compile_ckbx, self.interactions_storage_combo,
        ):
            widget.setEnabled(not busy)
        self.progress.setVisible(busy)
        result = self._result_layer()
        if result is not None:
            result.editable = self._manual_refinement_active and not busy
        if self.lasso_layer_name in self._viewer.layers:
            self._viewer.layers[self.lasso_layer_name].editable = drawing

    def _start_task(self, function, *args, message: str, returned) -> None:
        if self._task_worker is not None:
            return
        self.task_status.setText(message)
        worker = create_worker(
            function, *args, _start_thread=False, _ignore_errors=True,
        )
        self._task_worker = worker
        worker.returned.connect(returned)
        worker.errored.connect(self._task_error)
        worker.finished.connect(self._task_finished)
        self._update_workflow()
        worker.start()

    def _task_error(self, error: BaseException) -> None:
        if isinstance(error, _SESSION_LOST_ERRORS):
            self._handle_session_expired()
        self.task_status.setText(f"Failed: {error}")
        show_error(str(error))

    def _task_finished(self) -> None:
        self._task_worker = None
        self._update_workflow()

    def _confirm_new_image(self) -> bool:
        if self._task_worker is not None or self._initializing or self._connecting:
            return False
        if not self._generated and not self._prompts:
            return True
        return QMessageBox.question(
            self, "Choose a new image",
            "Replace the current lassos and segmentation? Export the mask first if you want to keep it.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a 3D image", "",
            "Medical images (*.nii *.nii.gz *.mha *.mhd *.nrrd *.dcm);;All files (*)",
        )
        if path:
            self.path_edit.setText(path)
            self._load_entered_path()

    def _load_entered_path(self) -> None:
        path = self.path_edit.text().strip().strip('"')
        if not path or not self._confirm_new_image():
            return
        self._pending_original_layer = None
        self._start_task(
            prepare_path, path, message="Reading and resampling to 1 mm…",
            returned=self._on_prepared,
        )

    def _use_active_image(self) -> None:
        layer = self._viewer.layers.selection.active
        if not isinstance(layer, Image) or layer.rgb or layer.ndim != 3 or layer.multiscale:
            show_warning("Select a scalar 3D image layer first")
            return
        if layer is self._working_layer:
            show_warning("This is already the 1 mm working image; select the original image")
            return
        if not self._confirm_new_image():
            return
        self._pending_original_layer = layer
        # Capture Qt-owned layer state here; workers receive only array/geometry.
        self._start_task(
            prepare_layer, layer.data, layer._data_to_world.affine_matrix.copy(),
            dict(layer.metadata), layer.name, layer.source.path,
            message="Resampling to 1 mm…",
            returned=self._on_prepared,
        )

    def _on_prepared(self, volume: PreparedVolume) -> None:
        self._workflow_active = False
        if self._original_layer is not None and self._original_layer in self._viewer.layers:
            self._original_layer.visible = self._original_visible
        self._original_layer = self._pending_original_layer
        if self._original_layer is not None:
            self._original_visible = bool(self._original_layer.visible)
            self._original_layer.visible = False
        self._prepared = volume
        self._plane_names = volume.plane_names
        self._session_ready = False
        self._generated = False
        self._manual_refinement_active = False
        self._prompts.clear()
        self._clear_layers()
        self._unlock_session()
        transform = volume.layer_transform()
        if self._working_layer is None or self._working_layer not in self._viewer.layers:
            self._working_layer = self._viewer.add_image(
                volume.data, name="BEACON 3-plane - Image (1 mm)",
                **transform, interpolation2d="spline36",
                metadata={"beacon_triplanar_working": True},
            )
        else:
            self._working_layer.data = volume.data
            self._apply_display_geometry(self._working_layer)
            self._working_layer.visible = True
            self._working_layer.reset_contrast_limits()
        if self._original_layer is not None:
            for name in ("colormap", "contrast_limits", "gamma", "opacity"):
                setattr(self._working_layer, name, getattr(self._original_layer, name))
        self._working_layer.metadata["beacon_export_reference"] = volume.original_display_reference()
        for layer in (self._prompt_layer, self._result_layer()):
            if layer is not None:
                layer.data = np.zeros(volume.data.shape, dtype=np.uint8)
                self._apply_display_geometry(layer)
                layer.visible = False
                layer._clear_extent()
        self.image_selection.blockSignals(True)
        self.image_selection.setCurrentText(self._working_layer.name)
        self.image_selection.blockSignals(False)
        self.image_info.setText(volume.name)
        self.task_status.setText("Image ready. Initialize the model, then draw the three lassos.")
        self.refine_manual_button.setText("Refine Manually")
        self._viewer.dims.ndisplay = 2
        self._viewer.dims.order = (0, 1, 2)
        self._workflow_active = True
        self._restrict_navigation()
        centre = self._working_layer.data_to_world((np.array(volume.data.shape) - 1) / 2)
        for axis in range(3):
            self._viewer.dims.set_point(axis, centre[axis])
        self._fit_working_plane()
        self._viewer.layers.selection.active = self._working_layer

    def _apply_display_geometry(self, layer) -> None:
        for name, value in self._prepared.layer_transform().items():
            setattr(layer, name, value)

    def _set_slice(self, axis: int, index: float) -> None:
        point = np.array(self._working_layer.world_to_data(self._viewer.dims.point))
        point[axis] = index
        world = self._working_layer.data_to_world(point)
        self._viewer.dims.set_point(axis, world[axis])

    def _restrict_navigation(self, *_args) -> None:
        if (not self._workflow_active or self._restricting_ranges
                or self._working_layer is None or self._prepared is None
                or self._viewer.dims.ndim != 3):
            return
        # Invisible source layers still influence napari's slider ranges. Keep
        # navigation on the selected working image, including native sagittal
        # DICOMs whose source display coordinates differ from patient space.
        extent = self._working_layer.extent
        ranges = tuple((float(lo), float(hi), float(step))
                       for lo, hi, step in zip(*extent.world, extent.step))
        if not np.allclose(self._viewer.dims.range, ranges):
            self._restricting_ranges = True
            try:
                self._viewer.dims.range = ranges
            finally:
                self._restricting_ranges = False

    def _fit_working_plane(self) -> None:
        displayed = self._viewer.dims.displayed
        centre = self._working_layer.extent.world.mean(axis=0)
        self._viewer.camera.center = (0, float(centre[displayed[0]]), float(centre[displayed[1]]))
        canvas = np.asarray(self._viewer._canvas_size, dtype=float)
        extent = self._working_layer._extent_augmented.world
        sizes = (extent[1] - extent[0])[list(displayed)]
        self._viewer.camera.zoom = 0.95 * float(np.min(canvas / sizes))

    def hideEvent(self, event):  # noqa: N802
        self._workflow_active = False
        self._viewer._on_layers_change()
        super().hideEvent(event)

    def showEvent(self, event):  # noqa: N802
        self._workflow_active = self._prepared is not None
        self._restrict_navigation()
        super().showEvent(event)

    def _result_layer(self):
        name = getattr(self, "label_layer_name", "")
        return self._viewer.layers[name] if name in self._viewer.layers else None

    def on_image_selected(self):
        # Classic's hidden layer selector is populated by napari on every image
        # insertion. Only the two explicit image-selection actions may reset us.
        pass

    def on_init(self, *args, **kwargs):
        if self._prepared is None or self._task_worker is not None:
            return
        self.image_selection.blockSignals(True)
        self.image_selection.setCurrentText(self._working_layer.name)
        self.image_selection.blockSignals(False)
        # Reuse the same label layer on reinitialization (also avoids repeated
        # VisPy resource destruction). An empty buffer does not seed prediction.
        layer = self._result_layer()
        if layer is not None and layer.data.shape == self._prepared.data.shape:
            self._resume_after_reconnect = True
            self._resume_image_layer = self._working_layer
        self._initializing = not self._remote_mode or self._remote_connected
        self._update_workflow()
        try:
            super().on_init(*args, **kwargs)
        except Exception as error:
            self._initializing = False
            self._session_ready = False
            self.task_status.setText(f"Initialization failed: {error}")
            show_error(str(error))
            self._update_workflow()

    def _finish_init(self) -> None:
        super()._finish_init()
        self._initializing = False
        if self.session is None:
            return
        if not self.session.supported_interactions.get("lasso", False):
            self._session_ready = False
            self.task_status.setText("This checkpoint does not support lasso prompts")
            self._update_workflow()
            return
        self._session_ready = True
        lasso = self._viewer.layers[self.lasso_layer_name]
        lasso._shape = self._prepared.data.shape
        self._apply_display_geometry(lasso)
        self.task_status.setText("Draw one lasso in each anatomical plane.")
        self._ensure_prompt_layer()
        if self._generated:
            self._viewer.layers[self.lasso_layer_name].editable = False
        else:
            self._select_plane(0)
        self._update_workflow()

    def _set_connecting(self, connecting: bool) -> None:
        self._connecting = connecting
        super()._set_connecting(connecting)
        self._update_workflow()

    def _on_upload_errored(self, error: BaseException) -> None:
        self._initializing = False
        super()._on_upload_errored(error)

    def _handle_session_expired(self) -> None:
        self._initializing = False
        super()._handle_session_expired()

    def add_lasso_layer(self) -> None:
        super().add_lasso_layer()
        layer = self._viewer.layers[self.lasso_layer_name]

        def ensure_cursor_position(layer, event):
            if (getattr(layer, "_is_creating", False)
                    and getattr(layer, "_last_cursor_position", None) is None):
                layer._last_cursor_position = np.array(event.pos)

        layer.mouse_move_callbacks.insert(0, ensure_cursor_position)

    def add_label_layer(self, data, name) -> None:
        super().add_label_layer(data, name)
        if name == self.label_layer_name:
            # Native Paint/Erase works in every plane; the BEACON autofill
            # helper is axial-only and is not part of this workflow.
            self._viewer.layers[name].autofill = False

    def _set_interaction_button_support(self, supported: dict[int, bool]) -> None:
        super()._set_interaction_button_support(
            {0: False, 1: False, 2: False, 3: supported.get(3, True)}
        )

    def _lock_session(self) -> None:
        # Backend spacing describes the physical 1 mm grid, independently of
        # an active source layer's local viewer units, flips or translation.
        if self.session_cfg is not None and self._prepared is not None:
            self.session_cfg["spacing"] = np.ones(3)
        super()._lock_session()
        self._update_workflow()

    def _unlock_session(self) -> None:
        self._session_ready = False
        super()._unlock_session()
        self._update_workflow()

    def _ensure_prompt_layer(self) -> None:
        if self._prompt_layer is None or self._prompt_layer not in self._viewer.layers:
            self._prompt_layer = self._viewer.add_labels(
                np.zeros(self._prepared.data.shape, dtype=np.uint8),
                name="BEACON 3-plane - Lasso masks", opacity=0.65,
                colormap=DirectLabelColormap(color_dict={
                    0: (0, 0, 0, 0), 1: (1, 0.12, 0.12, 1), None: (0, 0, 0, 0),
                }),
                **self._prepared.layer_transform(),
            )
            self._prompt_layer.editable = False
        self._refresh_prompt_masks()

    def _refresh_prompt_masks(self) -> None:
        if self._prompt_layer is None or self._prepared is None:
            return
        data = self._prompt_layer.data
        if data.shape != self._prepared.data.shape:
            data = np.zeros(self._prepared.data.shape, dtype=np.uint8)
            self._prompt_layer.data = data
        else:
            data.fill(0)
        for prompt in self._prompts.values():
            # Retain the bright red styling, but display the actual single
            # slice without expanding its trace into neighbouring slices.
            slices = tuple(slice(lo, hi) for lo, hi in prompt.bbox)
            data[slices] |= prompt.crop.astype(np.uint8)
        self._prompt_layer.visible = not self._generated
        self._prompt_layer.refresh()

    def _select_plane(self, axis: int) -> None:
        if not self._session_ready or self._generated or self._task_worker is not None:
            return
        self._viewer.dims.ndisplay = 2
        self._viewer.dims.order = (axis,) + tuple(i for i in range(3) if i != axis)
        self._restrict_navigation()
        if axis in self._prompts:
            self._set_slice(axis, self._prompts[axis].bbox[axis][0])
        self._fit_working_plane()
        self.interaction_button._uncheck()
        self.interaction_button._check(3)
        self.on_interaction_selected()
        layer = self._viewer.layers[self.lasso_layer_name]
        layer.editable = True
        layer.mode = "add_polygon_lasso"
        self.task_status.setText(f"Draw the {self._plane_names[axis].lower()} lasso. Reposition the slice if needed.")

    def on_interaction_selected(self):
        if self._generated or self._task_worker is not None:
            return
        self.interaction_button._uncheck()
        self.interaction_button._check(3)
        self.prompt_button._uncheck()
        self.prompt_button._check(0)
        super().on_interaction_selected()

    def on_interaction_shortcut(self, idx: int) -> None:
        if idx == 3 and self._session_ready and not self._generated:
            self.on_interaction_selected()

    def on_interaction(self, event) -> None:
        if (event.action == ActionType.ADDED and not event.source.is_free()
                and not self._capture_pending):
            # napari emits ADDED before finishing its mouse-drawing callback.
            # Clearing the polygon or switching planes inside that callback
            # would re-enter _finish_drawing while _is_creating is still True.
            self._capture_pending = True
            QTimer.singleShot(0, self._capture_finished_lasso)

    def _capture_finished_lasso(self) -> None:
        self._capture_pending = False
        if self.lasso_layer_name not in self._viewer.layers:
            return
        layer = self._viewer.layers[self.lasso_layer_name]
        if not len(layer.data):
            layer._is_free = True
            return
        self.add_interaction()

    def add_interaction(self, *_args, **_kwargs):
        if not self._session_ready or self._generated or self._task_worker is not None:
            return
        layer = self._viewer.layers[self.lasso_layer_name]
        if layer.is_free():
            return
        try:
            crop, bbox = layer.get_last()
            axis = int(self._viewer.dims.not_displayed[0])
            prompt = LassoPrompt(axis, np.array(crop, dtype=np.uint8, copy=True), bbox)
            prompt.validate(self._prepared.data.shape)
        except (ValueError, IndexError) as error:
            self._clear_lasso(layer)
            show_warning(f"Invalid lasso: {error}")
            return
        layer.run()
        self._clear_lasso(layer)
        self._prompts.pop(axis, None)
        self._prompts[axis] = prompt
        self._refresh_prompt_masks()
        # The remaining planes should initially cut through this lesion,
        # rather than through an unrelated central slice of the full scan.
        centre = np.argwhere(prompt.crop).mean(axis=0) + np.array(prompt.bbox)[:, 0]
        for other in range(3):
            if other != axis:
                self._set_slice(other, centre[other])
        missing = next((i for i in range(3) if i not in self._prompts), None)
        if missing is not None:
            self._select_plane(missing)
        else:
            self.task_status.setText("Three masks ready. Generate the segmentation or redraw a plane.")
        self._update_workflow()

    def _generate(self) -> None:
        if not self._session_ready or self._generated or len(self._prompts) != 3:
            return
        layer = self._viewer.layers[self.lasso_layer_name]
        layer.editable = False
        self._start_task(
            predict_three_lassos, self.session, list(self._prompts.values()),
            tuple(self._prepared.data.shape), message="Segmenting from the three masks…",
            returned=self._on_predicted,
        )

    def _on_predicted(self, mask: np.ndarray) -> None:
        result = self._result_layer()
        np.copyto(result.data, mask)
        result.visible = True
        result.editable = False
        result.refresh()
        self._generated = True
        self._prompt_layer.visible = False
        self._viewer.layers[self.lasso_layer_name].editable = False
        self._viewer.layers.selection.active = result
        self.task_status.setText(
            "Segmentation ready. Undo to change the lassos, or Refine Manually."
            if np.any(mask) else "The predicted mask is empty. Undo to adjust the lassos."
        )

    def on_undo(self, *_args, **_kwargs):
        if self._task_worker is not None:
            return
        if self._manual_refinement_active:
            self._result_layer().undo()
            return
        if self._generated:
            self._generated = False
            layer = self._result_layer()
            layer.data.fill(0)
            layer.refresh()
            self._refresh_prompt_masks()
            self._select_plane(0)
            self.task_status.setText("Prediction undone. Adjust a lasso or generate again.")
        elif self._prompts:
            axis, _ = self._prompts.popitem()
            self._refresh_prompt_masks()
            self._select_plane(axis)
        self._update_workflow()

    def on_refine_manually(self, *_args):
        if not self._generated or self._task_worker is not None:
            return
        super().on_refine_manually()
        if self._manual_refinement_active:
            self.task_status.setText("Manual refinement: Paint/Erase. Ctrl+Z undoes manual edits.")
        self._update_workflow()

    def _export(self) -> None:
        if not self._generated or self._task_worker is not None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export mask on the original grid", "segmentation.nii.gz",
            "NIfTI (*.nii.gz *.nii);;MetaImage (*.mha);;NRRD (*.nrrd)",
        )
        if not path:
            return
        if not any(path.lower().endswith(ext) for ext in (".nii", ".nii.gz", ".mha", ".nrrd")):
            path += ".nii.gz"
        self._start_task(
            self._write_mask, self._prepared, self._result_layer().data.copy(), path,
            message="Exporting on the original grid…",
            returned=lambda saved: self.task_status.setText(f"Exported: {saved}"),
        )

    @staticmethod
    def _write_mask(volume: PreparedVolume, mask: np.ndarray, path: str) -> str:
        exported = volume.mask_in_original_space(mask)
        exported.SetMetaData("descrip", SEGMENTATION_MARKER)
        exported.SetMetaData("ITK_FileNotes", SEGMENTATION_MARKER)
        exported.SetMetaData("beacon_segmentation", "1")
        sitk.WriteImage(exported, path, True)
        return str(Path(path))

    def _clear_layers(self) -> None:
        # Clear prompts in place; no layer destruction on reset or Undo.
        for name in getattr(self, "layer_dict", {}).values():
            if name in self._viewer.layers:
                self._clear_lasso(self._viewer.layers[name])
        self._interaction_history = []

    @staticmethod
    def _clear_lasso(layer) -> None:
        with layer.events.blocker():
            # Also abort an unfinished mouse stroke when changing the image.
            layer._is_creating = False
            layer._moving_value = (None, None)
            layer._value = (None, None)
            layer.selected_data = set()
            while len(layer.data):
                layer.remove_last()
            layer._is_free = True
            layer._finish_drawing()
        layer.refresh()

    def on_next(self, *_args, **_kwargs):
        return None

    def on_run(self, *_args, **_kwargs):
        self._generate()

    def on_load_mask(self, *_args, **_kwargs):
        return None

    def on_reset_interactions(self, *_args, **_kwargs):
        return None

    def _store_in_progress_segmentation(self) -> None:
        # This workflow has no object aggregation. The current result stays in
        # its layer and remains exportable when model settings change.
        pass

    def on_uninit(self, *_args, **_kwargs) -> None:
        if self._task_worker is not None:
            return
        if self._remote_mode:
            self._disconnect_remote()
        else:
            self.session = None
        self._session_ready = False
        self._unlock_session()
        self._update_workflow()
