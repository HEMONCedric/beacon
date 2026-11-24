from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from weakref import WeakSet

import numpy as np
import SimpleITK as sitk
from app_model.types import KeyCode, KeyMod
from napari.layers.labels._labels_constants import Mode
from napari.qt.threading import create_worker
from napari.utils.notifications import show_info, show_warning
from napari_beacon_layers import ManualLabelsLayer
from napari_nninteractive import nnInteractiveWidget
from qtpy.QtCore import QSignalBlocker, Signal
from qtpy.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QShortcut, QWidget,
)

from .native_grid import grid_metadata
from .ssh_tunnel import DEFAULT_SERVER_URL, SSH_TARGET_SETTING, manager, spec_from_url
from .volume_geometry import (
    declared_modality, export_display_reference, labels_on_original_grid, read_layer_image, write_segmentation,
)


LOGGER = logging.getLogger(__name__)
ENGINE_SETTING = "beacon/interactive_engine"
ONCOREP_CHECKPOINT_SETTING = "beacon/oncorep/checkpoint"
UNDO_KEY = KeyMod.CtrlCmd | KeyCode.KeyZ


def initialize_nninteractive(checkpoint_text, model_id, compile_model, autozoom, storage):
    import nnInteractive
    import torch
    from nnInteractive.model_management import ensure_model_available
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    checkpoint = Path(checkpoint_text) if checkpoint_text else None
    if checkpoint is None or not checkpoint.exists():
        checkpoint = ensure_model_available(model_id)
    class_file = checkpoint / "inference_session_class.json"
    inference_class = json.loads(class_file.read_text()) if class_file.is_file() else "nnInteractiveInferenceSession"
    if isinstance(inference_class, dict):
        inference_class = inference_class["inference_class"]
    factory = recursive_find_python_class(
        str(Path(nnInteractive.__path__[0]) / "inference"), inference_class, "nnInteractive.inference",
    )
    if factory is None:
        raise ValueError(f"Unknown nnInteractive inference class: {inference_class}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    session = factory(
        device=device, use_torch_compile=compile_model, torch_n_threads=os.cpu_count() or 1,
        verbose=False, do_autozoom=autozoom and device.type == "cuda", interactions_storage=storage,
    )
    session.initialize_from_trained_model_folder(str(checkpoint), 0, "checkpoint_final.pth")
    return session


def initialize_oncorep(weights, modality, layer_args, prepared, progress):
    try:
        from .oncorep_session import OncoRepSession, resolve_modality
    except ModuleNotFoundError as error:
        if error.name != "oncorep":
            raise
        raise RuntimeError("Install IMPACT-OncoRep following BEACON's Local installation guide.") from error

    if prepared is not None:
        native_data, native_grid = prepared.source_data, prepared.source_grid
        displayed, original = prepared.working, prepared.original
        metadata_modality = prepared.modality
    else:
        displayed, original = read_layer_image(*layer_args)
        native_data = sitk.GetArrayViewFromImage(displayed)
        native_grid = grid_metadata(displayed)
        metadata_modality = declared_modality(layer_args[2])
    modality = resolve_modality(modality, metadata_modality, native_data)
    progress("Loading IMPACT-OncoRep weights…")
    session = OncoRepSession(weights)
    session.set_volume(native_data, native_grid, displayed, modality, progress)
    return session, (displayed, original)


def export_classic_masks(input_args, geometry, output, masks):
    displayed, original = geometry if geometry is not None else read_layer_image(*input_args)
    output.mkdir(exist_ok=True)
    for filename, mask in masks:
        write_segmentation(labels_on_original_grid(mask, displayed, original), output / filename)
    return str(output), export_display_reference(input_args[1], displayed, original)


class nnInteractiveWidgetBeacon(nnInteractiveWidget):
    """nnInteractive Classic with an explicit final manual-refinement phase."""

    backend_progress = Signal(str)
    _supports_oncorep = True
    _background_classic_prompts = True
    _AI_CONTROLS = (
        "init_button",
        "prompt_button",
        "interaction_button",
        "reset_interaction_button",
        "undo_button",
        "auto_run_ckbx",
        "run_button",
        "load_mask_btn",
        "label_for_init",
        "class_for_init",
        "auto_refine",
        "aggregation_output_combo",
        "aggregation_overlap_combo",
        "aggregation_class_id",
        "export_button",
        "propagate_ckbx",
    )

    def __init__(self, napari_viewer, *args: Any, **kwargs: Any):
        self._manual_refinement_active = False
        self._saved_ai_control_states: dict[str, bool] = {}
        self._saved_prompt_editability: dict[str, bool] = {}
        self._saved_prompt_modes: dict[str, Any] = {}
        self._ssh_tunnel_worker = None
        self._beacon_worker = None
        self._image_changed_during_worker = False
        self._undo_layers = WeakSet()
        self._oncorep_export_reference = None
        self._active_oncorep_options = None
        self._classic_nninteractive_target = None

        super().__init__(napari_viewer, *args, **kwargs)
        from qtpy.QtWidgets import QSizePolicy

        self.setMinimumSize(0, 0)
        self.setMinimumHeight(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
)
        self.refine_manual_button = QPushButton("Refine Manually")
        self.refine_manual_button.setToolTip(
            "Freeze nnInteractive prompts and refine the current mask with "
            "Napari Paint/Erase before Next Object."
        )
        self.refine_manual_button.clicked.connect(self.on_refine_manually)
        self._insert_manual_refinement_button()
        self.refine_manual_button.setEnabled(self.session is not None)
        self._nninteractive_autorefine_tooltip = self.auto_refine.toolTip()
        self.backend_progress.connect(self._show_backend_phase)
        if self._supports_oncorep:
            for shortcut in self.undo_button.findChildren(QShortcut):
                shortcut.setEnabled(False)
            self._show_engine_settings()

    def _init_model_selection(self) -> QGroupBox:
        if self._supports_oncorep and self._local_available:
            # The upstream widget fetches the Hugging Face manifest synchronously
            # during construction. Use its validated disk cache for these docks.
            self._local_available = False
            try:
                group = super()._init_model_selection()
            finally:
                self._local_available = True
            self._load_cached_nninteractive_models()
        else:
            group = super()._init_model_selection()
        if not self.server_url_edit.text().strip():
            self.server_url_edit.blockSignals(True)
            self.server_url_edit.setText(DEFAULT_SERVER_URL)
            self.server_url_edit.blockSignals(False)
        self.backend_status = QLabel()
        self.backend_status.setWordWrap(True)
        self.backend_spinner = QProgressBar()
        self.backend_spinner.setRange(0, 0)
        self.backend_spinner.setTextVisible(False)
        self.backend_status.hide()
        self.backend_spinner.hide()
        if self._supports_oncorep:
            self.engine_combo = QComboBox()
            self.engine_combo.addItem("nnInteractive", "nninteractive")
            self.engine_combo.addItem("IMPACT-OncoRep", "oncorep")
            saved_engine = self._settings.value(ENGINE_SETTING, "nninteractive", type=str)
            self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(saved_engine)))
            engines = QFormLayout()
            engines.addRow("Model", self.engine_combo)
            group.layout().insertLayout(0, engines)
            self.oncorep_options = QWidget()
            options = QFormLayout(self.oncorep_options)
            options.setContentsMargins(0, 0, 0, 0)
            self.oncorep_weights = QLineEdit(self._saved_oncorep_checkpoint())
            self.oncorep_weights.setPlaceholderText("Released interactive.pt")
            self.oncorep_weights.setToolTip("Use the Hub release, or enter a local interactive.pt file.")
            self.oncorep_modality = QComboBox()
            for label, value in (("Auto", "Auto"), ("CT (HU)", "CT"), ("MR", "MR"), ("PET (SUV)", "PET")):
                self.oncorep_modality.addItem(label, value)
            options.addRow("Checkpoint", self.oncorep_weights)
            options.addRow("Modality", self.oncorep_modality)
            group.layout().insertWidget(3, self.oncorep_options)
            self.oncorep_options.hide()
            self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
            self.oncorep_modality.currentIndexChanged.connect(self._on_oncorep_settings_changed)
            self.oncorep_weights.editingFinished.connect(self._save_oncorep_checkpoint)

        self.ssh_target_edit = QLineEdit(self._saved_ssh_target())
        self.ssh_target_edit.setPlaceholderText("user@host or SSH alias (optional for a local server)")
        self.ssh_target_edit.setToolTip(
            "For a localhost URL, open an SSH tunnel automatically before Connect or Initialize. "
            "Leave blank when the server is already reachable locally. SSH key authentication is required "
            "for automatic tunnels; the server API key is independent and optional."
        )
        self.ssh_target_edit.textChanged.connect(
            lambda value: self._settings.setValue(SSH_TARGET_SETTING, value)
        )
        self.ssh_target_edit.textChanged.connect(self.on_remote_settings_changed)
        row = QHBoxLayout()
        row.addWidget(QLabel("SSH target"))
        row.addWidget(self.ssh_target_edit, 1)
        self.remote_container.layout().insertLayout(1, row)
        return group

    def _load_cached_nninteractive_models(self):
        from nnInteractive.model_management import get_model_root_dir

        self.local_container.setEnabled(True)
        self.mode_switch._uncheck()
        self.mode_switch._check(int(self._remote_mode))
        self.mode_switch.buttons[0].setToolTip("Run inference locally")
        try:
            path = get_model_root_dir() / "models.json"
            models = json.loads(path.read_text(encoding="utf-8"))["models"]
            models = [item for item in models if str(item["id"]).startswith("nnInteractive_")]
        except (OSError, KeyError, TypeError, ValueError):
            models = []
        if not models:
            models = [{"id": "nnInteractive_v1.0", "display_name": "nnInteractive v1.0", "default": True}]
        self._model_ids = [item["id"] for item in models]
        default = next((index for index, item in enumerate(models) if item.get("default")), 0)
        self.model_selection.blockSignals(True)
        self.model_selection.addItems([item.get("display_name", item["id"]) for item in models])
        self.model_selection.setCurrentIndex(default)
        self.model_selection.blockSignals(False)

    def _oncorep_selected(self):
        return self._supports_oncorep and self.engine_combo.currentData() == "oncorep"

    def _saved_ssh_target(self):
        return self._settings.value(
            SSH_TARGET_SETTING,
            self._settings.value("beacon/nninteractive_ssh_target", "", type=str),
            type=str,
        )

    def _saved_oncorep_checkpoint(self):
        saved = self._settings.value(ONCOREP_CHECKPOINT_SETTING, "", type=str)
        return os.environ.get("ONCOREP_INTERACTIVE_WEIGHTS", saved)

    def _show_engine_settings(self):
        oncorep = self._oncorep_selected()
        # Upstream inference_mode and server_url describe nnInteractive only.
        self._remote_mode = not oncorep and self._restore_remote_mode()
        self.mode_switch._uncheck()
        self.mode_switch._check(int(self._remote_mode))
        if not oncorep and not self._local_available:
            self._grey_local_switch_button()
        self.mode_switch.buttons[1].setEnabled(not oncorep)
        self.local_container.setVisible(not self._remote_mode and not oncorep)
        self.remote_container.setVisible(self._remote_mode and not oncorep)
        self.oncorep_options.setVisible(oncorep)
        self._update_init_button_text(initialized=self._session_locked)

    def _on_engine_changed(self):
        if self._remote_connected:
            self._disconnect_remote()
        self._settings.setValue(ENGINE_SETTING, self.engine_combo.currentData())
        # The other BEACON dock may have updated the shared preferences.
        with QSignalBlocker(self.server_url_edit), QSignalBlocker(self.ssh_target_edit):
            url = self._settings.value("server_url", "", type=str).strip()
            self.server_url_edit.setText(url or DEFAULT_SERVER_URL)
            self.ssh_target_edit.setText(self._saved_ssh_target())
        if self._oncorep_selected():
            self.oncorep_weights.setText(self._saved_oncorep_checkpoint())
        self._show_engine_settings()
        self.on_model_selected()
        self.auto_refine.setToolTip(self._nninteractive_autorefine_tooltip)
        self.backend_status.hide()

    def _save_oncorep_checkpoint(self):
        self._settings.setValue(ONCOREP_CHECKPOINT_SETTING, self.oncorep_weights.text().strip())
        self._on_oncorep_settings_changed()

    def _on_oncorep_settings_changed(self):
        if self._oncorep_selected() and self.session is not None and self._beacon_worker is None:
            if self._active_oncorep_options != self._oncorep_settings():
                self.on_model_selected()
                self.backend_status.hide()

    def _oncorep_settings(self):
        weights = self.oncorep_weights.text().strip()
        local_release = Path("checkpoints/oncorep/interactive.pt")
        if not weights and local_release.is_file():
            weights = str(local_release.resolve())
        return weights, self.oncorep_modality.currentData()

    def _show_backend_phase(self, message):
        self.backend_status.setText(message)
        self.backend_status.show()

    def _watch_image(self, layer):
        if layer is None:
            return lambda: True
        name, data = layer.name, layer.data
        transform = layer._data_to_world.affine_matrix.copy()
        return lambda: (
            layer in self._viewer.layers and layer.name == name and layer.data is data
            and np.array_equal(layer._data_to_world.affine_matrix, transform)
        )

    def on_image_selected(self):
        if self._beacon_worker is not None:
            # Removing or renaming a viewer layer can change the combo box
            # even while it is disabled. Keep the session alive until Qt
            # receives the worker's terminal signal.
            self._image_changed_during_worker = True
            return
        return super().on_image_selected()

    def _backend_image_changed(self):
        self.on_image_selected()
        self._show_backend_phase("The selected image changed. Initialize again.")

    def _run_backend_worker(self, function, args, message, done, failed=None):
        if self._beacon_worker is not None:
            return
        self._image_changed_during_worker = False
        layers = self._viewer.layers
        name = self.image_selection.currentText()
        image_is_current = self._watch_image(layers[name] if name in layers else None)
        result_layer = layers[self.label_layer_name] if self.label_layer_name in layers else None
        session = self.session
        widgets = [
            self.init_button, self.reset_button, self.image_selection, self.mode_switch,
            self.local_container, self.remote_container, self.propagate_ckbx,
            self.prompt_button, self.interaction_button, self.reset_interaction_button,
            self.undo_button, self.run_button, self.export_button, self.load_mask_btn,
            self.auto_run_ckbx, self.refine_manual_button,
        ]
        if self._supports_oncorep:
            widgets.extend((self.engine_combo, self.oncorep_options))
        states = [(widget, widget.isEnabled()) for widget in widgets]
        for widget, _ in states:
            widget.setEnabled(False)
        prompts = []
        for name in self.layer_dict.values():
            if name in self._viewer.layers:
                layer = self._viewer.layers[name]
                if layer.editable:
                    prompts.append((layer, layer.mode))
                    layer.editable = False
        self._show_backend_phase(message)
        self.backend_spinner.show()
        worker = create_worker(function, *args, _start_thread=False, _ignore_errors=True)
        self._beacon_worker = worker
        self._backend_busy_changed()

        def finish():
            same_session = session is self.session
            valid = (
                same_session and not self._image_changed_during_worker and image_is_current()
                and (result_layer is None or result_layer in layers)
            )
            self._beacon_worker = None
            self._image_changed_during_worker = False
            self.backend_spinner.hide()
            for widget, enabled in states:
                widget.setEnabled(enabled)
            if valid:
                for layer, mode in prompts:
                    if layer in layers:
                        layer.editable = True
                        layer.mode = mode
            elif same_session:
                self._backend_image_changed()
            self._backend_busy_changed()
            return valid

        def succeeded(result):
            if finish():
                done(result)

        def errored(error):
            if not finish():
                return
            self._show_backend_phase(str(error))
            show_warning(str(error))
            if failed is not None:
                failed()

        worker.returned.connect(succeeded)
        worker.errored.connect(errored)
        worker.start()

    def _backend_busy_changed(self):
        pass

    @staticmethod
    def _layer_args(layer):
        return layer.data, layer._data_to_world.affine_matrix.copy(), dict(layer.metadata), layer.source.path

    def _prepared_for_oncorep(self):
        return None

    def on_init(self, *args, **kwargs):
        if self._beacon_worker is not None:
            return
        if not self._oncorep_selected():
            if self._supports_oncorep and self._local_available and not self._remote_mode and self.session is None:
                return self._init_nninteractive_local()
            return super().on_init(*args, **kwargs)
        name = self.image_selection.currentText()
        if name not in self._viewer.layers:
            show_warning("Choose a scalar 3D image first.")
            return self._backend_init_failed()
        layer = self._viewer.layers[name]
        if layer.ndim != 3 or layer.rgb or layer.multiscale:
            show_warning("IMPACT-OncoRep requires a scalar 3D image.")
            return self._backend_init_failed()
        settings = self._oncorep_settings()
        self._run_backend_worker(
            initialize_oncorep,
            (*settings, self._layer_args(layer), self._prepared_for_oncorep(), self.backend_progress.emit),
            "Initializing IMPACT-OncoRep…",
            lambda result: self._oncorep_initialized(layer, settings, result),
            self._backend_init_failed,
        )

    def _init_nninteractive_local(self):
        name = self.image_selection.currentText()
        if name not in self._viewer.layers:
            show_warning("Choose an image first.")
            return self._backend_init_failed()
        image = self._viewer.layers[name]
        index = self.model_selection.currentIndex()
        model_id = self._model_ids[index] if 0 <= index < len(self._model_ids) else "nnInteractive_v1.0"
        options = (self.model_selection_local.text(), model_id, self.use_torch_compile_ckbx.isChecked(),
                   self.propagate_ckbx.isChecked(), self.interactions_storage_combo.currentText())

        def done(session):
            if image not in self._viewer.layers or self.image_selection.currentText() != image.name:
                self._show_backend_phase("The selected image changed. Initialize again.")
                return self._backend_init_failed()
            self.session = session
            self._local_running_on_cpu = session.device.type == "cpu"
            if self._local_running_on_cpu:
                self.propagate_ckbx.setChecked(False)
            try:
                # The downloaded model is already loaded and warmed up. Let
                # napari-nninteractive configure its regular layers and image.
                super(nnInteractiveWidgetBeacon, self).on_init()
            except Exception as error:
                self.session = None
                self._show_backend_phase(str(error))
                show_warning(str(error))
                return self._backend_init_failed()
            self._show_backend_phase(f"nnInteractive ready · {str(session.device).upper()}")

        self._run_backend_worker(
            initialize_nninteractive, options, "Loading nnInteractive…", done, self._backend_init_failed,
        )

    def _backend_init_failed(self):
        self._unlock_session()

    def _oncorep_initialized(self, image, settings, initialized):
        if image not in self._viewer.layers or self.image_selection.currentText() != image.name:
            self._show_backend_phase("The selected image changed. Initialize again.")
            self._backend_init_failed()
            return
        self.session, self._oncorep_export_reference = initialized
        self._active_oncorep_options = settings
        self.source_cfg = {
            "name": image.name, "model": "IMPACT-OncoRep", "ndim": image.ndim,
            "shape": image.data.shape, "affine": image.affine, "scale": image.scale,
            "translate": image.translate, "rotate": image.rotate, "shear": image.shear,
            "source": image.source, "metadata": image.metadata,
        }
        self.session_cfg = self.source_cfg.copy()
        self.session_cfg["spacing"] = np.asarray(self._oncorep_export_reference[0].GetSpacing()[::-1])
        layer = self._viewer.layers[self.label_layer_name] if self.label_layer_name in self._viewer.layers else None
        resume = (
            self._resume_after_reconnect and self._resume_image_layer is image
            and layer is not None and layer.data.shape == image.data.shape
        )
        if resume:
            self._data_result = layer.data
        else:
            self._data_result = np.zeros(image.data.shape, dtype=np.uint8)
            self.object_index = self._next_object_index()
            if layer is not None:
                self._viewer.layers.remove(layer)
            self.add_label_layer(self._data_result, self.label_layer_name)
            self._viewer.layers[self.label_layer_name].colormap = self.colormap[self.object_index]
        # A worker owns the model buffer; Qt publishes completed results itself.
        self.session.set_target_buffer(np.zeros_like(self._data_result))
        self._resume_image_layer = image
        self._interaction_history = []
        self._reset_object_bbox()
        self._local_running_on_cpu = False  # Display the actual device in BEACON's status below.
        self._lock_session()
        self._set_interaction_button_support(dict.fromkeys(range(4), True))
        self.auto_refine.setEnabled(False)
        self.auto_refine.setToolTip("OncoRep uses the initial mask with the next prompt.")
        self._update_license_display(self.session.license)
        self._finish_init()
        self._show_backend_phase(
            f"OncoRep ready · {self.session.modality} · {str(self.session.device).upper()} · "
            f"preprocessing {self.session.preprocess_s:.1f} s"
        )
        LOGGER.info(self.backend_status.text())

    def _oncorep_session(self):
        return getattr(self.session, "segmentation_engine", "") == "oncorep"

    def _finish_init(self):
        super()._finish_init()
        if (self._supports_oncorep and self._background_classic_prompts
                and self.session is not None and not self._remote_mode and not self._oncorep_session()):
            self._classic_nninteractive_target = self._data_result.copy()
            self.session.set_target_buffer(self._classic_nninteractive_target)

    def _async_prompts(self):
        return self.session is not None and (self._oncorep_session() or self._classic_nninteractive_target is not None)

    def _publish_async_result(self, bbox=None):
        if self._oncorep_session():
            return self._publish_oncorep()
        if self.label_layer_name not in self._viewer.layers:
            return
        section = tuple(slice(lo, hi) for lo, hi in bbox) if bbox is not None else (...,)
        np.copyto(self._data_result[section], self._classic_nninteractive_target[section])
        self._accumulate_object_bbox(bbox)
        self._viewer.layers[self.label_layer_name].refresh()
        self._show_backend_phase("nnInteractive segmentation ready.")

    def _publish_oncorep(self, _result=None):
        if self.label_layer_name not in self._viewer.layers:
            return
        np.copyto(self._data_result, self.session.target)
        self._object_bbox_reliable = False
        self._viewer.layers[self.label_layer_name].refresh()
        summary = (
            f"OncoRep · inference {self.session.last_prediction_s:.2f} s · "
            f"mask projection {self.session.last_projection_s:.2f} s"
        )
        self._show_backend_phase(summary)
        LOGGER.info(summary)

    def _set_connecting(self, connecting: bool) -> None:
        super()._set_connecting(connecting)
        ssh_edit = getattr(self, "ssh_target_edit", None)
        if ssh_edit is not None:
            ssh_edit.setEnabled(not connecting)
        engine = getattr(self, "engine_combo", None)
        if engine is not None:
            engine.setEnabled(not connecting)

    def _start_claim(self, on_connected=None) -> None:
        """Open a localhost tunnel in a worker, then claim the remote session."""

        if self._ssh_tunnel_worker is not None:
            return
        try:
            spec = spec_from_url(self.server_url_edit.text())
        except ValueError as exc:
            self.remote_status_label.setText(f"Invalid server URL: {exc}")
            return
        if spec is None:
            super()._start_claim(on_connected=on_connected)
            return

        self._set_connecting(True)
        self.connect_btn.setText("Opening SSH...")
        self.remote_status_label.setText("Opening SSH tunnel...")
        worker = create_worker(
            manager().ensure,
            spec,
            self.ssh_target_edit.text().strip(),
            _connect={
                "returned": lambda _status: self._on_ssh_tunnel_ready(on_connected),
                "errored": lambda error: self._on_ssh_tunnel_error(error),
            },
            _start_thread=False,
            _ignore_errors=True,
        )
        self._ssh_tunnel_worker = worker
        worker.finished.connect(lambda: setattr(self, "_ssh_tunnel_worker", None))
        worker.start()

    def _on_ssh_tunnel_ready(self, on_connected=None) -> None:
        super()._start_claim(on_connected=on_connected)

    def _on_ssh_tunnel_error(self, error: BaseException) -> None:
        self._set_connecting(False)
        self.connect_btn.setText("Connect")
        self.remote_status_label.setText(f"SSH tunnel failed: {error}")
        show_warning(f"Could not open nnInteractive SSH tunnel: {error}")

    def _insert_manual_refinement_button(self) -> None:
        layout = self.reset_button.parentWidget().layout()
        index = layout.indexOf(self.reset_button)
        if index < 0:
            layout.addWidget(self.refine_manual_button)
        else:
            layout.insertWidget(index, self.refine_manual_button)
        layout.addWidget(self.backend_status)
        layout.addWidget(self.backend_spinner)

    def add_label_layer(self, data, name) -> None:
        if name != self.label_layer_name:
            super().add_label_layer(data, name)
            return

        layer = ManualLabelsLayer(
            data,
            name=name,
            opacity=0.9,
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
        )
        layer.contour = 1
        layer.editable = False
        layer.mode = Mode.PAN_ZOOM
        layer._source = self.session_cfg["source"]
        self._bind_layer_undo(layer)
        self._viewer.add_layer(layer)

    def _bind_layer_undo(self, layer):
        if self._supports_oncorep:
            layer.bind_key(UNDO_KEY, self._undo_from_layer, overwrite=True)
            self._undo_layers.add(layer)

    def _undo_from_layer(self, layer):
        if self._viewer.layers.selection.active is not layer or self._beacon_worker is not None:
            return
        if self._manual_refinement_active and layer.name == self.label_layer_name:
            if layer.editable:
                layer.undo()
        elif self.undo_button.isEnabled():
            self.on_undo()

    def on_interaction_selected(self):
        name = self.layer_dict.get(self.interaction_button.index)
        new_layer = name is not None and name not in self._viewer.layers
        super().on_interaction_selected()
        if new_layer and name in self._viewer.layers:
            self._bind_layer_undo(self._viewer.layers[name])

    def closeEvent(self, event):  # noqa: N802
        for layer in self._undo_layers:
            layer.bind_key(UNDO_KEY, None, overwrite=True)
        self._undo_layers.clear()
        super().closeEvent(event)

    def on_refine_manually(self, *_args: Any) -> None:
        if self._manual_refinement_active:
            return

        if self.label_layer_name not in self._viewer.layers:
            show_warning("Initialize nnInteractive before starting manual refinement.")
            return

        layer = self._viewer.layers[self.label_layer_name]
        if not np.any(np.asarray(layer.data)):
            show_warning("Create an nnInteractive segmentation before refining manually.")
            return

        self._manual_refinement_active = True
        self._object_bbox_reliable = False
        self._freeze_ai_interactions()

        layer.editable = True
        layer.selected_label = 1
        layer.mode = Mode.PAINT

        selection = self._viewer.layers.selection
        selection.clear()
        selection.add(layer)
        selection.active = layer

        self.refine_manual_button.setText("Manual Refinement Active")
        self.refine_manual_button.setEnabled(False)
        self._on_manual_refinement_started()

    def _freeze_ai_interactions(self) -> None:
        self._saved_ai_control_states.clear()
        for name in self._AI_CONTROLS:
            widget = getattr(self, name, None)
            if widget is None or not hasattr(widget, "setEnabled"):
                continue
            self._saved_ai_control_states[name] = bool(widget.isEnabled())
            widget.setEnabled(False)

        self._saved_prompt_editability.clear()
        self._saved_prompt_modes.clear()
        for layer_name in set(getattr(self, "layer_dict", {}).values()):
            if layer_name not in self._viewer.layers:
                continue
            layer = self._viewer.layers[layer_name]
            if not hasattr(layer, "editable"):
                continue
            self._saved_prompt_editability[layer_name] = bool(layer.editable)
            if hasattr(layer, "mode"):
                self._saved_prompt_modes[layer_name] = layer.mode
            layer.editable = False

    def _restore_ai_interactions(self) -> None:
        for name, enabled in self._saved_ai_control_states.items():
            widget = getattr(self, name, None)
            if widget is not None and hasattr(widget, "setEnabled"):
                widget.setEnabled(enabled)
        self._saved_ai_control_states.clear()

        for layer_name, editable in self._saved_prompt_editability.items():
            if layer_name not in self._viewer.layers:
                continue
            layer = self._viewer.layers[layer_name]
            if hasattr(layer, "editable"):
                layer.editable = editable
            if editable and layer_name in self._saved_prompt_modes and hasattr(layer, "mode"):
                layer.mode = self._saved_prompt_modes[layer_name]

        self._saved_prompt_editability.clear()
        self._saved_prompt_modes.clear()

    def _finish_manual_refinement_after_next(self) -> None:
        self._manual_refinement_active = False

        if self.label_layer_name in self._viewer.layers:
            layer = self._viewer.layers[self.label_layer_name]
            layer.editable = False
            layer.mode = Mode.PAN_ZOOM

        self._restore_ai_interactions()
        self.refine_manual_button.setText("Refine Manually")
        self.refine_manual_button.setEnabled(self.session is not None)
        self._on_manual_refinement_finished()

    def add_interaction(self, *_args: Any, **_kwargs: Any):
        if self._manual_refinement_active or self._beacon_worker is not None or self.session is None:
            return None
        if not self._async_prompts():
            return super().add_interaction()
        tool = self.interaction_button.index
        name = self.layer_dict.get(tool)
        if name not in self._viewer.layers:
            return
        layer = self._viewer.layers[name]
        if layer.is_free():
            return
        data = layer.get_last()
        layer.run()
        if data is None:
            return
        positive = self.prompt_button.index == 0
        run = self.auto_run_ckbx.isChecked()
        if tool == 0:
            operation, args = self.session.add_point_interaction, (data, positive, run)
        elif tool == 1:
            operation = self.session.add_bbox_interaction
            args = self._bbox_to_half_open_intervals(data), positive, run
        else:
            crop, bbox = data
            args = crop.copy(), positive, run, bbox
            if tool == 2:
                operation = self.session.add_scribble_interaction
            else:
                operation = self.session.add_lasso_interaction
                axes = self._viewer.dims.not_displayed
                if self._oncorep_session():
                    args += (axes[0] if len(axes) == 1 else None,)

        def done(bbox):
            self._interaction_history.append(name)
            if run:
                self._publish_async_result(bbox)

        def undo_marker():
            if layer in self._viewer.layers:
                layer.remove_last()
                layer.refresh()

        if run:
            self._run_backend_worker(operation, args, "Segmenting…", done, undo_marker)
        else:
            try:
                done(operation(*args))
            except ValueError as error:
                undo_marker()
                show_warning(str(error))

    def on_run(self, *_args: Any, **_kwargs: Any):
        if self._manual_refinement_active or self._beacon_worker is not None:
            return None
        if self._async_prompts():
            return self._run_backend_worker(
                self.session._predict, (), "Segmenting…", self._publish_async_result,
            )
        return super().on_run()

    def on_reset_interactions(self, *_args: Any, **_kwargs: Any):
        if self._manual_refinement_active or self._beacon_worker is not None:
            return None
        result = super().on_reset_interactions()
        if self._async_prompts():
            self._publish_async_result()
        return result

    def on_undo(self, *_args: Any, **_kwargs: Any):
        if self._manual_refinement_active or self._beacon_worker is not None:
            return None
        result = super().on_undo()
        if self._async_prompts():
            self._publish_async_result()
        return result

    def on_load_mask(self, *_args: Any, **_kwargs: Any):
        if self._manual_refinement_active or self._beacon_worker is not None:
            return None
        if not self._async_prompts():
            return super().on_load_mask()
        selected = self.label_for_init.currentText()
        if selected not in self._viewer.layers:
            show_warning("Choose a Labels layer first.")
            return
        labels = self._viewer.layers[selected]
        if (labels.data.shape != self._data_result.shape or not np.allclose(
            labels._data_to_world.affine_matrix,
            self._viewer.layers[self.source_cfg["name"]]._data_to_world.affine_matrix,
        )):
            show_warning("The initial mask must match the selected image grid.")
            return
        mask, label = labels.data, self.class_for_init.value()
        session = self.session
        oncorep = self._oncorep_session()
        auto_refine = not oncorep and self.auto_refine.isChecked()

        def seed():
            binary = mask == label
            if not binary.any():
                raise ValueError("The selected label is empty.")
            session.add_initial_seg_interaction(binary.astype(np.uint8), run_prediction=auto_refine)

        def done(_result):
            self._interaction_history.append(None)
            self._publish_async_result()
            if oncorep:
                self._show_backend_phase("Initial mask set. Draw an OncoRep prompt to refine it.")

        self._run_backend_worker(seed, (), "Loading the initial mask…", done)

    def on_interaction_shortcut(self, idx: int) -> None:
        if self._manual_refinement_active or self._beacon_worker is not None:
            return
        super().on_interaction_shortcut(idx)

    def on_next(self, *args: Any, **kwargs: Any):
        if self._beacon_worker is not None or self.label_layer_name not in self._viewer.layers:
            return None
        if not np.any(np.asarray(self._viewer.layers[self.label_layer_name].data)):
            return None

        was_manual = self._manual_refinement_active
        result = super().on_next(*args, **kwargs)
        if self._async_prompts():
            # The upstream reset clears the backend's private target buffer.
            self._data_result.fill(0)
            self._viewer.layers[self.label_layer_name].refresh()

        if was_manual:
            self._finish_manual_refinement_after_next()

        return result

    def on_uninit(self, *args, **kwargs):
        if self._beacon_worker is None:
            return super().on_uninit(*args, **kwargs)

    def _export(self):
        if self._beacon_worker is not None or not self.source_cfg:
            return
        image_name = self.source_cfg["name"]
        if self.source_cfg["ndim"] != 3 or not self._supports_oncorep:
            return super()._export()
        if image_name not in self._viewer.layers:
            show_warning("Open the source image before exporting its segmentation.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select an output folder")
        if not folder:
            return
        image = self._viewer.layers[image_name]
        base = Path(image.source.path).name if image.source.path else image_name
        base = base.removesuffix(".gz")
        base = Path(base).stem
        self.on_next()
        objects = []
        for layer in self._viewer.layers:
            if layer.name.startswith("object ") and layer.name.endswith(f" - {image_name}"):
                index = layer.name.removesuffix(f" - {image_name}").removeprefix("object ")
                if index.isdigit():
                    objects.append((f"{base}_{int(index):04d}.nii.gz", layer.data))
            elif layer.name in (f"instance map - {image_name}", f"semantic map - {image_name}"):
                name = layer.name.split(" - ", 1)[0].replace(" ", "_")
                objects.append((f"{base}_{name}.nii.gz", layer.data))
        if not objects:
            show_warning("There is no segmentation to export.")
            return
        oncorep = self._oncorep_session()
        geometry = self._oncorep_export_reference if oncorep else None
        backend = "OncoRep" if oncorep else "nnInteractive"

        def done(result):
            saved, reference = result
            if image in self._viewer.layers:
                image.metadata["beacon_export_reference"] = reference
            self._show_backend_phase(f"Exported: {saved}")
            show_info(f"Exported: {saved}")

        self._run_backend_worker(
            export_classic_masks,
            (self._layer_args(image), geometry, Path(folder) / f"{base}_{backend}", objects),
            "Exporting on the original grid…",
            done,
        )

    def _unlock_session(self) -> None:
        super()._unlock_session()
        self._classic_nninteractive_target = None
        self._manual_refinement_active = False
        self._saved_ai_control_states.clear()
        self._saved_prompt_editability.clear()
        self._saved_prompt_modes.clear()

        button = getattr(self, "refine_manual_button", None)
        if button is not None:
            button.setText("Refine Manually")
            button.setEnabled(False)

    def _lock_session(self) -> None:
        super()._lock_session()
        button = getattr(self, "refine_manual_button", None)
        if button is not None and not self._manual_refinement_active:
            button.setEnabled(True)

    def _on_manual_refinement_started(self) -> None:
        pass

    def _on_manual_refinement_finished(self) -> None:
        pass
