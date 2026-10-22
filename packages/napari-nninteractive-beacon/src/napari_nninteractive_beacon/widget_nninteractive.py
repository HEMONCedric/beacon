from __future__ import annotations

from typing import Any

import numpy as np
from napari.layers.labels._labels_constants import Mode
from napari.qt.threading import create_worker
from napari.utils.notifications import show_warning
from napari_beacon_layers import ManualLabelsLayer
from napari_nninteractive import nnInteractiveWidget
from qtpy.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton

from .ssh_tunnel import DEFAULT_SERVER_URL, SSH_TARGET_SETTING, manager, spec_from_url


class nnInteractiveWidgetBeacon(nnInteractiveWidget):
    """nnInteractive Classic with an explicit final manual-refinement phase."""

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

    def _init_model_selection(self) -> QGroupBox:
        group = super()._init_model_selection()
        if not self.server_url_edit.text().strip():
            self.server_url_edit.blockSignals(True)
            self.server_url_edit.setText(DEFAULT_SERVER_URL)
            self.server_url_edit.blockSignals(False)

        saved = self._settings.value(
            SSH_TARGET_SETTING,
            self._settings.value("beacon/nninteractive_ssh_target", "", type=str),
            type=str,
        )
        self.ssh_target_edit = QLineEdit(saved)
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

    def _set_connecting(self, connecting: bool) -> None:
        super()._set_connecting(connecting)
        ssh_edit = getattr(self, "ssh_target_edit", None)
        if ssh_edit is not None:
            ssh_edit.setEnabled(not connecting)

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
        self._viewer.add_layer(layer)

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
        if self._manual_refinement_active:
            return None
        return super().add_interaction()

    def on_run(self, *_args: Any, **_kwargs: Any):
        if self._manual_refinement_active:
            return None
        return super().on_run()

    def on_reset_interactions(self, *_args: Any, **_kwargs: Any):
        if self._manual_refinement_active:
            return None
        return super().on_reset_interactions()

    def on_undo(self, *_args: Any, **_kwargs: Any):
        if self._manual_refinement_active:
            return None
        return super().on_undo()

    def on_load_mask(self, *_args: Any, **_kwargs: Any):
        if self._manual_refinement_active:
            return None
        return super().on_load_mask()

    def on_interaction_shortcut(self, idx: int) -> None:
        if self._manual_refinement_active:
            return
        super().on_interaction_shortcut(idx)

    def on_next(self, *args: Any, **kwargs: Any):
        if self.label_layer_name not in self._viewer.layers:
            return None
        if not np.any(np.asarray(self._viewer.layers[self.label_layer_name].data)):
            return None

        was_manual = self._manual_refinement_active
        result = super().on_next(*args, **kwargs)

        if was_manual:
            self._finish_manual_refinement_after_next()

        return result

    def _unlock_session(self) -> None:
        super()._unlock_session()
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
