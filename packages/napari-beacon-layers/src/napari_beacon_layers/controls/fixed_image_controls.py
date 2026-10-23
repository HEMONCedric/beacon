from napari.utils.action_manager import action_manager

from .image_window_controls import QtBeaconImageControls


class CustomQtFixedImageControls(QtBeaconImageControls):
    """Shared image/window controls with transform and projection hidden."""

    def __init__(self, layer):
        super().__init__(layer)

        # We don't need this Fields -> Hide them
        self._projection_mode_control.projection_combobox.setHidden(True)
        self._projection_mode_control.projection_combobox_label.setHidden(True)

        # We don't need all these button -> hide and disable tem + remove key binding
        buttons_to_hide = [
            {"button": self.transform_button, "shortcut": "napari:activate_image_transform_mode"},
        ]

        for button in buttons_to_hide:
            button["button"].setDisabled(True)
            button["button"].hide()
            action_manager.unbind_shortcut(button["shortcut"])
