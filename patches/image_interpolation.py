"""Default rendering interpolation for images; source data is unchanged."""

from typing import Any

from napari.layers import Image


def install_image_interpolation_default(viewer: Any) -> None:
    """Use spline36 for new images, retaining napari's interpolation controls."""

    def set_default(layer: Any) -> None:
        if isinstance(layer, Image) and layer.interpolation2d == "nearest":
            layer.interpolation2d = "spline36"

    viewer.layers.events.inserted.connect(lambda event: set_default(event.value))
    for layer in viewer.layers:
        set_default(layer)
