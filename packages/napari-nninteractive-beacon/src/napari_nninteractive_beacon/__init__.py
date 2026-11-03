__all__ = ["nnInteractiveWidgetBeacon", "nnInteractiveWidgetTriplanar"]


def __getattr__(name):
    # The remote server imports only geometry/transport, never Qt or napari.
    if name == "nnInteractiveWidgetBeacon":
        from .widget_nninteractive import nnInteractiveWidgetBeacon
        return nnInteractiveWidgetBeacon
    if name == "nnInteractiveWidgetTriplanar":
        from .widget_triplanar import nnInteractiveWidgetTriplanar
        return nnInteractiveWidgetTriplanar
    raise AttributeError(name)
