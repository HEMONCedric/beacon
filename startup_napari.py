from patches.windows_opengl import configure_windows_opengl

configure_windows_opengl()

from patches.nninteractive_windows_fix import apply_nninteractive_windows_fix

apply_nninteractive_windows_fix()

import napari
from napari import Viewer
from napari_beacon_dicom import configure_dicom_reader_preference

configure_dicom_reader_preference()

viewer = Viewer()

from patches.image_interpolation import install_image_interpolation_default
install_image_interpolation_default(viewer)

from artist_standalone_app import StandaloneAppWidget
#widget = StandaloneAppWidget(viewer)
#viewer.window.add_dock_widget(widget, name="ARTIST Standalone", area="left")

napari.run()
