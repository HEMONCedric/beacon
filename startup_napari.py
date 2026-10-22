from patches.windows_opengl import configure_windows_opengl

configure_windows_opengl()

import napari
from napari import Viewer
from napari_beacon_dicom import configure_dicom_reader_preference

configure_dicom_reader_preference()

viewer = Viewer()

from artist_standalone_app import StandaloneAppWidget
#widget = StandaloneAppWidget(viewer)
#viewer.window.add_dock_widget(widget, name="ARTIST Standalone", area="left")

napari.run()
