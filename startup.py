from patches.windows_opengl import configure_windows_opengl

configure_windows_opengl()

from patches.nninteractive_windows_fix import apply_nninteractive_windows_fix
apply_nninteractive_windows_fix()

from napari_beacon_dicom import configure_dicom_reader_preference
configure_dicom_reader_preference()

import napari
from napari import Viewer

viewer = Viewer()

from patches.image_interpolation import install_image_interpolation_default
install_image_interpolation_default(viewer)

from artist_study_app import StudyAppWidget
widget = StudyAppWidget(viewer)
viewer.window.add_dock_widget(
    widget, name="ARTIST study", area="left"
)

napari.run()
