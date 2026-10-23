from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from napari.components import ViewerModel
from napari.layers import Image, Labels
from napari.plugins.utils import get_preferred_reader
from napari.settings import get_settings
from npe2 import PluginManager

from napari_beacon_dicom.reader import configure_dicom_reader_preference
from napari_beacon_dicom.volume_reader import (
    SEGMENTATION_MARKER, geometry_metadata, read_medical_volume,
)


def image_grid(array):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.85546875, 0.85546875, 1.5))
    image.SetOrigin((437.14453125, 437.14453125, -269))
    image.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    return image


class VolumeReaderTests(unittest.TestCase):
    def test_legacy_gzip_preference_does_not_override_medical_volume_reader(self):
        settings = get_settings()
        previous = dict(settings.plugins.extension2reader)
        try:
            settings.plugins.extension2reader = {
                "*.gz": "napari-nifti", "*": "napari", "*.tif": "napari",
                "*.nii.gz": "napari-nifti",
            }
            configure_dicom_reader_preference(settings)
            configured = list(settings.plugins.extension2reader.items())
            for path in ("mask.nii.gz", "image.nii", "image.nrrd", "slice.dcm"):
                self.assertEqual(get_preferred_reader(path), "napari-beacon-dicom", path)
            self.assertEqual(get_preferred_reader("archive.gz"), "napari-nifti")
            self.assertEqual(settings.plugins.extension2reader["*.tif"], "napari")
            configure_dicom_reader_preference(settings)
            self.assertEqual(list(settings.plugins.extension2reader.items()), configured)
        finally:
            settings.plugins.extension2reader = previous

    def test_image_and_exported_mask_import_on_the_same_grid(self):
        PluginManager.instance().discover()
        configure_dicom_reader_preference()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = image_grid(np.arange(5 * 7 * 9, dtype=np.int16).reshape(5, 7, 9))
            mask_data = np.zeros((5, 7, 9), dtype=np.uint8)
            mask_data[2:4, 3:5, 4:6] = 1
            mask = image_grid(mask_data)
            sitk.WriteImage(image, str(root / "image.nii.gz"))
            sitk.WriteImage(mask, str(root / "segmentation.nii.gz"))
            viewer = ViewerModel()
            preferred = get_preferred_reader(root / "segmentation.nii.gz")
            self.assertEqual(preferred, "napari-beacon-dicom")
            source = viewer.open(str(root / "image.nii.gz"), plugin=None)[0]
            segmentation = viewer.open(str(root / "segmentation.nii.gz"), plugin=None)[0]
            self.assertIsInstance(source, Image)
            self.assertIsInstance(segmentation, Labels)
            np.testing.assert_allclose(source.scale, (1.5, 0.85546875, 0.85546875))
            np.testing.assert_allclose(segmentation._data_to_world.affine_matrix,
                                       source._data_to_world.affine_matrix)
            np.testing.assert_array_equal(segmentation.data, mask_data)
            np.testing.assert_allclose(segmentation.data_to_world((3, 4, 5)),
                                       source.data_to_world((3, 4, 5)))
            self.assertEqual(segmentation.metadata["beacon_volume"]["origin_xyz_lps_mm"], image.GetOrigin())

    def test_dicom_mask_reimport_with_legacy_gzip_preference(self):
        from test_reader import _write_ct_series

        PluginManager.instance().discover()
        settings = get_settings()
        previous = dict(settings.plugins.extension2reader)
        try:
            settings.plugins.extension2reader = {"*.gz": "napari-nifti"}
            configure_dicom_reader_preference(settings)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                files = _write_ct_series(root / "dicom")
                viewer = ViewerModel()
                source = viewer.open(str(files[0]), plugin=None)[0]
                mask = sitk.GetImageFromArray(np.ones_like(source.data, dtype=np.uint8))
                geometry = source.metadata["dicom"]
                mask.SetSpacing(geometry["spacing_xyz_mm"])
                mask.SetOrigin(geometry["origin_xyz_lps_mm"])
                mask.SetDirection(geometry["direction_xyz_lps"])
                path = root / "segmentation.nii.gz"
                sitk.WriteImage(mask, str(path))
                segmentation = viewer.open(str(path), plugin=None)[0]
                self.assertIsInstance(segmentation, Labels)
                self.assertEqual(segmentation.source.reader_plugin, "napari-beacon-dicom")
                np.testing.assert_allclose(segmentation._data_to_world.affine_matrix,
                                           source._data_to_world.affine_matrix)
                np.testing.assert_array_equal(segmentation.data, np.ones_like(source.data))
        finally:
            settings.plugins.extension2reader = previous

    def test_import_uses_existing_dicom_display_translation_and_flip(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = image_grid(np.ones((5, 7, 9), dtype=np.uint8))
            path = Path(temporary) / "segmentation.nii.gz"
            sitk.WriteImage(image, str(path))
            viewer = ViewerModel()
            source = viewer.add_image(
                np.zeros((5, 7, 9)), scale=(-1.5, 0.85546875, 0.85546875),
                translate=(100, 20, 30), metadata={"dicom": geometry_metadata(image)},
            )
            data, attributes, kind = read_medical_volume(path, viewer=viewer)[0]
            imported = viewer.add_labels(data, **attributes)
            self.assertEqual(kind, "labels")
            np.testing.assert_allclose(imported._data_to_world.affine_matrix,
                                       source._data_to_world.affine_matrix)

    def test_working_volume_reference_preserves_source_pixel_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = image_grid(np.ones((5, 7, 9), dtype=np.uint8))
            path = Path(temporary) / "mask.nii.gz"
            sitk.WriteImage(image, str(path))
            matrix = np.eye(4)
            matrix[:3, 3] = (11, 22, 33)
            viewer = ViewerModel()
            viewer.add_image(np.zeros((8, 6, 8)), metadata={
                "beacon_export_reference": {
                    "geometry": geometry_metadata(image), "display_matrix_zyx": matrix.tolist(),
                },
            })
            data, attributes, _ = read_medical_volume(path, viewer=viewer)[0]
            imported = viewer.add_labels(data, **attributes)
            np.testing.assert_allclose(imported._data_to_world.affine_matrix, matrix)

    def test_marked_multiclass_export_is_labels_regardless_of_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            for suffix in (".nii.gz", ".mha", ".nrrd"):
                image = image_grid(np.full((5, 7, 9), 3, dtype=np.uint8))
                image.SetMetaData("ITK_FileNotes", SEGMENTATION_MARKER)
                image.SetMetaData("beacon_segmentation", "1")
                path = Path(temporary) / f"prediction{suffix}"
                sitk.WriteImage(image, str(path))
                data, attributes, kind = read_medical_volume(path, viewer=ViewerModel())[0]
                self.assertEqual(kind, "labels", suffix)
                np.testing.assert_allclose(attributes["scale"], (1.5, 0.85546875, 0.85546875))
                np.testing.assert_array_equal(data, np.full((5, 7, 9), 3, dtype=np.uint8))

    def test_unmarked_integer_image_is_not_guessed_to_be_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.nii.gz"
            sitk.WriteImage(image_grid(np.ones((5, 7, 9), dtype=np.uint8)), str(path))
            self.assertEqual(read_medical_volume(path, viewer=ViewerModel())[0][2], "image")


if __name__ == "__main__":
    unittest.main()
