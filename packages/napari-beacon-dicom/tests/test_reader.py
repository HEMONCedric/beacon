from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk
from npe2 import PluginManager, read as npe2_read

from napari_beacon_dicom.reader import (
    configure_dicom_reader_preference,
    discover_dicom_series,
    read_dicom,
    read_dicom_image,
    read_dicom_series,
)


def _write_ct_series(
    directory: Path,
    *,
    series_uid: str = "1.2.826.0.1.3680043.2.1125.10",
    series_number: int = 7,
    description: str = "Planning CT",
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for z in range(3):
        array = np.full((6, 8), z * 100, dtype=np.int16)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.7, 0.8))
        tags = {
            "0008|0016": "1.2.840.10008.5.1.4.1.1.2",
            "0008|0018": f"{series_uid}.{z + 1}",
            "0008|0060": "CT",
            "0008|103e": description,
            "0010|0010": "TEST^PATIENT",
            "0010|0020": "SECRET-ID",
            "0018|0050": "2.5",
            "0018|0088": "2.5",
            "0020|000d": "1.2.826.0.1.3680043.2.1125.20",
            "0020|000e": series_uid,
            "0020|0011": str(series_number),
            "0020|0013": str(z + 1),
            "0020|0032": f"10\\20\\{30 + 2.5 * z}",
            "0020|0037": "1\\0\\0\\0\\1\\0",
            "0020|0052": "1.2.826.0.1.3680043.2.1125.30",
            "0028|0030": "0.8\\0.7",
            "0028|1050": "40",
            "0028|1051": "400",
            "0028|1052": "0",
            "0028|1053": "1",
        }
        for tag, value in tags.items():
            image.SetMetaData(tag, value)
        destination = directory / f"series-{series_number}-slice-{z + 1:03d}.dcm"
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("GDCMImageIO")
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(str(destination))
        writer.Execute(image)
        files.append(destination.resolve())
    return files


class DicomReaderTests(unittest.TestCase):
    def test_beacon_reader_is_preferred_for_drag_and_drop(self) -> None:
        settings = SimpleNamespace(
            plugins=SimpleNamespace(
                extension2reader={"*.dcm": "napari", "*.tif": "napari"}
            )
        )

        configure_dicom_reader_preference(settings)

        self.assertEqual(
            settings.plugins.extension2reader["*.dcm"],
            "napari-beacon-dicom",
        )
        self.assertEqual(
            settings.plugins.extension2reader["*.dicom"],
            "napari-beacon-dicom",
        )
        self.assertEqual(
            settings.plugins.extension2reader["*.ima"],
            "napari-beacon-dicom",
        )
        self.assertEqual(settings.plugins.extension2reader["*.tif"], "napari")
        self.assertEqual(settings.plugins.extension2reader["*.nii.gz"], "napari-beacon-dicom")

    def test_discovers_and_reads_geometry_without_patient_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_ct_series(root / "study")

            discovered = discover_dicom_series(root)
            self.assertEqual(len(discovered), 1)
            series = discovered[0]
            self.assertEqual(series.modality, "CT")
            self.assertEqual(series.series_description, "Planning CT")
            self.assertEqual(len(series.files), 3)

            volume = read_dicom_series(series)
            self.assertEqual(volume.data.shape, (3, 6, 8))
            self.assertEqual(volume.scale_zyx, (2.5, 0.8, 0.7))
            np.testing.assert_array_equal(volume.data[:, 0, 0], [0, 100, 200])
            np.testing.assert_allclose(
                volume.affine_zyx,
                np.diag([2.5, 0.8, 0.7, 1.0])
                + np.array(
                    [
                        [0, 0, 0, 30],
                        [0, 0, 0, 20],
                        [0, 0, 0, 10],
                        [0, 0, 0, 0],
                    ]
                ),
            )
            self.assertEqual(volume.contrast_limits, (-160.0, 240.0))
            self.assertNotIn("patient_name", volume.metadata["dicom"])
            self.assertNotIn("patient_id", volume.metadata["dicom"])

    def test_file_selects_its_complete_series(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            files = _write_ct_series(Path(temporary))
            layer_data = read_dicom(files[1])
            self.assertEqual(len(layer_data), 1)
            data, attributes, layer_type = layer_data[0]
            self.assertEqual(data.shape, (3, 6, 8))
            self.assertEqual(layer_type, "image")
            self.assertEqual(attributes["name"], "7 - CT Planning CT")
            self.assertEqual(attributes["scale"], (2.5, 0.8, 0.7))
            self.assertNotIn("affine", attributes)

    def test_sitk_entrypoint_expands_one_file_to_complete_series(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            files = _write_ct_series(Path(temporary))
            image = read_dicom_image(files[2])

            self.assertEqual(image.GetSize(), (8, 6, 3))
            np.testing.assert_array_equal(
                sitk.GetArrayFromImage(image)[:, 0, 0], [0, 100, 200]
            )

    def test_multi_series_folder_requires_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_ct_series(root, series_number=7)
            _write_ct_series(
                root,
                series_uid="1.2.826.0.1.3680043.2.1125.11",
                series_number=8,
                description="Contrast CT",
            )
            discovered = discover_dicom_series(root)
            self.assertEqual([item.series_number for item in discovered], ["7", "8"])
            with self.assertRaisesRegex(ValueError, "Found 2 DICOM series"):
                read_dicom(root)

    def test_npe2_reader_opens_a_single_series_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_ct_series(root)
            PluginManager.instance().discover()
            layers = npe2_read(
                [str(root)], stack=False, plugin_name="napari-beacon-dicom"
            )
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0][0].shape, (3, 6, 8))
            self.assertEqual(layers[0][2], "image")

    def test_npe2_reader_expands_one_dcm_file_to_its_complete_series(self) -> None:
        """This is the path used by napari File > Open for one selected slice."""
        with tempfile.TemporaryDirectory() as temporary:
            files = _write_ct_series(Path(temporary))
            PluginManager.instance().discover()

            layers = npe2_read(
                [str(files[1])],
                stack=False,
                plugin_name="napari-beacon-dicom",
            )

            self.assertEqual(len(layers), 1)
            data, attributes, layer_type = layers[0]
            self.assertEqual(data.shape, (3, 6, 8))
            np.testing.assert_array_equal(data[:, 0, 0], [0, 100, 200])
            self.assertEqual(attributes["metadata"]["dicom"]["instance_count"], 3)
            self.assertEqual(layer_type, "image")

    def test_viewer_drag_drop_path_uses_beacon_reader(self) -> None:
        """The Qt drag-and-drop path lets napari resolve the preferred reader."""
        from napari.components import ViewerModel
        from napari.settings import get_settings

        settings = get_settings()
        previous = dict(settings.plugins.extension2reader)
        try:
            configure_dicom_reader_preference(settings)
            with tempfile.TemporaryDirectory() as temporary:
                files = _write_ct_series(Path(temporary))
                viewer = ViewerModel()
                # QtViewer._qt_open explicitly passes plugin=None.  Calling
                # ViewerModel.open without that argument is different: napari
                # defaults it to its own generic reader.
                layers = viewer.open(
                    [str(files[1])], stack=False, plugin=None
                )

                self.assertEqual(len(layers), 1)
                self.assertEqual(layers[0].data.shape, (3, 6, 8))
                np.testing.assert_allclose(layers[0].scale, (2.5, 0.8, 0.7))
                self.assertEqual(
                    layers[0].metadata["dicom"]["instance_count"], 3
                )
        finally:
            settings.plugins.extension2reader = previous

    def test_reads_single_file_multiframe_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "enhanced.dcm"
            image = sitk.GetImageFromArray(
                np.arange(3 * 6 * 8, dtype=np.int16).reshape(3, 6, 8)
            )
            image.SetSpacing((0.7, 0.8, 2.5))
            image.SetOrigin((10.0, 20.0, 30.0))
            image.SetMetaData("0008|0060", "CT")
            image.SetMetaData("0008|103e", "Enhanced CT")
            image.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125.99")
            writer = sitk.ImageFileWriter()
            writer.SetImageIO("GDCMImageIO")
            writer.KeepOriginalImageUIDOn()
            writer.SetFileName(str(destination))
            writer.Execute(image)

            discovered = discover_dicom_series(root)
            self.assertEqual(len(discovered), 1)
            volume = read_dicom_series(discovered[0])
            self.assertEqual(volume.data.shape, (3, 6, 8))


if __name__ == "__main__":
    unittest.main()
