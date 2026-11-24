"""Geometry and cache checks for the optional released OncoRep backend."""

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from napari_nninteractive_beacon.mask_refinement import refine_with_lassos
from napari_nninteractive_beacon.volume_geometry import (
    LassoPrompt, labels_on_original_grid, predict_three_lassos, prepare_volume, write_segmentation,
)


@unittest.skipUnless(find_spec("oncorep"), "optional IMPACT-OncoRep installation")
class OncoRepTests(unittest.TestCase):
    def test_three_native_contours_refine_with_cached_encoding_and_export(self):
        import torch
        from oncorep import interactive
        from napari_nninteractive_beacon.oncorep_session import OncoRepSession

        class TestNetwork(torch.nn.Module):
            prompt_channels = 8
            multiple = 8
            encodes = 0

            def encode(self, images, present):
                self.encodes += 1
                return images[:, :1]

            def decode(self, features, channels):
                # A fixed foreground cube demonstrates mapping and export.
                out = torch.zeros_like(features)
                center = tuple(slice(n // 2 - 2, n // 2 + 3) for n in out.shape[-3:])
                out[(slice(None), slice(None), *center)] = 1
                self.last_channels = channels
                return torch.cat((torch.zeros_like(out), out), 1)

        data = np.full((13, 17, 15), -1000, dtype=np.int16)
        data[2:-2, 2:-2, 2:-2] = 55
        image = sitk.GetImageFromArray(data)
        image.SetOrigin((12.5, -84, 34))
        image.SetSpacing((1.0, 1.0, 2.0))
        image.SetDirection((0, 0, -1, 0, 1, 0, 1, 0, 0))
        prepared = prepare_volume(image, "test")
        session = OncoRepSession.__new__(OncoRepSession)
        session.device = torch.device("cpu")
        session.network = TestNetwork()
        session.set_volume(
            prepared.source_data, prepared.source_grid, prepared.working, "CT", lambda _: None,
        )
        center = (12, 8, 7)
        lassos = []
        for axis in range(3):
            bbox = [[val, val + 1] if i == axis else [val - 4, val + 5] for i, val in enumerate(center)]
            crop = np.ones(tuple(hi - lo for lo, hi in bbox), dtype=bool)
            lassos.append(LassoPrompt(axis, crop, bbox))

        # The published minimum patch is much larger than this synthetic image.
        with patch.object(interactive, "block_for", return_value=(32, 32, 32)):
            mask = predict_three_lassos(session, lassos, prepared.data.shape)
            self.assertEqual(session.network.encodes, 1)
            self.assertIsNotNone(session.engine.lasso)
            self.assertEqual(session.engine.extra_lassos, [])
            self.assertTrue(mask[center])
            extra = LassoPrompt(0, lassos[0].crop, [[13, 14], *lassos[0].bbox[1:]])
            refined, snapshot = refine_with_lassos(session, mask, [*lassos, extra], 0)
            self.assertEqual(session.network.encodes, 1)
            self.assertEqual(len(session.engine.extra_lassos), 1)
            self.assertTrue(session.network.last_channels[0, 5].any())
            np.testing.assert_array_equal(snapshot.restore(), mask)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "segmentation.nii.gz"
            write_segmentation(prepared.mask_in_original_space(refined), path)
            result = sitk.ReadImage(str(path))
        self.assertTrue(result.IsSameImageGeometryAs(image))
        self.assertEqual(result[7, 8, 6], 1)

    def test_oncorep_modality_requires_override_for_ambiguous_image(self):
        from napari_nninteractive_beacon.oncorep_session import resolve_modality

        image = np.ones((4, 5, 6), dtype=np.float32)
        self.assertEqual(resolve_modality("Auto", "MR", image), "MR")
        self.assertEqual(resolve_modality("PET", "MR", image), "PET")
        with self.assertRaisesRegex(ValueError, "modality"):
            resolve_modality("Auto", None, image)


class ClassicExportTests(unittest.TestCase):
    def test_preserve_multi_label_ids_after_existing_axis_permutation(self):
        native = sitk.Image((7, 8, 9), sitk.sitkUInt16)
        native.SetOrigin((12.5, -4.5, 17))
        native.SetSpacing((0.5, 2., 3.))
        displayed = sitk.PermuteAxes(native, (2, 1, 0))
        mask = np.zeros(displayed.GetSize()[::-1], np.uint16)
        mask[3, 4, 5] = 302
        output = labels_on_original_grid(mask, displayed, native)
        self.assertTrue(output.IsSameImageGeometryAs(native))
        self.assertEqual(output[3, 4, 5], 302)
