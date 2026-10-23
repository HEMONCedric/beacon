import unittest

import numpy as np
from napari.components import ViewerModel

from patches.image_interpolation import install_image_interpolation_default


class ImageInterpolationTests(unittest.TestCase):
    def test_default_preserves_data_geometry_and_manual_choice(self) -> None:
        viewer = ViewerModel()
        data = np.zeros((3, 4, 5))
        first = viewer.add_image(data, scale=(2.5, 0.8, 0.7))
        install_image_interpolation_default(viewer)
        self.assertEqual(first.interpolation2d, "spline36")
        self.assertIs(first.data, data)
        np.testing.assert_allclose(first.scale, (2.5, 0.8, 0.7))

        second = viewer.add_image(data)
        self.assertEqual(second.interpolation2d, "spline36")
        second.interpolation2d = "nearest"
        self.assertEqual(second.interpolation2d, "nearest")
        self.assertEqual(len(viewer.layers), 2)
