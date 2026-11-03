import unittest

import numpy as np

from napari_nninteractive_beacon.mask_refinement import (
    MaskSnapshot, dilate_mask_mm, refine_with_lassos,
)
from napari_nninteractive_beacon.volume_geometry import LassoPrompt


class MaskRefinementTests(unittest.TestCase):
    def test_dilation_is_a_2_mm_sphere_not_a_cube(self):
        mask = np.zeros((9, 9, 9), np.uint8)
        mask[4, 4, 4] = 1
        result = dilate_mask_mm(mask, (1, 1, 1))
        self.assertEqual(np.count_nonzero(result), 33)
        for offset in ((2, 0, 0), (0, -2, 0), (0, 0, 2), (1, 1, 1)):
            self.assertEqual(result[tuple(np.array(offset) + 4)], 1)
        self.assertEqual(result[6, 5, 4], 0)  # sqrt(5) mm is outside the radius.
        self.assertEqual(result[6, 6, 6], 0)
        self.assertEqual(np.count_nonzero(mask), 1)

    def test_dilation_uses_physical_spacing_and_clips_at_boundaries(self):
        mask = np.zeros((9, 9, 9), np.uint8)
        mask[4, 4, 4] = 1
        result = dilate_mask_mm(mask, (3, 1, .5))
        self.assertFalse(result[3].any())
        self.assertEqual(result[4, 6, 4], 1)
        self.assertEqual(result[4, 4, 8], 1)
        self.assertEqual(result[4, 6, 5], 0)
        corner = np.zeros((5, 5, 5), np.uint8)
        corner[0, 0, 0] = 1
        clipped = dilate_mask_mm(corner, (1, 1, 1))
        self.assertFalse(clipped[-1].any())
        np.testing.assert_array_equal(dilate_mask_mm(mask, (1, 1, 1), 0), mask)
        self.assertFalse(dilate_mask_mm(np.zeros_like(mask), (1, 1, 1)).any())

    def test_invalid_spacing_and_multilabel_masks_are_not_silently_modified(self):
        mask = np.zeros((5, 5, 5), np.uint8)
        for spacing in ((0, 1, 1), (-1, 1, 1), (float('nan'), 1, 1), (1, 1)):
            with self.assertRaises(ValueError):
                dilate_mask_mm(mask, spacing)
        mask[2, 2, 2] = 2
        with self.assertRaises(ValueError):
            dilate_mask_mm(mask, (1, 1, 1))

    def test_compressed_snapshot_is_independent_and_preserves_dtype(self):
        mask = np.zeros((7, 8, 9), np.uint16)
        mask[2:4, 3:5, 4:6] = 1
        original = mask.copy()
        snapshot = MaskSnapshot.capture(mask, 3)
        mask[:] = 0
        self.assertEqual(snapshot.extra_count, 3)
        self.assertEqual(snapshot.restore().dtype, np.uint16)
        np.testing.assert_array_equal(snapshot.restore(), original)

    def test_refinement_seeds_current_mask_then_replays_all_lassos_once(self):
        class Session:
            supports_initial_label = True
            def __init__(self):
                self.events = []
            def reset_interactions(self):
                self.events.append('reset')
            def set_target_buffer(self, target):
                self.target = target
            def add_initial_seg_interaction(self, mask, run_prediction):
                self.events.append('seed')
                self.target[:] = mask
            def add_lasso_interaction(self, crop, include, run_prediction, interaction_bbox):
                self.events.append(('lasso', include, run_prediction, interaction_bbox))
                if run_prediction:
                    self.target[0, 0, 0] = 1
        session = Session()
        prompts = [LassoPrompt(axis, np.ones(tuple(1 if dim == axis else 2 for dim in range(3)), np.uint8),
                               [[1, 2] if dim == axis else [1, 3] for dim in range(3)])
                   for axis in (0, 1, 2, 0, 1)]
        mask = np.zeros((5, 5, 5), np.uint8)
        mask[4, 4, 4] = 1
        result, snapshot = refine_with_lassos(session, mask, prompts, 1)
        self.assertEqual(session.events[:2], ['reset', 'seed'])
        self.assertEqual([event[2] for event in session.events[2:]], [False] * 4 + [True])
        self.assertEqual(result[4, 4, 4], 1)
        self.assertEqual(result[0, 0, 0], 1)
        self.assertEqual(mask[0, 0, 0], 0)
        np.testing.assert_array_equal(snapshot.restore(), mask)


if __name__ == '__main__':
    unittest.main()
