"""Regression checks for Grad-CAM boxes and SAM mask selection."""
import unittest

import cv2
import numpy as np

from backend.segmentation import SIZE, PreviewGeometry, automatic_box, select_mask, shared_cam_focus


class CandidateTests(unittest.TestCase):
    def test_non_square_geometry_preserves_coordinates_and_padding(self):
        for width, height in ((448, 224), (224, 448)):
            geometry = PreviewGeometry(width,height)
            x, y, w, h = geometry.rect
            original = np.zeros((height,width), dtype=np.uint8)
            original[height//4:height//2,width//4:width//2] = 1
            projected = geometry.project(original, cv2.INTER_NEAREST_EXACT)
            ys,xs = np.where(projected)
            self.assertEqual((xs.min(),ys.min()), (x+w//4,y+h//4))
            np.testing.assert_allclose(geometry.native_points([[xs.min(),ys.min()]]), [[width/4,height/4]])
            self.assertFalse(geometry.contains(0,0))

    def test_separate_hotspots_do_not_share_a_box(self):
        foreground = np.ones((SIZE, SIZE), dtype=bool)
        heatmap = np.zeros((SIZE, SIZE), dtype=np.float32)
        heatmap[45:75, 40:70] = 1
        heatmap[145:175, 140:170] = 0.75
        x1, y1, x2, y2 = automatic_box(heatmap, foreground)
        self.assertLess(x1, 40)
        self.assertLess(y1, 45)
        self.assertLess(x2, 100)
        self.assertLess(y2, 100)
        heatmap[145:175, 140:170] = 1
        self.assertIsNone(automatic_box(heatmap, foreground))

    def test_weak_localized_peak_can_form_box_but_flat_maps_cannot(self):
        foreground = np.ones((SIZE, SIZE), dtype=bool)
        heatmap = np.full((SIZE, SIZE), 0.02, dtype=np.float32)
        heatmap[70:94, 128:153] = 0.09
        self.assertIsNotNone(automatic_box(heatmap, foreground))
        self.assertIsNone(automatic_box(heatmap * 0.01, foreground))
        heatmap.fill(0.085)
        heatmap[70:94, 128:153] = 0.09
        self.assertIsNone(automatic_box(heatmap, foreground))

    def test_shared_focus_requires_overlap_between_stages(self):
        foreground = np.ones((SIZE, SIZE), dtype=bool)
        first = np.zeros((SIZE, SIZE), dtype=np.float32)
        second = np.zeros_like(first)
        first[40:66, 40:66] = 1
        second[100:126, 100:126] = 1
        self.assertIsNone(shared_cam_focus(first, second, foreground))
        second[44:70, 44:70] = 0.9
        x1, y1, x2, y2 = shared_cam_focus(first, second, foreground)
        self.assertTrue(x1 < 44 and y1 < 44 and 65 < x2 < 85 and 65 < y2 < 85)

    def test_mask_quality_cannot_override_poor_box_overlap(self):
        foreground = np.ones((SIZE, SIZE), dtype=bool)
        broad = np.zeros_like(foreground)
        broad[10:180, 10:180] = True
        local = np.zeros_like(foreground)
        local[45:60, 45:60] = True
        self.assertIsNone(select_mask([broad], [0.99], [40, 40, 70, 70], foreground))
        mask, score = select_mask([broad, local], [0.99, 0.85], [40, 40, 70, 70], foreground)
        np.testing.assert_array_equal(mask, local)
        self.assertEqual(score, 0.85)


if __name__ == "__main__":
    unittest.main()
