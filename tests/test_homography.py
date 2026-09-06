import unittest
import numpy as np
from src.edge.homography import HomographyProjector


class TestHomographyProjector(unittest.TestCase):
    def setUp(self):
        self.K = np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0]
        ])
        self.dist = np.zeros(5)
        # Identity-like planar homography with 100 pixel = 1 meter scaling
        self.H = np.array([
            [100.0, 0.0, 500.0],
            [0.0, 100.0, 500.0],
            [0.0, 0.0, 1.0]
        ])
        self.projector = HomographyProjector(self.K, self.dist, self.H)

    def test_anchor_extraction(self):
        bboxes = np.array([
            [100.0, 200.0, 300.0, 400.0],
            [50.0, 100.0, 150.0, 250.0]
        ])
        anchors = self.projector.extract_bottom_center_anchors(bboxes)
        expected = np.array([
            [200.0, 400.0],
            [100.0, 250.0]
        ])
        np.testing.assert_allclose(anchors, expected)

    def test_bidirectional_projection(self):
        # Forward metric -> pixel -> metric
        metric_pts = np.array([[5.0, 10.0], [0.0, 0.0], [-3.5, 8.2]])
        pixel_pts = self.projector.metric_to_pixel(metric_pts)
        reconstructed_metric = self.projector.pixel_to_metric(pixel_pts)
        np.testing.assert_allclose(reconstructed_metric, metric_pts, atol=1e-5)

    def test_calibration_drift_stable(self):
        ref_metric = np.array([[2.0, 3.0], [4.0, 5.0]])
        pixel_pts = self.projector.metric_to_pixel(ref_metric)
        is_stable, rmse = self.projector.evaluate_calibration_drift(pixel_pts, ref_metric, threshold_rmse_meters=0.15)
        self.assertTrue(is_stable)
        self.assertLess(rmse, 0.05)

    def test_calibration_drift_unstable(self):
        ref_metric = np.array([[2.0, 3.0], [4.0, 5.0]])
        # Perturb pixels to simulate camera mast vibration or mechanical shift
        pixel_pts = self.projector.metric_to_pixel(ref_metric) + 50.0
        is_stable, rmse = self.projector.evaluate_calibration_drift(pixel_pts, ref_metric, threshold_rmse_meters=0.15)
        self.assertFalse(is_stable)
        self.assertGreater(rmse, 0.15)


if __name__ == "__main__":
    unittest.main()
