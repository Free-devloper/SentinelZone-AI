import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger("SentinelHomography")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


class HomographyProjector:
    """
    Handles bidirectional planar projection between camera pixel coordinates
    and ground-plane metric coordinates (meters). Monitors calibration drift.
    """
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, homography_matrix: np.ndarray):
        assert camera_matrix.shape == (3, 3), "Camera matrix K must be 3x3"
        assert homography_matrix.shape == (3, 3), "Homography matrix H must be 3x3"
        self.K = camera_matrix.astype(np.float64)
        self.dist = dist_coeffs.astype(np.float64)
        self.H = homography_matrix.astype(np.float64)
        self.H_inv = np.linalg.inv(self.H)
        self.is_stable = True

    def pixel_to_metric(self, pixel_points: np.ndarray) -> np.ndarray:
        """
        Transforms pixel coordinates [N, 2] to ground-plane coordinates [N, 2] in meters.
        """
        if len(pixel_points) == 0:
            return np.empty((0, 2), dtype=np.float64)

        if HAS_CV2 and cv2 is not None:
            pts = pixel_points.reshape(-1, 1, 2).astype(np.float32)
            undistorted = cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)
        else:
            undistorted = pixel_points.astype(np.float64).reshape(-1, 2)

        n = len(undistorted)
        homog = np.hstack([undistorted, np.ones((n, 1), dtype=np.float64)])

        world_homog = (self.H_inv @ homog.T).T
        w = world_homog[:, 2:3]
        w = np.where(np.abs(w) < 1e-8, 1e-8, w)
        return world_homog[:, :2] / w

    def metric_to_pixel(self, metric_points: np.ndarray) -> np.ndarray:
        """
        Transforms metric coordinates [N, 2] back to image pixel coordinates [N, 2].
        """
        if len(metric_points) == 0:
            return np.empty((0, 2), dtype=np.float64)

        n = len(metric_points)
        homog = np.hstack([metric_points, np.ones((n, 1), dtype=np.float64)])
        img_homog = (self.H @ homog.T).T
        w = img_homog[:, 2:3]
        w = np.where(np.abs(w) < 1e-8, 1e-8, w)
        return img_homog[:, :2] / w

    def extract_bottom_center_anchors(self, bboxes: np.ndarray) -> np.ndarray:
        """
        Extracts bottom-center coordinate for each bounding box [x1, y1, x2, y2].
        """
        if len(bboxes) == 0:
            return np.empty((0, 2), dtype=np.float64)
        u_mid = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        v_bottom = bboxes[:, 3]
        return np.column_stack([u_mid, v_bottom])

    def evaluate_calibration_drift(
        self, 
        observed_pixel_anchors: np.ndarray, 
        reference_metric_anchors: np.ndarray, 
        threshold_rmse_meters: float = 0.15
    ) -> Tuple[bool, float]:
        """
        Monitors invariant ground survey targets (e.g. ArUco markers, static bollards).
        """
        projected_metric = self.pixel_to_metric(observed_pixel_anchors)
        errors = np.linalg.norm(projected_metric - reference_metric_anchors, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        self.is_stable = (rmse <= threshold_rmse_meters)
        if not self.is_stable:
            logger.error(f"CRITICAL: Calibration drift RMSE = {rmse:.4f}m exceeds {threshold_rmse_meters}m")
        return self.is_stable, rmse
