import unittest
import json
import numpy as np
from src.edge.edge_runtime import SentinelEdgeRuntime


class TestSentinelEdgeRuntime(unittest.TestCase):
    def setUp(self):
        with open("config/default_config.json", "r") as f:
            cfg = json.load(f)
        self.homography_cfg = cfg["homography"]
        self.runtime = SentinelEdgeRuntime(
            homography_cfg=self.homography_cfg,
            gnn_checkpoint_path=None,
            mqtt_host="127.0.0.1"
        )

    def test_empty_detections_cycle(self):
        empty_dets = np.empty((0, 6), dtype=np.float32)
        res = self.runtime.execute_frame_cycle(empty_dets, timestamp=1.0)
        self.assertLessEqual(res["cycle_latency_ms"], 120.0)
        self.assertEqual(res["active_tracks_count"], 0)
        self.assertEqual(res["debounced_alarm"], "NORMAL_LEVEL_0")

    def test_multi_agent_tracking_and_forecasting(self):
        # Frame cycle simulation with 1 worker (cls=0) and 1 heavy loader (cls=2)
        # Class IDs: 0 = WORKER, 2 = HEAVY_EQUIPMENT
        for frame_idx in range(6):
            dets = np.array([
                # Worker bounding box in pixels [x1, y1, x2, y2, conf, cls]
                [500.0 + frame_idx * 5, 600.0, 540.0 + frame_idx * 5, 720.0, 0.95, 0],
                # Heavy loader bounding box in pixels
                [800.0 - frame_idx * 5, 550.0, 950.0 - frame_idx * 5, 750.0, 0.92, 2]
            ], dtype=np.float32)

            res = self.runtime.execute_frame_cycle(dets, timestamp=frame_idx * 0.1)
            # Check latency budget bound (<= 120ms rule)
            self.assertLessEqual(res["cycle_latency_ms"], 120.0)

        # After 6 frames, both tracks should be confirmed and active
        self.assertGreaterEqual(res["active_tracks_count"], 2)


if __name__ == "__main__":
    unittest.main()
