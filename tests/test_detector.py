import unittest
import os
import json
import numpy as np
from pathlib import Path

from src.perception.dataset_downloader import (
    create_sample_construction_dataset,
    download_construction_safety_dataset,
    ROBOFLOW_CLASSES,
    SENTINEL_CLASS_MAPPING
)
from src.perception.detector import ConstructionSafetyDetector
from src.edge.edge_runtime import SentinelEdgeRuntime


class TestConstructionDetector(unittest.TestCase):
    def setUp(self):
        self.output_dir = "data/test_dataset"
        self.detector = ConstructionSafetyDetector()

    def tearDown(self):
        # Clean up test dataset
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_sample_dataset_generation(self):
        yaml_path = create_sample_construction_dataset(self.output_dir)
        self.assertTrue(os.path.exists(yaml_path))

        # Check directory structure
        p = Path(self.output_dir)
        self.assertTrue((p / "images" / "train").exists())
        self.assertTrue((p / "labels" / "train").exists())
        self.assertTrue((p / "images" / "val").exists())
        self.assertTrue((p / "labels" / "val").exists())

        # Check generated files count
        train_images = list((p / "images" / "train").glob("*.jpg"))
        self.assertGreater(len(train_images), 0)

    def test_class_mappings(self):
        # Verify Person -> WORKER (0)
        self.assertEqual(SENTINEL_CLASS_MAPPING[5], 0)
        # Verify machinery -> HEAVY_EQUIPMENT (2)
        self.assertEqual(SENTINEL_CLASS_MAPPING[8], 2)
        # Verify vehicle -> LIGHT_VEHICLE (3)
        self.assertEqual(SENTINEL_CLASS_MAPPING[9], 3)
        self.assertIn("Hardhat", ROBOFLOW_CLASSES)
        self.assertIn("Safety Vest", ROBOFLOW_CLASSES)

    def test_detector_inference_format(self):
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dets, ppe = self.detector.detect(dummy_frame)

        self.assertIsInstance(dets, np.ndarray)
        self.assertEqual(dets.ndim, 2)
        self.assertEqual(dets.shape[1], 6)
        self.assertGreater(len(dets), 0)

        # Check that class IDs are valid SentinelZone IDs (0, 1, 2, 3)
        class_ids = dets[:, 5]
        for cid in class_ids:
            self.assertIn(int(cid), [0, 1, 2, 3])

    def test_detector_to_runtime_integration(self):
        with open("config/default_config.json", "r") as f:
            cfg = json.load(f)

        runtime = SentinelEdgeRuntime(
            homography_cfg=cfg["homography"],
            gnn_checkpoint_path=None,
            mqtt_host="127.0.0.1"
        )

        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dets, _ = self.detector.detect(dummy_frame)

        # Pass detector output directly into edge cycle
        cycle_res = runtime.execute_frame_cycle(dets, timestamp=1.0)
        self.assertLessEqual(cycle_res["cycle_latency_ms"], 120.0)
        self.assertGreaterEqual(cycle_res["active_tracks_count"], 1)


if __name__ == "__main__":
    unittest.main()
