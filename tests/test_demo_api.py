import unittest
from fastapi.testclient import TestClient
from src.api.app import app


class TestDemoAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertIn("cuda", data["device"])

    def test_dashboard_serve(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("SENTINELZONE-AI", resp.text)
        self.assertIn("vision-canvas", resp.text)
        self.assertIn("webgl-wrapper", resp.text)

    def test_demo_scenarios(self):
        resp = self.client.get("/api/v1/demo/scenarios")
        self.assertEqual(resp.status_code, 200)
        scenarios = resp.json()
        self.assertGreaterEqual(len(scenarios), 8)
        scenario_ids = [s["id"] for s in scenarios]
        self.assertIn("scenario_worker_in_excavator_blind_spot", scenario_ids)
        self.assertIn("scenario_exca_near_miss", scenario_ids)
        self.assertIn("scenario_worker_and_excavator_near_barrier", scenario_ids)
        self.assertIn("scenario_user_field", scenario_ids)
        self.assertIn("scenario_user_crew", scenario_ids)
        self.assertIn("scenario_user_workforce", scenario_ids)
        self.assertIn("scenario_1_real_site", scenario_ids)
        self.assertIn("scenario_2_roboflow_test", scenario_ids)
        self.assertIn("scenario_3_roboflow_valid", scenario_ids)

    def test_telemetry_endpoint(self):
        resp = self.client.get("/api/v1/demo/telemetry/scenario_1_real_site")
        self.assertEqual(resp.status_code, 200)
        frames = resp.json()
        self.assertGreater(len(frames), 20)
        first_frame = frames[0]
        self.assertIn("annotations", first_frame)
        self.assertIn("tracks_3d", first_frame)
        self.assertIn("conflict", first_frame)
        self.assertIn("comparison", first_frame)

    def test_single_frame_endpoint(self):
        resp = self.client.get("/api/v1/demo/frame/scenario_1_real_site/5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["frame_index"], 5)
        self.assertIn("edge_cycle_latency_ms", data)
        self.assertLess(data["edge_cycle_latency_ms"], 120.0)

    def test_frame_image_endpoint(self):
        resp = self.client.get("/api/v1/demo/frame_image/scenario_1_real_site/0")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/jpeg")
        self.assertGreater(len(resp.content), 1000)

    def test_adjudication_submission(self):
        payload = {
            "event_id": "INC-TEST-001",
            "verdict": "TRUE_POSITIVE",
            "confidence": 1.0,
            "spotter_verified": True,
            "worker_awareness_observed": False,
            "root_cause_summary": "Test automated verification",
            "retraining_priority": "LOW",
            "recommended_mitigation": "Automated verification"
        }
        resp = self.client.post("/api/v1/adjudication/submit", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ADJUDICATION_STORED")


if __name__ == "__main__":
    unittest.main()
