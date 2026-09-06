import unittest
from fastapi.testclient import TestClient
from src.api.app import app


class TestAgentAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_supervisor_chat_endpoint(self):
        payload = {
            "message": "Generate the daily toolbox briefing for SITE-NORTH on excavator swing zones",
            "site_id": "SITE-NORTH"
        }
        resp = self.client.post("/api/v1/agent/supervisor/chat", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("response", data)
        self.assertIn("tool_executions", data)
        self.assertGreaterEqual(len(data["tool_executions"]), 1)
        self.assertIn("Toolbox", data["response"])

    def test_pipeline1_compile_manifest_endpoint(self):
        payload = {
            "permit_text": "Permit #512: Trench excavation at Pier B4. Heavy articulated loader active. Spotter present.",
            "site_id": "SITE-NORTH",
            "shift_date": "2026-09-07"
        }
        resp = self.client.post("/api/v1/agent/pipeline1/compile_manifest", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("shift_id", data)
        self.assertGreater(data["tasks_count"], 0)
        self.assertIn("SUCCESS_DISPATCHED", data["dispatch_status"])
        self.assertIsNotNone(data["manifest"])

    def test_pipeline2_adjudicate_endpoint(self):
        payload = {
            "event_id": "INC-TEST-API-001",
            "telemetry_json": {"min_ttc": 1.2, "p_col": 0.95},
            "shift_context": "Worker inside excavator blind spot without spotter"
        }
        resp = self.client.post("/api/v1/agent/pipeline2/adjudicate", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["event_id"], "INC-TEST-API-001")
        self.assertIsNotNone(data["verdict"])
        self.assertEqual(data["verdict"]["verdict"], "TRUE_POSITIVE")
        self.assertIn("archive_path", data)

    def test_toolbox_talk_get_endpoint(self):
        resp = self.client.get("/api/v1/agent/toolbox_talk/SITE-EAST?focus_hazard=Crane%20Lifting")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["site_id"], "SITE-EAST")
        self.assertIn("osha_standard", data)
        self.assertIn("field_protocols", data)

    def test_active_learning_queue_endpoint(self):
        resp = self.client.get("/api/v1/agent/active_learning/queue")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_curated", data)
        self.assertIn("high_priority_count", data)


if __name__ == "__main__":
    unittest.main()
