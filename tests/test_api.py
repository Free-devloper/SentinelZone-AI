import unittest
from fastapi.testclient import TestClient
from src.api.app import app


class TestFastAPIGateway(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertEqual(data["system"], "SentinelZone-AI")

    def test_publish_edge_incident(self):
        incident_packet = {
            "event_id": "INC-TEST-001",
            "timestamp": 1725642000.0,
            "severity": "CRITICAL_LEVEL_3",
            "min_ttc": 1.25,
            "p_col": 0.89,
            "agents": ["Worker #101", "Wheel Loader 02"]
        }
        response = self.client.post("/api/v1/incidents/publish", json=incident_packet)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "BROADCAST_SUCCESS")
        self.assertEqual(response.json()["received_event"], "INC-TEST-001")

    def test_submit_adjudication(self):
        payload = {
            "event_id": "INC-TEST-001",
            "verdict": "TRUE_POSITIVE",
            "confidence": 0.97,
            "spotter_verified": False,
            "worker_awareness_observed": False,
            "root_cause_summary": "Uncontrolled machine reversing in worker pathway.",
            "retraining_priority": "HIGH",
            "recommended_mitigation": "Install physical demarcation line."
        }
        response = self.client.post("/api/v1/adjudication/submit", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ADJUDICATION_STORED")
        self.assertEqual(response.json()["event_id"], "INC-TEST-001")


if __name__ == "__main__":
    unittest.main()
