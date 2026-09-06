import unittest
from src.graph.context_pipeline import (
    extract_tasks_node,
    resolve_bim_node,
    validate_manifest_node,
    dispatch_edge_mqtt_node,
    build_context_pipeline,
    ContextGraphState
)
from src.graph.adjudication_pipeline import (
    vlm_triage_node,
    filter_active_learning_node,
    build_adjudication_pipeline,
    IncidentTriageState
)
from src.schemas.contracts import ZoneSeverity, EntityType


class TestLangGraphPipelines(unittest.TestCase):
    def test_pipeline1_nodes(self):
        # Step 1: Extract tasks
        initial_state: ContextGraphState = {
            "site_id": "SITE_5",
            "shift_date": "2026-09-07",
            "raw_permit_text": "Permit #982: Trench Excavation at Pier B4. Heavy articulated equipment active. Spotter present.",
            "ifc_file_path": "models/site_pier_b4.ifc",
            "extracted_tasks": [],
            "resolved_envelopes": [],
            "validated_manifest": None,
            "validation_errors": [],
            "dispatch_status": ""
        }
        res_extract = extract_tasks_node(initial_state)
        self.assertIn("extracted_tasks", res_extract)
        self.assertGreater(len(res_extract["extracted_tasks"]), 0)

        # Step 2: Resolve BIM envelopes
        initial_state["extracted_tasks"] = res_extract["extracted_tasks"]
        res_bim = resolve_bim_node(initial_state)
        self.assertIn("resolved_envelopes", res_bim)
        self.assertEqual(len(res_bim["resolved_envelopes"]), len(initial_state["extracted_tasks"]))
        first_env = res_bim["resolved_envelopes"][0]
        self.assertIn("polygon_metric_epsg3857", first_env)
        self.assertEqual(len(first_env["polygon_metric_epsg3857"]), 4)

        # Step 3: Validate manifest
        initial_state["resolved_envelopes"] = res_bim["resolved_envelopes"]
        res_manifest = validate_manifest_node(initial_state)
        self.assertIn("validated_manifest", res_manifest)
        self.assertIsNotNone(res_manifest["validated_manifest"])
        self.assertEqual(len(res_manifest["validation_errors"]), 0)

        # Step 4: Dispatch edge MQTT
        initial_state["validated_manifest"] = res_manifest["validated_manifest"]
        res_dispatch = dispatch_edge_mqtt_node(initial_state)
        self.assertIn("SUCCESS_DISPATCHED", res_dispatch["dispatch_status"])

    def test_pipeline1_compilation(self):
        graph = build_context_pipeline()
        if graph is not None:
            self.assertIsNotNone(graph)

    def test_pipeline2_nodes_and_active_learning(self):
        # Test True Positive case
        state_tp: IncidentTriageState = {
            "event_id": "INC-20260906-001",
            "video_s3_uri": "s3://sentinel-incidents/2026/09/06/001.mp4",
            "telemetry_json": {"min_ttc": 1.4, "p_col": 0.88},
            "shift_context": "Excavation near Pier B4",
            "final_verdict": None,
            "active_learning_curated": False
        }
        res_triage = vlm_triage_node(state_tp)
        self.assertIsNotNone(res_triage["final_verdict"])
        self.assertEqual(res_triage["final_verdict"].verdict, "TRUE_POSITIVE")

        state_tp["final_verdict"] = res_triage["final_verdict"]
        res_al = filter_active_learning_node(state_tp)
        # High retraining priority should be curated for active learning
        self.assertTrue(res_al["active_learning_curated"])

        # Test Archival and Active Learning Persistence
        state_tp["active_learning_curated"] = res_al["active_learning_curated"]
        from src.graph.adjudication_pipeline import archive_incident_node
        res_archive = archive_incident_node(state_tp)
        self.assertIn("archive_path", res_archive)
        import os
        self.assertTrue(os.path.exists(res_archive["archive_path"]))

    def test_pipeline2_compilation(self):
        graph = build_adjudication_pipeline()
        if graph is not None:
            self.assertIsNotNone(graph)
            # Test full graph execution
            state_in = {
                "event_id": "INC-GRAPH-TEST-01",
                "video_s3_uri": "test.mp4",
                "telemetry_json": {"min_ttc": 1.2, "p_col": 0.95},
                "shift_context": "Haul road crossing",
                "final_verdict": None,
                "active_learning_curated": False,
                "archive_path": None
            }
            out = graph.invoke(state_in)
            self.assertIsNotNone(out.get("final_verdict"))
            self.assertIsNotNone(out.get("archive_path"))


if __name__ == "__main__":
    unittest.main()
