import unittest
from pydantic import ValidationError
from src.schemas.contracts import (
    EntityType,
    ZoneSeverity,
    DynamicEnvelopeConfig,
    ShiftSafetyManifest,
    IncidentTriageVerdict,
)


class TestOperationalContracts(unittest.TestCase):
    def test_dynamic_envelope_valid(self):
        env = DynamicEnvelopeConfig(
            envelope_id="ENV_SITE5_PIER_B4",
            zone_name="Pier B4 Excavation",
            activity_type="Trench Excavation",
            valid_from="2026-09-07T07:00:00Z",
            valid_until="2026-09-07T17:00:00Z",
            polygon_metric_epsg3857=[(10.0, 5.0), (25.0, 5.0), (25.0, 15.0), (10.0, 15.0)],
            severity=ZoneSeverity.CRITICAL_EXCLUSION,
            exempt_entities=[EntityType.SPOTTER],
            ttc_multiplier=1.2,
            max_allowable_speed_ms=2.5
        )
        self.assertEqual(env.envelope_id, "ENV_SITE5_PIER_B4")
        self.assertEqual(env.severity, ZoneSeverity.CRITICAL_EXCLUSION)
        self.assertIn(EntityType.SPOTTER, env.exempt_entities)

    def test_dynamic_envelope_invalid_multiplier(self):
        with self.assertRaises(ValidationError):
            DynamicEnvelopeConfig(
                envelope_id="ENV_INVALID",
                zone_name="Invalid Zone",
                activity_type="Test",
                valid_from="2026-09-07T07:00:00Z",
                valid_until="2026-09-07T17:00:00Z",
                polygon_metric_epsg3857=[(0.0, 0.0), (1.0, 1.0)],
                severity=ZoneSeverity.WARNING_BUFFER,
                ttc_multiplier=5.0  # Exceeds max 2.5
            )

    def test_shift_safety_manifest(self):
        env = DynamicEnvelopeConfig(
            envelope_id="ENV_01",
            zone_name="Haul Corridor",
            activity_type="Transit",
            valid_from="2026-09-07T07:00:00Z",
            valid_until="2026-09-07T17:00:00Z",
            polygon_metric_epsg3857=[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
            severity=ZoneSeverity.WARNING_BUFFER
        )
        manifest = ShiftSafetyManifest(
            shift_id="SHIFT_20260907_SITE01",
            site_id="SITE01",
            compiled_at="2026-09-06T18:00:00Z",
            envelopes=[env]
        )
        self.assertEqual(manifest.site_id, "SITE01")
        self.assertEqual(len(manifest.envelopes), 1)

    def test_incident_triage_verdict(self):
        verdict = IncidentTriageVerdict(
            event_id="INC-001",
            verdict="TRUE_POSITIVE",
            confidence=0.98,
            spotter_verified=False,
            worker_awareness_observed=False,
            root_cause_summary="Loader breached exclusion radius.",
            retraining_priority="HIGH",
            recommended_mitigation="Enforce dedicated spotter protocol."
        )
        self.assertEqual(verdict.verdict, "TRUE_POSITIVE")
        self.assertEqual(verdict.confidence, 0.98)


if __name__ == "__main__":
    unittest.main()
