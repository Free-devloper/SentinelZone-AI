from enum import Enum
from typing import List, Tuple, Optional, Literal
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    WORKER = "WORKER"
    HEAVY_RIGID = "HEAVY_RIGID"
    HEAVY_ARTICULATED = "HEAVY_ARTICULATED"
    LIGHT_VEHICLE = "LIGHT_VEHICLE"
    SPOTTER = "SPOTTER"


class ZoneSeverity(str, Enum):
    CRITICAL_EXCLUSION = "CRITICAL_EXCLUSION"
    WARNING_BUFFER = "WARNING_BUFFER"
    AUTHORIZED_ACTIVITY = "AUTHORIZED_ACTIVITY"


class DynamicEnvelopeConfig(BaseModel):
    envelope_id: str
    zone_name: str
    activity_type: str
    valid_from: str
    valid_until: str
    polygon_metric_epsg3857: List[Tuple[float, float]] = Field(
        description="Metric ground-plane boundary points (X, Y) relative to site origin"
    )
    severity: ZoneSeverity
    exempt_entities: List[EntityType] = Field(default_factory=list)
    ttc_multiplier: float = Field(default=1.0, ge=0.5, le=2.5)
    max_allowable_speed_ms: Optional[float] = None


class ShiftSafetyManifest(BaseModel):
    shift_id: str
    site_id: str
    compiled_at: str
    envelopes: List[DynamicEnvelopeConfig]


class IncidentTriageVerdict(BaseModel):
    event_id: str
    verdict: Literal["TRUE_POSITIVE", "FALSE_POSITIVE", "CONTROLLED_WORK"]
    confidence: float = Field(ge=0.0, le=1.0)
    spotter_verified: bool
    worker_awareness_observed: bool
    root_cause_summary: str
    retraining_priority: Literal["LOW", "MEDIUM", "HIGH"]
    recommended_mitigation: str
