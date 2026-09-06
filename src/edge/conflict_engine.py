import numpy as np
from shapely.geometry import Polygon, Point
from typing import List, Dict, Any, Tuple


class DynamicConflictEngine:
    """
    Evaluates spatial risk, computes multimodal Time-To-Collision (TTC),
    and applies dynamic envelope geofence multipliers and entity exemptions.
    """
    def __init__(self, warning_ttc: float = 2.5, critical_ttc: float = 1.5, prob_threshold: float = 0.65):
        self.warning_ttc = warning_ttc
        self.critical_ttc = critical_ttc
        self.prob_threshold = prob_threshold
        self.active_manifest_envelopes: List[Dict[str, Any]] = []

    def load_manifest_envelopes(self, envelopes: List[Dict[str, Any]]):
        self.active_manifest_envelopes = envelopes

    def evaluate_pair_risk(self, agent_a: dict, agent_b: dict) -> dict:
        min_ttc = float('inf')
        peak_collision_prob = 0.0
        r_col = agent_a["footprint_radius"] + agent_b["footprint_radius"]

        num_modes = agent_a["mode_probs"].shape[0]
        steps = agent_a["mu_x"].shape[1]

        # Multiplier check from dynamic site manifest
        ttc_multiplier = 1.0
        pos_a = Point(agent_a["mu_x"][0, 0], agent_a["mu_y"][0, 0])
        for env in self.active_manifest_envelopes:
            poly_coords = env.get("polygon") or env.get("polygon_metric_epsg3857")
            if poly_coords:
                poly = Polygon(poly_coords)
                if poly.contains(pos_a):
                    exempt_entities = env.get("exempt_entities", [])
                    exempt_str = [e.value if hasattr(e, "value") else str(e) for e in exempt_entities]
                    if agent_a.get("class_name") in exempt_str:
                        return {"state": "NORMAL_LEVEL_0", "min_ttc": -1.0, "p_col": 0.0}
                    ttc_multiplier = max(ttc_multiplier, env.get("ttc_multiplier", 1.0))

        effective_warning_ttc = self.warning_ttc * ttc_multiplier
        effective_critical_ttc = self.critical_ttc * ttc_multiplier

        for m_a in range(num_modes):
            prob_a = float(agent_a["mode_probs"][m_a])
            path_a = np.column_stack([agent_a["mu_x"][m_a], agent_a["mu_y"][m_a]])

            for m_b in range(num_modes):
                prob_b = float(agent_b["mode_probs"][m_b])
                path_b = np.column_stack([agent_b["mu_x"][m_b], agent_b["mu_y"][m_b]])

                joint_prob = prob_a * prob_b
                distances = np.linalg.norm(path_a - path_b, axis=1)
                breaches = np.where(distances <= r_col)[0]

                if len(breaches) > 0:
                    first_breach = breaches[0]
                    ttc = (first_breach + 1) * 0.1
                    if ttc < min_ttc:
                        min_ttc = ttc
                    peak_collision_prob = max(peak_collision_prob, joint_prob)

        if min_ttc <= effective_critical_ttc and peak_collision_prob >= self.prob_threshold:
            state = "CRITICAL_LEVEL_3"
        elif min_ttc <= effective_warning_ttc and peak_collision_prob >= (self.prob_threshold * 0.6):
            state = "WARNING_LEVEL_2"
        elif min_ttc <= 5.0 and peak_collision_prob >= 0.20:
            state = "ADVISORY_LEVEL_1"
        else:
            state = "NORMAL_LEVEL_0"

        return {
            "state": state,
            "min_ttc": min_ttc if min_ttc != float('inf') else -1.0,
            "p_col": float(peak_collision_prob)
        }


class AlertHysteresisDebouncer:
    """
    Debounces multi-frame conflict severity transitions.
    Implements asymmetric hysteresis with fast escalation (3 frames)
    and slow de-escalation (15 frames) to avoid flapping.
    """
    def __init__(self, escalation_frames: int = 3, deescalation_frames: int = 15):
        self.escalation_req = escalation_frames
        self.deescalation_req = deescalation_frames
        self.current_state = "NORMAL_LEVEL_0"
        self.high_state_counter = 0
        self.zero_state_counter = 0

    def step(self, candidate_state: str) -> str:
        level_map = {"NORMAL_LEVEL_0": 0, "ADVISORY_LEVEL_1": 1, "WARNING_LEVEL_2": 2, "CRITICAL_LEVEL_3": 3}
        current_lvl = level_map[self.current_state]
        candidate_lvl = level_map[candidate_state]

        if candidate_lvl > current_lvl:
            self.high_state_counter += 1
            self.zero_state_counter = 0
            if self.high_state_counter >= self.escalation_req:
                self.current_state = candidate_state
                self.high_state_counter = 0
        elif candidate_lvl < current_lvl:
            self.zero_state_counter += 1
            self.high_state_counter = 0
            if self.zero_state_counter >= self.deescalation_req:
                self.current_state = candidate_state
                self.zero_state_counter = 0
        else:
            self.high_state_counter = 0
            self.zero_state_counter = 0

        return self.current_state
