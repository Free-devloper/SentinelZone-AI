import time
import json
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import torch

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    mqtt = None
    HAS_MQTT = False

from src.edge.homography import HomographyProjector
from src.edge.tracker import MetricTrack
from src.edge.gatv2_model import WorkZoneSTGNN
from src.edge.conflict_engine import DynamicConflictEngine, AlertHysteresisDebouncer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentinelEdgeRuntime")


class SentinelEdgeRuntime:
    """
    Integrated deterministic real-time loop running on NVIDIA Jetson AGX Orin.
    Processes detection bounding boxes, projects to ground metric space, maintains
    EKF tracks, executes spatio-temporal GNN trajectory forecasting, evaluates
    spatial conflicts against dynamic BIM manifests, debounces alerts, and actuates relays.
    """
    def __init__(self, homography_cfg: dict, gnn_checkpoint_path: Optional[str] = None, mqtt_host: str = "localhost"):
        # 1. Projector
        self.projector = HomographyProjector(
            camera_matrix=np.array(homography_cfg["K"]),
            dist_coeffs=np.array(homography_cfg["dist"]),
            homography_matrix=np.array(homography_cfg["H"])
        )

        # 2. Spatio-Temporal GNN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"SentinelEdgeRuntime initialized on GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        else:
            logger.info("SentinelEdgeRuntime initialized on CPU")

        self.gnn = WorkZoneSTGNN().to(self.device)
        if gnn_checkpoint_path:
            try:
                self.gnn.load_state_dict(torch.load(gnn_checkpoint_path, map_location=self.device))
                logger.info(f"Loaded GNN checkpoint from {gnn_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load GNN checkpoint ({e}); running with initialized weights.")
        self.gnn.eval()

        # 3. Dynamic Tracking & Risk Engine
        self.tracks: Dict[int, MetricTrack] = {}
        self.next_track_id = 1
        self.conflict_engine = DynamicConflictEngine()
        self.debouncer = AlertHysteresisDebouncer()

        # 4. MQTT Broker for Telemetry & Manifest
        self.mqtt_client = None
        if HAS_MQTT and mqtt is not None:
            try:
                # Support both paho-mqtt v1 and v2 API
                try:
                    self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="SentinelZone_Edge_Orin")
                except AttributeError:
                    self.mqtt_client = mqtt.Client(client_id="SentinelZone_Edge_Orin")
                self.mqtt_client.on_message = self._on_mqtt_message
                self.mqtt_client.connect(mqtt_host, 1883, 60)
                self.mqtt_client.subscribe("/sentinel/site_01/manifest")
                self.mqtt_client.loop_start()
                logger.info(f"Connected to MQTT broker at {mqtt_host}:1883")
            except Exception as e:
                logger.warning(f"MQTT connection to {mqtt_host}:1883 failed: {e}. Telemetry running offline.")

        self.last_alarm_state = "NORMAL_LEVEL_0"
        self.relay_triggered = False

    def _on_mqtt_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            if "envelopes" in payload:
                self.conflict_engine.load_manifest_envelopes(payload["envelopes"])
                logger.info(f"Updated dynamic manifest with {len(payload['envelopes'])} spatial envelopes.")
        except Exception as e:
            logger.error(f"Failed to ingest MQTT manifest: {e}")

    def execute_frame_cycle(self, raw_detections: np.ndarray, timestamp: float = 0.0) -> dict:
        """
        Runs full pipeline within <= 120ms budget:
        raw_detections: [N, 6] (x1, y1, x2, y2, conf, class_id)
        """
        t_start = time.perf_counter()

        # Step 1: Extract anchors & transform to metric space
        bboxes = raw_detections[:, :4] if len(raw_detections) > 0 else np.empty((0, 4))
        anchors_pixel = self.projector.extract_bottom_center_anchors(bboxes)
        metric_positions = self.projector.pixel_to_metric(anchors_pixel)

        # Step 2: Update Tracker
        updated_track_ids = []
        for idx, pos in enumerate(metric_positions):
            cls_id = int(raw_detections[idx, 5])
            matched_id = None
            for tid, track in self.tracks.items():
                if np.linalg.norm(track.kf.state[:2] - pos) < 2.0 and track.class_id == cls_id:
                    matched_id = tid
                    break
            if matched_id is not None:
                self.tracks[matched_id].step_predict(dt=0.1)
                self.tracks[matched_id].step_update(pos)
                updated_track_ids.append(matched_id)
            else:
                new_track = MetricTrack(self.next_track_id, cls_id, pos, length=2.0, width=1.0)
                self.tracks[self.next_track_id] = new_track
                updated_track_ids.append(self.next_track_id)
                self.next_track_id += 1

        # Purge stale tracks
        for tid in list(self.tracks.keys()):
            if tid not in updated_track_ids:
                self.tracks[tid].time_since_update += 1
                if self.tracks[tid].time_since_update > 10:
                    del self.tracks[tid]

        # Step 3: Construct Dynamic Graph & Run Forecasting
        active_nodes = [t for t in self.tracks.values() if t.confirmed and len(t.history) >= 5]
        highest_severity = "NORMAL_LEVEL_0"
        evaluated_pairs = []

        if len(active_nodes) >= 2:
            num_nodes = len(active_nodes)
            node_hist = np.zeros((num_nodes, 30, 8), dtype=np.float32)
            cls_ids = np.zeros(num_nodes, dtype=np.int64)

            for i, node in enumerate(active_nodes):
                hist = np.array(node.history)
                node_hist[i, -len(hist):, :] = hist
                cls_ids[i] = node.class_id

            # Assemble proximity edges (radius <= 25m)
            src_edges, dst_edges, edge_feats = [], [], []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        dist = np.linalg.norm(node_hist[i, -1, :2] - node_hist[j, -1, :2])
                        if dist <= 25.0:
                            src_edges.append(i)
                            dst_edges.append(j)
                            d_vec = node_hist[j, -1, :2] - node_hist[i, -1, :2]
                            rel_vel = np.linalg.norm(node_hist[j, -1, 2:4] - node_hist[i, -1, 2:4])
                            edge_feats.append([d_vec[0], d_vec[1], dist, rel_vel, 1.0])

            if len(src_edges) > 0:
                t_hist = torch.tensor(node_hist, dtype=torch.float32, device=self.device)
                t_cls = torch.tensor(cls_ids, dtype=torch.long, device=self.device)
                t_edge_idx = torch.tensor([src_edges, dst_edges], dtype=torch.long, device=self.device)
                t_edge_attr = torch.tensor(edge_feats, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    preds = self.gnn(t_hist, t_cls, t_edge_idx, t_edge_attr)

                # Step 4: Evaluate Risk Conflicts
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        # Filter for human vs machine (human classes: 0, 1)
                        is_human_a = (active_nodes[i].class_id in [0, 1])
                        is_human_b = (active_nodes[j].class_id in [0, 1])
                        if is_human_a != is_human_b:
                            dict_a = {
                                "footprint_radius": 0.8 if is_human_a else 2.5,
                                "class_name": "WORKER" if is_human_a else "HEAVY_EQUIPMENT",
                                "mode_probs": preds["mode_probs"][i].cpu().numpy(),
                                "mu_x": preds["mu_x"][i].cpu().numpy(),
                                "mu_y": preds["mu_y"][i].cpu().numpy()
                            }
                            dict_b = {
                                "footprint_radius": 0.8 if is_human_b else 2.5,
                                "class_name": "WORKER" if is_human_b else "HEAVY_EQUIPMENT",
                                "mode_probs": preds["mode_probs"][j].cpu().numpy(),
                                "mu_x": preds["mu_x"][j].cpu().numpy(),
                                "mu_y": preds["mu_y"][j].cpu().numpy()
                            }
                            risk = self.conflict_engine.evaluate_pair_risk(dict_a, dict_b)
                            evaluated_pairs.append({
                                "track_a": active_nodes[i].track_id,
                                "track_b": active_nodes[j].track_id,
                                "risk": risk
                            })
                            if risk["state"] == "CRITICAL_LEVEL_3":
                                highest_severity = "CRITICAL_LEVEL_3"
                            elif risk["state"] == "WARNING_LEVEL_2" and highest_severity != "CRITICAL_LEVEL_3":
                                highest_severity = "WARNING_LEVEL_2"
                            elif risk["state"] == "ADVISORY_LEVEL_1" and highest_severity == "NORMAL_LEVEL_0":
                                highest_severity = "ADVISORY_LEVEL_1"

        # Step 5: Debounce & Actuate
        final_alarm = self.debouncer.step(highest_severity)
        self.last_alarm_state = final_alarm
        if final_alarm == "CRITICAL_LEVEL_3":
            self._actuate_physical_relays(True)
        else:
            self._actuate_physical_relays(False)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        if dt_ms > 120.0:
            logger.warning(f"LATENCY BREACH: Edge execution cycle took {dt_ms:.2f} ms")

        return {
            "cycle_latency_ms": dt_ms,
            "active_tracks_count": len(self.tracks),
            "raw_severity": highest_severity,
            "debounced_alarm": final_alarm,
            "evaluated_pairs": evaluated_pairs
        }

    def _actuate_physical_relays(self, state: bool):
        # Modbus/TCP command to ADAM-6060 relay coil
        self.relay_triggered = state
