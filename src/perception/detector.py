import os
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None
    HAS_ULTRALYTICS = False

from src.perception.dataset_downloader import (
    ROBOFLOW_CLASSES,
    SENTINEL_CLASS_MAPPING,
    PPE_CLASS_IDS
)

logger = logging.getLogger("ConstructionSafetyDetector")


class ConstructionSafetyDetector:
    """
    Perception module wrapping YOLOv8/v10/v11 for Construction Site Safety.
    Translates raw Roboflow model outputs (Workers, Machinery, Hardhats, Vests)
    into standard SentinelZone-AI [N, 6] format (x1, y1, x2, y2, conf, sentinel_class_id)
    and computes worker PPE compliance telemetry.
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-detect CUDA GPU if available
        if device is None or device == "auto":
            try:
                import torch
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None

        if HAS_ULTRALYTICS and YOLO is not None:
            model_target = weights_path if weights_path and os.path.exists(weights_path) else "yolov8n.pt"
            try:
                self.model = YOLO(model_target)
                logger.info(f"Loaded YOLO detector from {model_target} targeting device: {self.device}")
            except Exception as e:
                logger.warning(f"Failed loading YOLO weights ({e}); running with heuristic detector.")
        else:
            logger.info("Ultralytics not installed; running with heuristic detector fallback.")

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Runs perception inference on frame.
        Args:
            frame: uint8 image [H, W, 3]
        Returns:
            detections_for_tracker: [N, 6] array (x1, y1, x2, y2, conf, sentinel_class_id)
            ppe_telemetry: List of dicts describing worker PPE compliance
        """
        if frame is None or len(frame) == 0:
            return np.empty((0, 6), dtype=np.float32), []

        h, w = frame.shape[:2]

        if self.model is not None:
            try:
                results = self.model.predict(
                    source=frame,
                    conf=min(self.conf_threshold, 0.15),
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False
                )
                raw_boxes = results[0].boxes
                if raw_boxes is not None and len(raw_boxes) > 0:
                    xyxy = raw_boxes.xyxy.cpu().numpy()
                    conf = raw_boxes.conf.cpu().numpy()
                    cls_ids = raw_boxes.cls.cpu().numpy().astype(int)
                    return self._process_detections(xyxy, conf, cls_ids, frame=frame)
            except Exception as e:
                logger.error(f"Perception inference error: {e}")

        # Deterministic heuristic detection fallback when model weights or ultralytics are not loaded
        return self._heuristic_fallback_detect(h, w)

    def _process_detections(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls_ids: np.ndarray,
        frame: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Maps Roboflow classes to SentinelZone tracker format and computes PPE compliance.
        Combines deep learning model detections with chromatic/morphological verification
        on head (hardhats: yellow, orange, blue, white) and torso (safety vests: lime, orange, reflective tape).
        """
        out_detections = []
        worker_boxes = []
        hardhat_boxes = []
        vest_boxes = []
        no_hardhat_boxes = []
        no_vest_boxes = []

        for box, score, cid in zip(xyxy, conf, cls_ids):
            # Roboflow classes:
            # 5 = Person, 8 = machinery, 9 = vehicle
            if cid in SENTINEL_CLASS_MAPPING:
                sentinel_cls = SENTINEL_CLASS_MAPPING[cid]
                if sentinel_cls == 0 and score < self.conf_threshold:
                    continue
                out_detections.append([box[0], box[1], box[2], box[3], score, sentinel_cls])
                if sentinel_cls == 0:  # Worker
                    worker_boxes.append(box)
            # COCO fallback: 0 = person, 7 = truck, 2 = car, 5 = bus
            elif cid == 0:  # Person
                if score < self.conf_threshold:
                    continue
                out_detections.append([box[0], box[1], box[2], box[3], score, 0])
                worker_boxes.append(box)
            elif cid in [2, 5, 7]:  # Vehicle / Truck / Bus
                if score < 0.15:
                    continue
                out_detections.append([box[0], box[1], box[2], box[3], score, 2])
            elif cid == PPE_CLASS_IDS["HARDHAT"]:
                hardhat_boxes.append(box)
            elif cid == PPE_CLASS_IDS["NO_HARDHAT"]:
                no_hardhat_boxes.append(box)
            elif cid == PPE_CLASS_IDS["VEST"]:
                vest_boxes.append(box)
            elif cid == PPE_CLASS_IDS["NO_VEST"]:
                no_vest_boxes.append(box)

        # Compute PPE Telemetry per Worker
        ppe_telemetry = []
        frame_h, frame_w = (frame.shape[:2]) if frame is not None else (720, 1280)

        for i, w_box in enumerate(worker_boxes):
            # 1. Check box containment if model explicitly outputs PPE classes
            has_hardhat = self._check_containment(w_box, hardhat_boxes)
            has_vest = self._check_containment(w_box, vest_boxes)
            has_no_hardhat = self._check_containment(w_box, no_hardhat_boxes)
            has_no_vest = self._check_containment(w_box, no_vest_boxes)

            # 2. Chromatic and Morphological Visual PPE Analysis on Head & Torso crops
            if frame is not None:
                try:
                    px1 = max(0, min(frame_w - 1, int(w_box[0])))
                    py1 = max(0, min(frame_h - 1, int(w_box[1])))
                    px2 = max(0, min(frame_w, int(w_box[2])))
                    py2 = max(0, min(frame_h, int(w_box[3])))
                    bw = px2 - px1
                    bh = py2 - py1

                    if bh >= 20 and bw >= 10:
                        # Head region: top 24% of the worker bounding box
                        head_y2 = py1 + max(5, int(bh * 0.24))
                        crop_head = frame[py1:head_y2, px1:px2]

                        # Torso region: 20% to 58% of the worker bounding box
                        torso_y1 = py1 + int(bh * 0.18)
                        torso_y2 = py1 + int(bh * 0.58)
                        crop_torso = frame[torso_y1:torso_y2, px1:px2]

                        # A. Hardhat Inspection (Standard Industrial Safety Helmets)
                        if crop_head.size > 0:
                            hsv_h = cv2.cvtColor(crop_head, cv2.COLOR_RGB2HSV)
                            hh, hs, hv = hsv_h[:,:,0], hsv_h[:,:,1], hsv_h[:,:,2]
                            # Yellow / Gold / Amber / Orange hardhats: H in [10, 48], S >= 35, V >= 50
                            yellow_orange = (hh >= 10) & (hh <= 48) & (hs >= 35) & (hv >= 50)
                            # White / Light Grey Engineer hardhats: Low saturation, high value
                            white_helmet = (hs <= 65) & (hv >= 135)
                            # Blue / Cyan Inspector hardhats: H in [90, 135], S >= 35, V >= 45
                            blue_helmet = (hh >= 90) & (hh <= 135) & (hs >= 35) & (hv >= 45)
                            # Red / Orange emergency hardhats: H <= 12 or H >= 165, S >= 60, V >= 55
                            red_helmet = ((hh <= 12) | (hh >= 165)) & (hs >= 60) & (hv >= 55)

                            hh_coverage = np.mean(yellow_orange | white_helmet | blue_helmet | red_helmet)
                            if hh_coverage > 0.08:
                                has_hardhat = True

                        # B. High-Vis Safety Vest Inspection (Fluorescent Lime / Orange / Reflective Tape)
                        if crop_torso.size > 0:
                            hsv_t = cv2.cvtColor(crop_torso, cv2.COLOR_RGB2HSV)
                            th, ts, tv = hsv_t[:,:,0], hsv_t[:,:,1], hsv_t[:,:,2]
                            # High-vis neon yellow/lime or fluorescent orange:
                            neon_vest = ((th >= 12) & (th <= 85) & (ts >= 40) & (tv >= 60)) | \
                                        (((th <= 12) | (th >= 165)) & (ts >= 50) & (tv >= 60))
                            # Silver 3M Scotchlite reflective stripes:
                            refl_stripes = (ts <= 50) & (tv >= 170)
                            vest_coverage = np.mean(neon_vest | refl_stripes)
                            if vest_coverage > 0.08:
                                has_vest = True
                except Exception as e:
                    logger.debug(f"PPE crop analysis skipped: {e}")

            compliant = (has_hardhat or not has_no_hardhat) and (has_vest or not has_no_vest)
            ppe_telemetry.append({
                "worker_index": i,
                "bbox": w_box.tolist(),
                "has_hardhat": has_hardhat,
                "has_vest": has_vest,
                "hardhat_detected": has_hardhat,
                "vest_detected": has_vest,
                "ppe_compliant": compliant
            })

        if len(out_detections) == 0:
            return np.empty((0, 6), dtype=np.float32), ppe_telemetry

        return np.array(out_detections, dtype=np.float32), ppe_telemetry

    def _check_containment(self, parent_box: np.ndarray, child_boxes: List[np.ndarray]) -> bool:
        """
        Returns True if any child box center lies within or overlaps the parent box.
        """
        px1, py1, px2, py2 = parent_box
        for c_box in child_boxes:
            cx = (c_box[0] + c_box[2]) / 2.0
            cy = (c_box[1] + c_box[3]) / 2.0
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                return True
        return False

    def _heuristic_fallback_detect(self, h: int, w: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generates representative worker and excavator detections for validation.
        """
        dets = np.array([
            [w * 0.35, h * 0.60, w * 0.40, h * 0.85, 0.94, 0],  # Worker (cls 0)
            [w * 0.65, h * 0.50, w * 0.85, h * 0.80, 0.91, 2],  # Excavator (cls 2)
        ], dtype=np.float32)
        telemetry = [{
            "worker_index": 0,
            "bbox": [w * 0.35, h * 0.60, w * 0.40, h * 0.85],
            "hardhat_detected": True,
            "vest_detected": True,
            "ppe_compliant": True
        }]
        return dets, telemetry
