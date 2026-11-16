from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

WEAPON_MODEL_URL = (
    "https://raw.githubusercontent.com/wSubham/AI-Weapon-Detection-and-Multi-Camera-Surveillance"
    "/main/models/rik_best_best.pt"
)


@dataclass(slots=True)
class WeaponDetection:
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int
    label: str
    confirmed: bool
    consecutive_hits: int
    warning_text: str = "WARNING — POSSIBLE WEAPON DETECTED"


class WeaponDetector:
    """Real-time weapon detector for pistols and knives with temporal confirmation.

    Features:
    - Runs a dedicated custom YOLO weapon detection model (rik_best_best.pt).
    - Inspects actual model class labels dynamically ({0: 'pistol', 1: 'knife'}).
    - Auto-downloads weights if not present locally, verifying artifact integrity.
    - Runs at configurable frame intervals (inference_interval) to minimize CPU load.
    - Employs spatial IoU matching and temporal confirmation to reject spurious single-frame false positives.
    - Generates standardized warning metadata for rendering and incident alerting.
    """

    def __init__(
        self,
        model_path: str = "models/rik_best_best.pt",
        enabled: bool = True,
        confidence_threshold: float = 0.60,
        device: str = "cpu",
        inference_interval: int = 3,
        confirmation_frames: int = 2,
        alert_cooldown_seconds: float = 10.0,
        input_size: int = 640,
    ) -> None:
        self.enabled = enabled
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.device = self._resolve_device(device)
        self.inference_interval = max(1, int(inference_interval))
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.alert_cooldown_seconds = float(alert_cooldown_seconds)
        self.input_size = int(input_size)

        self.model: Any = None
        self.names: dict[int, str] = {}
        self._tracked_candidates: list[dict[str, Any]] = []
        self._last_alert_time: dict[str, float] = {}
        self._last_detections: list[WeaponDetection] = []

        if self.enabled:
            self._initialize_model()

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve device safely, falling back to CPU if CUDA is unavailable."""
        req = str(requested_device).lower().strip()
        if "cuda" in req:
            try:
                import torch

                if torch.cuda.is_available():
                    return req
                LOGGER.warning("CUDA requested for weapon detector, but not available. Falling back to CPU.")
            except ImportError:
                pass
            return "cpu"
        return "cpu"

    def _ensure_weights_available(self) -> None:
        """Verify model weights exist or download them from the reference repository."""
        if self.model_path.exists() and self.model_path.stat().st_size > 1_000_000:
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Weapon model not found at %s. Downloading rik_best_best.pt from reference repository...",
            self.model_path,
        )
        try:
            urllib.request.urlretrieve(WEAPON_MODEL_URL, str(self.model_path))
            size = self.model_path.stat().st_size
            if size < 1_000_000:
                raise ValueError(f"Downloaded weapon model is too small ({size} bytes). Possibly corrupted.")
            LOGGER.info("Weapon model downloaded successfully (%d bytes).", size)
        except Exception as exc:
            LOGGER.error("Failed to download weapon model: %s", exc)
            raise RuntimeError(f"Could not retrieve weapon model artifact at {self.model_path}") from exc

    def _initialize_model(self) -> None:
        """Load YOLO model and inspect its class labels."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for weapon detection.") from exc

        self._ensure_weights_available()
        LOGGER.info("Loading weapon detection model from %s on %s...", self.model_path, self.device)
        self.model = YOLO(str(self.model_path))
        self.names = dict(self.model.names) if hasattr(self.model, "names") else {}
        LOGGER.info("Weapon detector initialized. Model class labels: %s", self.names)

    def detect(self, frame: np.ndarray, frame_index: int, current_timestamp: float) -> list[WeaponDetection]:
        """Detect weapons in the frame with frame-skip scheduling and temporal confirmation."""
        if not self.enabled or self.model is None:
            return []

        # If skipping inference on this frame, return the latest confirmed detections
        # while decaying candidate ages
        if frame_index % self.inference_interval != 0:
            return self._last_detections

        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                imgsz=self.input_size,
                verbose=False,
            )
        except Exception as exc:
            LOGGER.exception("Weapon detector inference error: %s", exc)
            return []

        raw_candidates: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                label = str(self.names.get(class_id, f"weapon_{class_id}"))
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_candidates.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "score": conf,
                        "class_id": class_id,
                        "label": label,
                    }
                )

        # Apply temporal confirmation across recent frames
        confirmed_detections = self._update_temporal_confirmation(
            raw_candidates, current_timestamp=current_timestamp
        )
        self._last_detections = confirmed_detections
        return confirmed_detections

    def _update_temporal_confirmation(
        self, raw_candidates: list[dict[str, Any]], current_timestamp: float
    ) -> list[WeaponDetection]:
        """Match detections across frames using IoU to build confidence and reduce false alarms."""
        updated_candidates: list[dict[str, Any]] = []
        output_detections: list[WeaponDetection] = []

        # Mark all previous candidates as unmatched initially
        unmatched_prev = list(self._tracked_candidates)

        for candidate in raw_candidates:
            c_box = candidate["bbox"]
            c_label = candidate["label"]
            best_match_idx = -1
            best_iou = 0.0

            for idx, prev in enumerate(unmatched_prev):
                if prev["label"] == c_label:
                    iou = _compute_iou(c_box, prev["bbox"])
                    if iou > 0.3 and iou > best_iou:
                        best_iou = iou
                        best_match_idx = idx

            if best_match_idx >= 0:
                matched_prev = unmatched_prev.pop(best_match_idx)
                hits = matched_prev["consecutive_hits"] + 1
            else:
                hits = 1

            confirmed = hits >= self.confirmation_frames

            updated_record = {
                "bbox": c_box,
                "score": candidate["score"],
                "class_id": candidate["class_id"],
                "label": c_label,
                "consecutive_hits": hits,
                "last_seen": current_timestamp,
            }
            updated_candidates.append(updated_record)

            output_detections.append(
                WeaponDetection(
                    bbox=c_box,
                    score=candidate["score"],
                    class_id=candidate["class_id"],
                    label=c_label,
                    confirmed=confirmed,
                    consecutive_hits=hits,
                )
            )

        # Retain previous candidates that might have had a single frame dropout
        for prev in unmatched_prev:
            if current_timestamp - prev["last_seen"] < 0.6 and prev["consecutive_hits"] > 1:
                # Age candidate slightly but preserve state across brief occlusion
                prev["consecutive_hits"] = max(1, prev["consecutive_hits"] - 1)
                updated_candidates.append(prev)

        self._tracked_candidates = updated_candidates
        return output_detections


def _compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Calculate Intersection-over-Union between two bounding boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area
