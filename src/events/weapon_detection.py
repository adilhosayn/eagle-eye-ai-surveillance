from __future__ import annotations

from datetime import datetime
from typing import Any

from src.inference.tracker import Track
from src.inference.weapon_detector import WeaponDetection


class WeaponEventDetector:
    """Evaluates weapon detections and formats centralized events for Eagle Eye AI."""

    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = 10.0,
        severity: str = "critical",
    ) -> None:
        self.enabled = enabled
        self.cooldown_seconds = float(cooldown_seconds)
        self.severity = severity
        self._last_event_ts: dict[str, datetime] = {}
        self._counter = 0

    def evaluate(
        self,
        weapon_detections: list[WeaponDetection],
        tracks: list[Track],
        frame_index: int,
        timestamp: datetime,
        fps: float,
        camera_id: str,
    ) -> list[dict[str, Any]]:
        """Evaluate weapon detections, optionally linking to a visible person track."""
        if not self.enabled or not weapon_detections:
            return []

        events: list[dict[str, Any]] = []

        for det in weapon_detections:
            # Only trigger formal alert events for confirmed detections
            if not det.confirmed:
                continue

            # Check cooldown per weapon class
            cooldown_key = f"{camera_id}_{det.label}"
            last_event = self._last_event_ts.get(cooldown_key)
            if last_event and (timestamp - last_event).total_seconds() < self.cooldown_seconds:
                continue

            # Associate weapon with person track if bounding box is contained/intersected
            matched_track_id: int | None = self._match_person_track(det.bbox, tracks)

            self._counter += 1
            self._last_event_ts[cooldown_key] = timestamp

            events.append(
                {
                    "event_id": f"evt_{self._counter:06d}",
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "camera_id": camera_id,
                    "event_type": "weapon_detected",
                    "track_id": matched_track_id,
                    "class": det.label,
                    "confidence": round(det.score, 4),
                    "zone": "camera_view",
                    "frame_index": frame_index,
                    "severity": self.severity,
                    "fps": round(fps, 2),
                }
            )

        return events

    def _match_person_track(
        self, weapon_bbox: tuple[float, float, float, float], tracks: list[Track]
    ) -> int | None:
        """Find person track that spatially encloses or best overlaps the weapon."""
        wx1, wy1, wx2, wy2 = weapon_bbox
        w_center_x = (wx1 + wx2) / 2.0
        w_center_y = (wy1 + wy2) / 2.0

        for track in tracks:
            if track.label != "person":
                continue
            px1, py1, px2, py2 = track.bbox
            # Check if weapon center is inside or near the person's bounding box
            if px1 <= w_center_x <= px2 and py1 <= w_center_y <= py2:
                return track.track_id

        return None
