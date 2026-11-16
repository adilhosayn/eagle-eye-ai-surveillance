from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np

from src.events.zones import LineZone, PolygonZone, Zone
from src.inference.tracker import Track

SEVERITY_COLORS = {
    "weapon_detected": (0, 0, 255),  # High-visibility red for critical weapons
    "after_hours_occupancy": (0, 64, 255),
    "abandoned_object": (60, 60, 255),
    "intrusion": (0, 96, 255),
    "wrong_way": (0, 140, 255),
    "vehicle_in_pedestrian_zone": (0, 180, 255),
    "line_crossing": (0, 215, 255),
    "loitering": (0, 255, 255),
}


def draw_frame(
    frame,
    zones: list[Zone],
    tracks: list[Track],
    events: list[dict],
    fps: float,
    track_history: dict[int, deque[tuple[int, int]]] | None = None,
    dwell_timers: dict[int, float] | None = None,
    zone_heatmap_overlay=None,
    zone_heatmap_opacity: float = 0.35,
    weapon_detections: list[Any] | None = None,
    system_status: str | None = None,
):
    """Render Eagle Eye AI visualization with zones, tracks, weapons, and status banners."""
    h, w = frame.shape[:2]

    # 1. Zone heatmap overlay if present
    if zone_heatmap_overlay is not None:
        heatmap_mask = zone_heatmap_overlay.any(axis=2)
        blended = cv2.addWeighted(
            zone_heatmap_overlay,
            zone_heatmap_opacity,
            frame,
            1.0 - zone_heatmap_opacity,
            0.0,
        )
        frame[heatmap_mask] = blended[heatmap_mask]

    # 2. Zone polygons and lines
    overlay = frame.copy()
    for zone in zones:
        if isinstance(zone, PolygonZone):
            points = np.array(
                [(int(x), int(y)) for x, y in zone.points], dtype=np.int32
            )
            cv2.fillPoly(overlay, [points], (0, 120, 255))
            cv2.polylines(
                frame, [points], isClosed=True, color=(0, 165, 255), thickness=2
            )
            cv2.putText(
                frame,
                zone.name,
                tuple(points[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
            )
        elif isinstance(zone, LineZone):
            start = (int(zone.start[0]), int(zone.start[1]))
            end = (int(zone.end[0]), int(zone.end[1]))
            cv2.line(frame, start, end, (255, 215, 0), 2)
            cv2.putText(
                frame, zone.name, start, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 215, 0), 2
            )
    frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0.0)

    # 3. Object tracks (people, vehicles, bags)
    history_map = track_history or {}
    dwell_map = dwell_timers or {}
    for track in tracks:
        color = _track_color(track.label)
        history = history_map.get(track.track_id)
        if history and len(history) > 1:
            _draw_trail(frame, list(history), color)
        if track.predicted_path:
            _draw_prediction(frame, track.center, list(track.predicted_path), color)

        x1, y1, x2, y2 = map(int, track.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track.label} #{track.track_id} {track.score:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        if track.track_id in dwell_map:
            dwell_text = f"dwell {dwell_map[track.track_id]:.1f}s"
            cv2.putText(
                frame,
                dwell_text,
                (x1, min(frame.shape[0] - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # 4. Weapon Detections (High visibility red bounding boxes + labels)
    active_weapons = weapon_detections or []
    has_weapon = False
    for det in active_weapons:
        has_weapon = True
        wx1, wy1, wx2, wy2 = map(int, det.bbox)
        # Bounding box in vibrant red with double corner styling
        box_color = (0, 0, 255)
        cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), box_color, 3)

        status_tag = " [CONFIRMED]" if getattr(det, "confirmed", False) else ""
        label_text = f"{det.label.upper()} {det.score:.2f}{status_tag}"

        # Solid background tag for projector readability
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tag_y1 = max(38, wy1 - 6)
        cv2.rectangle(
            frame,
            (wx1, tag_y1 - th - 6),
            (wx1 + tw + 10, tag_y1 + 4),
            (0, 0, 220),
            -1,
        )
        cv2.putText(
            frame,
            label_text,
            (wx1 + 5, tag_y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # 5. Top Eagle Eye AI Header & Status Bar
    _draw_eagle_eye_header(
        frame=frame,
        fps=fps,
        events=events,
        has_weapon=has_weapon,
        status_override=system_status,
    )

    # 6. Active incident alert stack
    if events:
        _draw_alert_stack(frame, events)

    return frame


def _draw_eagle_eye_header(
    frame: np.ndarray,
    fps: float,
    events: list[dict],
    has_weapon: bool,
    status_override: str | None = None,
) -> None:
    """Draw professional top telemetry and security status bar."""
    w = frame.shape[1]
    header_h = 36

    # Semi-transparent dark header banner
    banner = frame[0:header_h, 0:w].copy()
    cv2.rectangle(banner, (0, 0), (w, header_h), (18, 20, 26), -1)
    frame[0:header_h, 0:w] = cv2.addWeighted(banner, 0.88, frame[0:header_h, 0:w], 0.12, 0)

    # Branding
    cv2.putText(
        frame,
        "EAGLE EYE AI",
        (12, 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (0, 230, 255),
        2,
    )

    # FPS counter
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (190, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )

    # Determine status text and color
    if status_override:
        status_text = status_override
        status_color = (0, 0, 255) if "CRITICAL" in status_override else (0, 200, 255)
    elif has_weapon or any(e.get("event_type") == "weapon_detected" for e in events):
        status_text = "CRITICAL — POSSIBLE WEAPON DETECTED"
        status_color = (0, 0, 255)
    elif any(e.get("event_type") == "intrusion" for e in events):
        status_text = "SECURITY WARNING — RESTRICTED AREA INTRUSION"
        status_color = (0, 140, 255)
    elif events:
        first_type = events[0].get("event_type", "SECURITY ALERT").replace("_", " ").upper()
        status_text = f"SECURITY WARNING — {first_type}"
        status_color = (0, 165, 255)
    else:
        status_text = "MONITORING — NO ACTIVE ALERT"
        status_color = (80, 220, 100)

    # Right-aligned or center status badge
    (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    badge_x = max(290, w - tw - 24)
    cv2.circle(frame, (badge_x - 10, 20), 5, status_color, -1)
    cv2.putText(
        frame,
        status_text,
        (badge_x, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
    )


def _draw_trail(
    frame, history: list[tuple[int, int]], color: tuple[int, int, int]
) -> None:
    for idx in range(1, len(history)):
        alpha = idx / len(history)
        thickness = max(1, int(1 + 3 * alpha))
        cv2.line(frame, history[idx - 1], history[idx], color, thickness)


def _draw_alert_stack(frame, events: list[dict]) -> None:
    visible_events = events[-3:]
    banner_height = 38 + (len(visible_events)) * 26
    start_y = 38
    # Dark panel for active alerts
    overlay = frame[start_y:start_y + banner_height, 0:380].copy()
    cv2.rectangle(overlay, (0, 0), (380, banner_height), (15, 17, 24), -1)
    frame[start_y:start_y + banner_height, 0:380] = cv2.addWeighted(
        overlay, 0.85, frame[start_y:start_y + banner_height, 0:380], 0.15, 0
    )

    for idx, event in enumerate(reversed(visible_events)):
        ev_type = str(event.get("event_type", "alert"))
        color = SEVERITY_COLORS.get(ev_type, (0, 0, 180))
        y = start_y + 22 + idx * 26
        text = f"{ev_type.upper()}  {event.get('zone', 'camera')}"
        cv2.circle(frame, (16, y - 5), 6, color, -1)
        cv2.putText(
            frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )


def _draw_prediction(
    frame,
    start: tuple[float, float],
    predicted_path: list[tuple[float, float]],
    color: tuple[int, int, int],
) -> None:
    previous = (int(start[0]), int(start[1]))
    prediction_color = tuple(min(255, channel + 25) for channel in color)
    for idx, point in enumerate(predicted_path):
        current = (int(point[0]), int(point[1]))
        if idx % 2 == 0:
            cv2.line(frame, previous, current, prediction_color, 1)
        cv2.circle(frame, current, 2, prediction_color, -1)
        previous = current


def _track_color(label: str) -> tuple[int, int, int]:
    if label == "person":
        return (80, 230, 120)
    if label in {"car", "truck", "bus", "motorcycle", "bicycle"}:
        return (255, 120, 60)
    return (255, 200, 80)
