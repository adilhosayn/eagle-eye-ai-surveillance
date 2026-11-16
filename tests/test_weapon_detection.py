from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.events.weapon_detection import WeaponEventDetector
from src.inference.tracker import Track
from src.inference.weapon_detector import WeaponDetection, WeaponDetector, _compute_iou
from src.utils.draw import draw_frame
from src.utils.security_audio import SecurityAudioPlayer


def test_compute_iou() -> None:
    # Identical boxes
    box1 = (0.0, 0.0, 10.0, 10.0)
    assert _compute_iou(box1, box1) == pytest.approx(1.0)

    # Completely disjoint
    box2 = (20.0, 20.0, 30.0, 30.0)
    assert _compute_iou(box1, box2) == pytest.approx(0.0)

    # 50% overlap
    box3 = (0.0, 0.0, 10.0, 20.0)
    box4 = (0.0, 10.0, 10.0, 30.0)
    # intersection: 0..10, 10..20 -> 10*10 = 100. union: 10*20 + 10*20 - 100 = 300. iou = 1/3
    assert _compute_iou(box3, box4) == pytest.approx(1.0 / 3.0)


def test_weapon_detector_temporal_confirmation() -> None:
    detector = WeaponDetector(
        model_path="models/rik_best_best.pt",
        enabled=False,  # Don't load full neural net for unit testing temporal math
        confirmation_frames=2,
    )
    assert detector.confirmation_frames == 2

    # Simulated candidate prediction from frame 1
    c1 = [{"bbox": (10.0, 10.0, 50.0, 50.0), "score": 0.85, "class_id": 0, "label": "pistol"}]
    t1 = 100.0
    out1 = detector._update_temporal_confirmation(c1, current_timestamp=t1)
    assert len(out1) == 1
    assert out1[0].confirmed is False
    assert out1[0].consecutive_hits == 1

    # Same candidate in frame 2 (spatial overlap > 0.3)
    c2 = [{"bbox": (11.0, 10.0, 51.0, 50.0), "score": 0.88, "class_id": 0, "label": "pistol"}]
    t2 = 100.1
    out2 = detector._update_temporal_confirmation(c2, current_timestamp=t2)
    assert len(out2) == 1
    assert out2[0].confirmed is True  # Reached confirmation_frames (2)
    assert out2[0].consecutive_hits == 2


def test_weapon_event_detector_emission_and_cooldown() -> None:
    event_detector = WeaponEventDetector(
        enabled=True,
        cooldown_seconds=5.0,
        severity="critical",
    )

    det_unconfirmed = WeaponDetection(
        bbox=(20.0, 20.0, 40.0, 40.0),
        score=0.75,
        class_id=0,
        label="knife",
        confirmed=False,
        consecutive_hits=1,
    )
    now = datetime.now(timezone.utc)

    # 1. Unconfirmed detection should NOT emit formal security alert
    events = event_detector.evaluate(
        weapon_detections=[det_unconfirmed],
        tracks=[],
        frame_index=1,
        timestamp=now,
        fps=30.0,
        camera_id="cam_main",
    )
    assert len(events) == 0

    # 2. Confirmed detection should emit event with correct schema
    det_confirmed = WeaponDetection(
        bbox=(20.0, 20.0, 40.0, 40.0),
        score=0.88,
        class_id=0,
        label="knife",
        confirmed=True,
        consecutive_hits=2,
    )
    events2 = event_detector.evaluate(
        weapon_detections=[det_confirmed],
        tracks=[],
        frame_index=2,
        timestamp=now,
        fps=30.0,
        camera_id="cam_main",
    )
    assert len(events2) == 1
    evt = events2[0]
    assert evt["event_type"] == "weapon_detected"
    assert evt["class"] == "knife"
    assert evt["confidence"] == 0.88
    assert evt["severity"] == "critical"
    assert evt["track_id"] is None
    assert evt["zone"] == "camera_view"

    # 3. Repeated confirmed detection within cooldown (5.0s) should NOT emit duplicate
    events3 = event_detector.evaluate(
        weapon_detections=[det_confirmed],
        tracks=[],
        frame_index=3,
        timestamp=now,
        fps=30.0,
        camera_id="cam_main",
    )
    assert len(events3) == 0


def test_weapon_event_detector_person_track_matching() -> None:
    event_detector = WeaponEventDetector(enabled=True, cooldown_seconds=0.0)

    # Person track from (10, 10) to (100, 200)
    person_track = Track(
        track_id=42,
        bbox=(10.0, 10.0, 100.0, 200.0),
        label="person",
        score=0.92,
        last_seen_frame=1,
    )

    # Weapon inside person bbox
    weapon_in_hand = WeaponDetection(
        bbox=(40.0, 80.0, 70.0, 110.0),
        score=0.90,
        class_id=0,
        label="pistol",
        confirmed=True,
        consecutive_hits=3,
    )

    events = event_detector.evaluate(
        weapon_detections=[weapon_in_hand],
        tracks=[person_track],
        frame_index=10,
        timestamp=datetime.now(timezone.utc),
        fps=30.0,
        camera_id="gate_1",
    )
    assert len(events) == 1
    assert events[0]["track_id"] == 42


def test_security_audio_priorities() -> None:
    player = SecurityAudioPlayer(
        enabled=True,
        cooldown_seconds=0.05,
        weapon_priority=True,
        max_repeat_hz=50.0,
    )
    try:
        # Enqueue beep
        ok1 = player.trigger_beep("test_intrusion")
        assert ok1 is True

        # Enqueue siren (weapon priority should succeed)
        ok2 = player.trigger_siren("weapon_detected")
        assert ok2 is True

        # Silence clears active queue
        player.silence()
        assert player._queue.empty()
    finally:
        player.close()


def test_draw_frame_with_eagle_eye_visuals() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    weapon = WeaponDetection(
        bbox=(50.0, 60.0, 120.0, 140.0),
        score=0.89,
        class_id=0,
        label="pistol",
        confirmed=True,
        consecutive_hits=2,
    )
    annotated = draw_frame(
        frame=frame,
        zones=[],
        tracks=[],
        events=[{"event_type": "weapon_detected", "zone": "camera"}],
        fps=25.0,
        weapon_detections=[weapon],
    )
    assert annotated.shape == (480, 640, 3)
    # Ensure drawing modified the frame (non-zero pixels present)
    assert np.any(annotated > 0)
