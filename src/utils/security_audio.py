from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
from typing import Any

LOGGER = logging.getLogger(__name__)

# Windows sound library
try:
    import winsound

    HAS_WINSOUND = True
except ImportError:
    winsound = None  # type: ignore
    HAS_WINSOUND = False


class SecurityAudioPlayer:
    """Non-blocking, thread-safe audible security alert engine for Eagle Eye AI.

    Features:
    - Windows native audio playback using winsound.Beep without blocking frame processing.
    - Two distinct sound modes:
        1. Security Beep: Clear warning tone for intrusion, loitering, line-crossing.
        2. Emergency Siren: High-priority alternating dual-frequency tone for confirmed weapon detections.
    - Weapon alert prioritization: sirens supersede regular pending beeps.
    - Cooldown, deduplication, and max repeat frequency protection.
    - Clean shutdown and emergency silence capability.
    """

    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = 5.0,
        weapon_priority: bool = True,
        duration_ms: int = 500,
        max_repeat_hz: float = 2.0,
        event_mapping: dict[str, str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.cooldown_seconds = max(0.1, float(cooldown_seconds))
        self.weapon_priority = weapon_priority
        self.duration_ms = max(50, int(duration_ms))
        self.min_interval_seconds = 1.0 / max(0.1, float(max_repeat_hz))
        self.event_mapping = event_mapping or {
            "weapon_detected": "siren",
            "intrusion": "beep",
            "after_hours_occupancy": "beep",
            "abandoned_object": "beep",
            "wrong_way": "beep",
            "loitering": "beep",
            "line_crossing": "beep",
        }

        self._queue: queue.PriorityQueue[tuple[int, float, str]] = queue.PriorityQueue(
            maxsize=16
        )
        self._last_played_time: dict[str, float] = {}
        self._last_audio_time = 0.0
        self._running = True
        self._lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._worker_loop, name="security-audio-worker", daemon=True
        )
        self._worker.start()

    def play_for_event(self, event_type: str, severity: str = "medium") -> bool:
        """Trigger audio based on event type and severity mapping."""
        if not self.enabled:
            return False

        sound_type = self.event_mapping.get(event_type)
        if not sound_type:
            if severity == "critical":
                sound_type = "siren"
            elif severity == "high":
                sound_type = "beep"
            else:
                return False

        if sound_type == "siren":
            return self.trigger_siren(event_type=event_type)
        if sound_type in ("beep", "optional_beep"):
            return self.trigger_beep(event_type=event_type)
        return False

    def trigger_siren(self, event_type: str = "weapon_detected") -> bool:
        """Queue an emergency alternating siren (highest priority)."""
        return self._enqueue_sound(sound_type="siren", event_type=event_type, priority=1)

    def trigger_beep(self, event_type: str = "alert") -> bool:
        """Queue a standard security warning beep."""
        return self._enqueue_sound(sound_type="beep", event_type=event_type, priority=5)

    def _enqueue_sound(self, sound_type: str, event_type: str, priority: int) -> bool:
        if not self.enabled:
            return False

        now = time.time()
        with self._lock:
            # Check event-specific cooldown
            last_event_time = self._last_played_time.get(event_type, 0.0)
            if now - last_event_time < self.cooldown_seconds:
                return False

            # Check global repeat rate limiter (critical sirens bypass ordinary beep throttling)
            is_priority_siren = (sound_type == "siren" and self.weapon_priority)
            if not is_priority_siren and (now - self._last_audio_time < self.min_interval_seconds):
                return False

            self._last_played_time[event_type] = now
            self._last_audio_time = now

        try:
            # When weapon siren arrives and weapon_priority is set, clear lower-priority beeps
            if sound_type == "siren" and self.weapon_priority:
                self._purge_lower_priority()

            self._queue.put_nowait((priority, now, sound_type))
            return True
        except queue.Full:
            LOGGER.debug("Audio queue full; skipping sound %s", sound_type)
            return False

    def _purge_lower_priority(self) -> None:
        """Discard pending normal beeps when an emergency siren arrives."""
        retained: list[tuple[int, float, str]] = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item[0] <= 1:  # Keep existing sirens
                    retained.append(item)
            except queue.Empty:
                break
        for item in retained:
            self._queue.put_nowait(item)

    def silence(self) -> None:
        """Clear all queued sound alerts."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        """Stop worker thread and cleanup audio resources."""
        self._running = False
        self.silence()
        try:
            self._queue.put_nowait((0, 0.0, "exit"))
        except queue.Full:
            pass
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def _worker_loop(self) -> None:
        while self._running:
            try:
                priority, _, sound_type = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if sound_type == "exit" or not self._running:
                break

            try:
                if sound_type == "siren":
                    self._play_siren()
                elif sound_type == "beep":
                    self._play_beep()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Audio playback failed: %s", exc)

    def _play_beep(self) -> None:
        if HAS_WINSOUND and winsound is not None:
            # 1050 Hz clear warning tone
            winsound.Beep(1050, min(self.duration_ms, 350))
        else:
            LOGGER.info("[Audio Beep Triggered - Fallback non-Windows]")

    def _play_siren(self) -> None:
        if HAS_WINSOUND and winsound is not None:
            # High-low alternating siren: 1500 Hz -> 850 Hz -> 1500 Hz
            cycles = 2
            half_duration = max(80, int(self.duration_ms / (cycles * 2)))
            for _ in range(cycles):
                if not self._running:
                    break
                winsound.Beep(1550, half_duration)
                winsound.Beep(850, half_duration)
        else:
            LOGGER.info("[Audio Emergency Siren Triggered - Fallback non-Windows]")


def run_standalone_test() -> None:
    """Standalone test command for Eagle Eye security audio verification."""
    print("=" * 60)
    print("Eagle Eye AI — Security Audio Subsystem Test")
    print("=" * 60)
    print(f"Platform: {sys.platform}")
    print(f"Windows Sound Available: {HAS_WINSOUND}")

    player = SecurityAudioPlayer(enabled=True, cooldown_seconds=0.5, max_repeat_hz=5.0)

    print("\n[1/3] Testing standard Security Beep (Intrusion/Line Crossing warning)...")
    player.trigger_beep("test_intrusion")
    time.sleep(1.0)

    print("\n[2/3] Testing Emergency Siren (Weapon Detection alert)...")
    player.trigger_siren("test_weapon")
    time.sleep(1.5)

    print("\n[3/3] Testing Priority Preemption (Siren superseding Beep)...")
    player.trigger_beep("test_loiter")
    player.trigger_siren("test_critical_weapon")
    time.sleep(1.5)

    print("\nShutting down audio subsystem...")
    player.close()
    print("Audio test finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eagle Eye Security Audio Engine")
    parser.add_argument("--test", action="store_true", help="Run standalone audio verification")
    args = parser.parse_args()

    if args.test:
        run_standalone_test()
    else:
        parser.print_help()
