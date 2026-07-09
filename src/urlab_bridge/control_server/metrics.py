from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any


class ControlLoopMetrics:
    def __init__(self, *, freq_hz: float) -> None:
        self.freq_hz = float(freq_hz)
        self.deadline_s = 1.0 / self.freq_hz if self.freq_hz > 0.0 else 0.0
        self._lock = threading.Lock()
        self._tick_count = 0
        self._missed_deadlines = 0
        self._last_tick_duration_s = 0.0
        self._last_policy_duration_s = 0.0
        self._last_step_duration_s = 0.0
        self._last_deadline_overrun_s = 0.0
        self._max_tick_duration_s = 0.0
        self._active_robot_count = 0
        self._robots: dict[str, dict[str, Any]] = {}

    def record_tick(
        self,
        *,
        tick_duration_s: float,
        policy_duration_s: float,
        step_duration_s: float,
        active_robot_count: int,
        command_statuses: Mapping[str, Mapping[str, Any]],
    ) -> None:
        tick = max(0.0, float(tick_duration_s))
        policy = max(0.0, float(policy_duration_s))
        step = max(0.0, float(step_duration_s))
        overrun = max(0.0, tick - self.deadline_s) if self.deadline_s > 0.0 else 0.0
        with self._lock:
            self._tick_count += 1
            if overrun > 0.0:
                self._missed_deadlines += 1
            self._last_tick_duration_s = tick
            self._last_policy_duration_s = policy
            self._last_step_duration_s = step
            self._last_deadline_overrun_s = overrun
            self._max_tick_duration_s = max(self._max_tick_duration_s, tick)
            self._active_robot_count = int(active_robot_count)
            self._robots = {
                str(name): _json_safe_dict(status)
                for name, status in command_statuses.items()
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "freq_hz": self.freq_hz,
                "deadline_s": self.deadline_s,
                "tick_count": self._tick_count,
                "missed_deadlines": self._missed_deadlines,
                "last_tick_duration_s": self._last_tick_duration_s,
                "last_policy_duration_s": self._last_policy_duration_s,
                "last_step_duration_s": self._last_step_duration_s,
                "last_deadline_overrun_s": self._last_deadline_overrun_s,
                "max_tick_duration_s": self._max_tick_duration_s,
                "active_robot_count": self._active_robot_count,
                "robots": {
                    name: dict(status)
                    for name, status in self._robots.items()
                },
            }


def _json_safe_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(item) for key, item in value.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_safe_dict(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
