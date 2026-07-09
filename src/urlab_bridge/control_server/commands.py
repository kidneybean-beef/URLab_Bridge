from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from urlab_bridge.web_control import (
    KeyState,
    WebTwistConfig,
    _config_from_twist_control_reply,
    _twist_control_state_payload,
    twist_from_keys,
)

logger = logging.getLogger(__name__)

_KEY_FIELDS = ("w", "s", "q", "e", "a", "d", "space", "shift")
_ZERO_TWIST = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RobotCommandSnapshot:
    articulation: str
    source: str
    twist: tuple[float, float, float]
    active: bool
    brake: bool
    stale: bool
    stale_timeout_s: float
    last_command_age_s: float | None
    dash_active: bool
    keys: KeyState

    def to_dict(self) -> dict[str, Any]:
        return {
            "articulation": self.articulation,
            "source": self.source,
            "twist": list(self.twist),
            "active": self.active,
            "brake": self.brake,
            "stale": self.stale,
            "stale_timeout_s": self.stale_timeout_s,
            "last_command_age_s": self.last_command_age_s,
            "dash_active": self.dash_active,
            "keys": {name: bool(getattr(self.keys, name)) for name in _KEY_FIELDS},
        }


class _RobotCommandState:
    def __init__(
        self,
        *,
        articulation: str,
        source: str,
        config: WebTwistConfig,
        stale_timeout_s: float,
    ) -> None:
        self.articulation = articulation
        self.source = source
        self.config = config
        self.stale_timeout_s = max(0.0, float(stale_timeout_s))
        self.last_command_at: float | None = None
        self.last_twist: tuple[float, float, float] = _ZERO_TWIST
        self.last_keys = KeyState()
        self.last_ui_sync_state: tuple[tuple[str, float | bool], ...] | None = None
        self.ui_sync_disabled = False
        self.active = False
        self.stale = False


class CommandHub:
    def __init__(
        self,
        articulations: list[str] | tuple[str, ...],
        *,
        config: WebTwistConfig | None = None,
        stale_timeout_s: float = 0.5,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or WebTwistConfig()
        self.stale_timeout_s = max(0.0, float(stale_timeout_s))
        self._now = now_fn
        self._lock = threading.Lock()
        self._states: dict[str, _RobotCommandState] = {}
        for articulation in articulations:
            self._ensure_state(str(articulation), source="web")

    def port(self, articulation: str, *, source: str = "web") -> "RobotCommandPort":
        with self._lock:
            self._ensure_state(str(articulation), source=source)
        return RobotCommandPort(self, str(articulation), source=source)

    def apply_control(
        self,
        articulation: str,
        payload: Mapping[str, Any],
        *,
        source: str = "web",
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("control payload must be a JSON object")
        raw_keys = payload.get("keys", payload)
        keys = KeyState.from_payload(raw_keys)
        with self._lock:
            state = self._ensure_state(str(articulation), source=source)
            state.source = str(source)
            state.last_keys = keys
            state.last_twist = twist_from_keys(keys, state.config)
            state.last_command_at = self._now()
            state.active = state.last_twist != _ZERO_TWIST
            state.stale = False
            return self._snapshot_locked(state).to_dict()

    def release(self, articulation: str) -> dict[str, Any]:
        with self._lock:
            state = self._ensure_state(str(articulation), source="web")
            self._clear_state_locked(state, stale=False, clear_command_time=True)
            return self._snapshot_locked(state).to_dict()

    def stop_if_stale(self, articulation: str) -> bool:
        with self._lock:
            state = self._ensure_state(str(articulation), source="web")
            if (
                not state.active
                and not _any_key_pressed(state.last_keys)
            ) or state.last_command_at is None:
                return False
            if (self._now() - state.last_command_at) <= state.stale_timeout_s:
                return False
            self._clear_state_locked(state, stale=True, clear_command_time=False)
            return True

    def poll(self, articulation: str) -> tuple[float, float, float]:
        self.stop_if_stale(articulation)
        with self._lock:
            state = self._ensure_state(str(articulation), source="web")
            return tuple(state.last_twist)

    def status(self, articulation: str) -> dict[str, Any]:
        return self.snapshot(articulation).to_dict()

    def snapshot(self, articulation: str) -> RobotCommandSnapshot:
        with self._lock:
            state = self._ensure_state(str(articulation), source="web")
            return self._snapshot_locked(state)

    def snapshots(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                articulation: self._snapshot_locked(state).to_dict()
                for articulation, state in self._states.items()
            }

    def sync_runtime_ui(
        self,
        articulation: str,
        runtime: object,
        runtime_articulation: str,
    ) -> bool:
        with self._lock:
            state = self._ensure_state(str(articulation), source="web")
            if state.ui_sync_disabled:
                return False
            payload = _twist_control_state_payload(state.config, state.last_keys)
            sync_state = tuple(sorted(payload.items()))
            if not _any_key_pressed(state.last_keys) and sync_state == state.last_ui_sync_state:
                return False

        sync = getattr(runtime, "set_twist_control_state", None)
        if not callable(sync):
            with self._lock:
                state = self._ensure_state(str(articulation), source="web")
                state.ui_sync_disabled = True
            logger.warning("runtime has no set_twist_control_state; web dash UI sync disabled")
            return False

        try:
            reply = sync(str(runtime_articulation), **payload)
        except Exception as exc:  # pragma: no cover - live old-plugin fallback
            with self._lock:
                state = self._ensure_state(str(articulation), source="web")
                state.ui_sync_disabled = True
            logger.warning("web dash UI sync disabled after RPC failure: %s", exc)
            return False

        with self._lock:
            state = self._ensure_state(str(articulation), source="web")
            state.config = _config_from_twist_control_reply(state.config, reply)
            state.last_twist = twist_from_keys(state.last_keys, state.config)
            state.active = state.last_twist != _ZERO_TWIST
            state.last_ui_sync_state = sync_state
        return True

    def _ensure_state(self, articulation: str, *, source: str) -> _RobotCommandState:
        state = self._states.get(articulation)
        if state is None:
            state = _RobotCommandState(
                articulation=articulation,
                source=str(source),
                config=self.config,
                stale_timeout_s=self.stale_timeout_s,
            )
            self._states[articulation] = state
        return state

    def _snapshot_locked(self, state: _RobotCommandState) -> RobotCommandSnapshot:
        age = None
        if state.last_command_at is not None:
            age = max(0.0, self._now() - state.last_command_at)
        stale = bool(state.stale) or bool(
            (state.active or _any_key_pressed(state.last_keys))
            and age is not None
            and age > state.stale_timeout_s
        )
        return RobotCommandSnapshot(
            articulation=state.articulation,
            source=state.source,
            twist=tuple(float(value) for value in state.last_twist),
            active=bool(state.active),
            brake=bool(state.last_keys.space),
            stale=stale,
            stale_timeout_s=state.stale_timeout_s,
            last_command_age_s=age,
            dash_active=bool(state.last_keys.shift),
            keys=state.last_keys,
        )

    def _clear_state_locked(
        self,
        state: _RobotCommandState,
        *,
        stale: bool,
        clear_command_time: bool,
    ) -> None:
        if clear_command_time:
            state.last_command_at = None
        state.last_keys = KeyState()
        state.last_twist = _ZERO_TWIST
        state.active = False
        state.stale = bool(stale)


class RobotCommandPort:
    def __init__(self, hub: CommandHub, articulation: str, *, source: str = "web") -> None:
        self.hub = hub
        self.articulation = str(articulation)
        self.source = str(source)
        self._quit_requested = False

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def apply_control(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.hub.apply_control(
            self.articulation,
            payload,
            source=self.source,
        )

    def release(self) -> dict[str, Any]:
        return self.hub.release(self.articulation)

    def stop_if_stale(self) -> bool:
        return self.hub.stop_if_stale(self.articulation)

    def poll(self) -> tuple[float, float, float]:
        return self.hub.poll(self.articulation)

    def status(self) -> dict[str, Any]:
        return self.hub.status(self.articulation)

    def sync_runtime_ui(self, runtime: object, articulation: str) -> bool:
        return self.hub.sync_runtime_ui(self.articulation, runtime, articulation)

    def close(self) -> None:
        self.release()


def _any_key_pressed(keys: KeyState) -> bool:
    return any(bool(getattr(keys, name)) for name in _KEY_FIELDS)
