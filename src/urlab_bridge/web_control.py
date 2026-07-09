# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-free browser keyboard control for a single URLab twist robot."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

_KEY_FIELDS = ("w", "s", "q", "e", "a", "d", "space", "shift")
_ZERO_TWIST = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class WebTwistConfig:
    max_vx: float = 1.0
    max_vy: float = 0.5
    max_yaw: float = 1.57
    dash_max_vx: float = 2.0
    dash_max_vy: float = 1.0
    dash_max_yaw: float = 3.14


@dataclass(frozen=True)
class KeyState:
    w: bool = False
    s: bool = False
    q: bool = False
    e: bool = False
    a: bool = False
    d: bool = False
    space: bool = False
    shift: bool = False

    @classmethod
    def from_payload(cls, payload: "KeyState | Mapping[str, Any]") -> "KeyState":
        if isinstance(payload, KeyState):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError("key state must be a JSON object")
        values = {
            name: _is_pressed(payload.get(name, False))
            for name in _KEY_FIELDS
        }
        return cls(**values)


def twist_from_keys(
    keys: KeyState | Mapping[str, Any],
    config: WebTwistConfig | None = None,
) -> tuple[float, float, float]:
    state = KeyState.from_payload(keys)
    cfg = config or WebTwistConfig()
    if state.space:
        return _ZERO_TWIST

    vx_limit = _speed_limit(cfg.max_vx, cfg.dash_max_vx, state.shift)
    vy_limit = _speed_limit(cfg.max_vy, cfg.dash_max_vy, state.shift)
    yaw_limit = _speed_limit(cfg.max_yaw, cfg.dash_max_yaw, state.shift)

    vx_axis = int(state.w) - int(state.s)
    vy_axis = int(state.q) - int(state.e)
    yaw_axis = int(state.a) - int(state.d)
    return (
        vx_axis * vx_limit,
        vy_axis * vy_limit,
        yaw_axis * yaw_limit,
    )


class WebControlBroker:
    """Small thread-safe adapter from browser key snapshots to URLab RPC."""

    def __init__(
        self,
        client: object,
        *,
        articulation: str,
        config: WebTwistConfig | None = None,
        stale_timeout_s: float = 0.5,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.articulation = str(articulation)
        self.config = config or WebTwistConfig()
        self.stale_timeout_s = max(0.0, float(stale_timeout_s))
        self._now = now_fn
        self._lock = threading.Lock()
        self._last_command_at: float | None = None
        self._last_twist: tuple[float, float, float] = _ZERO_TWIST
        self._last_keys = KeyState()
        self._last_ui_sync_state: tuple[tuple[str, float | bool], ...] | None = None
        self._ui_sync_disabled = False
        self._active = False

    def apply_control(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("control payload must be a JSON object")
        raw_keys = payload.get("keys", payload)
        keys = KeyState.from_payload(raw_keys)
        with self._lock:
            self._sync_runtime_ui_locked(keys)
            twist = twist_from_keys(keys, self.config)
            self._send_twist_locked(twist)
            self._last_command_at = self._now()
            self._last_keys = keys
            self._last_twist = twist
            self._active = twist != _ZERO_TWIST
            return self._status_locked(ok=True)

    def release(self) -> dict[str, Any]:
        with self._lock:
            self._send_twist_locked(_ZERO_TWIST)
            self._sync_runtime_ui_locked(KeyState())
            self._last_command_at = None
            self._last_keys = KeyState()
            self._last_twist = _ZERO_TWIST
            self._active = False
            return self._status_locked(ok=True)

    def stop_if_stale(self) -> bool:
        with self._lock:
            if (
                not self._active
                and not _any_key_pressed(self._last_keys)
            ) or self._last_command_at is None:
                return False
            if (self._now() - self._last_command_at) <= self.stale_timeout_s:
                return False
            self._send_twist_locked(_ZERO_TWIST)
            self._sync_runtime_ui_locked(KeyState())
            self._last_command_at = None
            self._last_keys = KeyState()
            self._last_twist = _ZERO_TWIST
            self._active = False
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked(ok=True)

    def _send_twist_locked(self, twist: tuple[float, float, float]) -> None:
        vx, vy, yaw = twist
        runtime = getattr(self.client, "runtime")
        runtime.set_twist(
            self.articulation,
            linear=(float(vx), float(vy), 0.0),
            angular=(0.0, 0.0, float(yaw)),
        )

    def _sync_runtime_ui_locked(self, keys: KeyState) -> None:
        if self._ui_sync_disabled:
            return
        payload = _twist_control_state_payload(self.config, keys)
        sync_state = tuple(sorted(payload.items()))
        if not _any_key_pressed(keys) and sync_state == self._last_ui_sync_state:
            return

        runtime = getattr(self.client, "runtime")
        sync = getattr(runtime, "set_twist_control_state", None)
        if not callable(sync):
            self._ui_sync_disabled = True
            logger.warning("runtime has no set_twist_control_state; web dash UI sync disabled")
            return

        try:
            reply = sync(self.articulation, **payload)
        except Exception as exc:  # pragma: no cover - live old-plugin fallback
            self._ui_sync_disabled = True
            logger.warning("web dash UI sync disabled after RPC failure: %s", exc)
            return
        self.config = _config_from_twist_control_reply(self.config, reply)
        self._last_ui_sync_state = sync_state

    def _status_locked(self, *, ok: bool) -> dict[str, Any]:
        age = None
        if self._last_command_at is not None:
            age = max(0.0, self._now() - self._last_command_at)
        stale = bool(
            (self._active or _any_key_pressed(self._last_keys))
            and age is not None
            and age > self.stale_timeout_s
        )
        return {
            "ok": bool(ok),
            "articulation": self.articulation,
            "active": self._active,
            "stale": stale,
            "stale_timeout_s": self.stale_timeout_s,
            "last_command_age_s": age,
            "twist": list(self._last_twist),
        }


class WebCommandSource:
    """In-process command source for Go2 policy loops driven by the webpage."""

    def __init__(
        self,
        *,
        config: WebTwistConfig | None = None,
        stale_timeout_s: float = 0.5,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or WebTwistConfig()
        self.stale_timeout_s = max(0.0, float(stale_timeout_s))
        self._now = now_fn
        self._lock = threading.Lock()
        self._last_command_at: float | None = None
        self._last_twist: tuple[float, float, float] = _ZERO_TWIST
        self._last_keys = KeyState()
        self._last_ui_sync_state: tuple[tuple[str, float | bool], ...] | None = None
        self._ui_sync_disabled = False
        self._active = False
        self._quit_requested = False

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def poll(self) -> tuple[float, float, float]:
        self.stop_if_stale()
        with self._lock:
            return self._last_twist

    def apply_control(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("control payload must be a JSON object")
        raw_keys = payload.get("keys", payload)
        keys = KeyState.from_payload(raw_keys)
        twist = twist_from_keys(keys, self.config)
        with self._lock:
            self._last_command_at = self._now()
            self._last_keys = keys
            self._last_twist = twist
            self._active = twist != _ZERO_TWIST
            return self._status_locked(ok=True)

    def release(self) -> dict[str, Any]:
        with self._lock:
            self._last_command_at = None
            self._last_keys = KeyState()
            self._last_twist = _ZERO_TWIST
            self._active = False
            return self._status_locked(ok=True)

    def stop_if_stale(self) -> bool:
        with self._lock:
            if (
                not self._active
                and not _any_key_pressed(self._last_keys)
            ) or self._last_command_at is None:
                return False
            if (self._now() - self._last_command_at) <= self.stale_timeout_s:
                return False
            self._last_command_at = None
            self._last_keys = KeyState()
            self._last_twist = _ZERO_TWIST
            self._active = False
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked(ok=True)

    def sync_runtime_ui(self, runtime: object, articulation: str) -> bool:
        """Push web keyboard limits/dash display state to UE when it changes."""
        with self._lock:
            if self._ui_sync_disabled:
                return False
            payload = _twist_control_state_payload(self.config, self._last_keys)
            sync_state = tuple(sorted(payload.items()))
            if not _any_key_pressed(self._last_keys) and sync_state == self._last_ui_sync_state:
                return False

        sync = getattr(runtime, "set_twist_control_state", None)
        if not callable(sync):
            with self._lock:
                self._ui_sync_disabled = True
            logger.warning("runtime has no set_twist_control_state; web dash UI sync disabled")
            return False

        try:
            reply = sync(str(articulation), **payload)
        except Exception as exc:  # pragma: no cover - live old-plugin fallback
            with self._lock:
                self._ui_sync_disabled = True
            logger.warning("web dash UI sync disabled after RPC failure: %s", exc)
            return False

        with self._lock:
            self.config = _config_from_twist_control_reply(self.config, reply)
            self._last_twist = twist_from_keys(self._last_keys, self.config)
            self._active = self._last_twist != _ZERO_TWIST
            self._last_ui_sync_state = sync_state
        return True

    def close(self) -> None:
        self.release()

    def _status_locked(self, *, ok: bool) -> dict[str, Any]:
        age = None
        if self._last_command_at is not None:
            age = max(0.0, self._now() - self._last_command_at)
        stale = bool(
            (self._active or _any_key_pressed(self._last_keys))
            and age is not None
            and age > self.stale_timeout_s
        )
        return {
            "ok": bool(ok),
            "articulation": "",
            "active": self._active,
            "stale": stale,
            "stale_timeout_s": self.stale_timeout_s,
            "last_command_age_s": age,
            "twist": list(self._last_twist),
        }


def make_handler(
    broker: WebControlBroker,
    *,
    metrics_provider: Callable[[], Mapping[str, Any]] | object | None = None,
) -> type[BaseHTTPRequestHandler]:
    class WebControlHandler(BaseHTTPRequestHandler):
        server_version = "URLabWebControl/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send_html(_INDEX_HTML)
            elif self.path == "/health":
                self._send_json(200, broker.status())
            elif self.path == "/metrics" and metrics_provider is not None:
                self._send_json(200, _metrics_payload(metrics_provider))
            else:
                self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path == "/api/control":
                    payload = self._read_json()
                    self._send_json(200, broker.apply_control(payload))
                elif self.path == "/api/release":
                    self._send_json(200, broker.release())
                else:
                    self._send_json(404, {"ok": False, "error": "not_found"})
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            except Exception as exc:  # pragma: no cover - live RPC failure path
                logger.exception("web control request failed")
                self._send_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("HTTP %s - %s", self.address_string(), fmt % args)

        def _read_json(self) -> Mapping[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length <= 0:
                return {}
            if length > 64 * 1024:
                raise ValueError("request body too large")
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid JSON body") from exc
            if not isinstance(payload, Mapping):
                raise ValueError("JSON body must be an object")
            return payload

        def _send_html(self, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return WebControlHandler


def _metrics_payload(provider: Callable[[], Mapping[str, Any]] | object) -> Mapping[str, Any]:
    if callable(provider):
        payload = provider()
    else:
        snapshot = getattr(provider, "snapshot", None)
        payload = snapshot() if callable(snapshot) else provider
    if not isinstance(payload, Mapping):
        raise ValueError("metrics provider must return a JSON object")
    return payload


def run_server(
    client: object,
    *,
    articulation: str,
    bind: str = "127.0.0.1",
    port: int = 8088,
    config: WebTwistConfig | None = None,
    stale_timeout_s: float = 0.5,
) -> None:
    broker = WebControlBroker(
        client,
        articulation=articulation,
        config=config,
        stale_timeout_s=stale_timeout_s,
    )
    server = ThreadingHTTPServer((bind, int(port)), make_handler(broker))
    stop_event = threading.Event()
    interval_s = max(0.05, min(0.25, stale_timeout_s / 2.0 if stale_timeout_s else 0.05))

    def watchdog() -> None:
        while not stop_event.wait(interval_s):
            broker.stop_if_stale()

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()
    logger.info("URLab web control listening on http://%s:%d", bind, server.server_address[1])
    try:
        server.serve_forever()
    finally:
        stop_event.set()
        watchdog_thread.join(timeout=1.0)
        try:
            broker.release()
        except Exception:  # pragma: no cover - teardown best effort
            logger.exception("failed to release web control twist during shutdown")
        server.server_close()


def _is_pressed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    return False


def _speed_limit(normal: float, dash_cap: float, dash: bool) -> float:
    base = max(0.0, float(normal))
    if not dash:
        return base
    return min(base * 2.0, max(0.0, float(dash_cap)))


def _any_key_pressed(keys: KeyState) -> bool:
    return any(getattr(keys, name) for name in _KEY_FIELDS)


def _twist_control_state_payload(
    config: WebTwistConfig,
    keys: KeyState,
) -> dict[str, float | bool]:
    return {
        "dash_active": bool(keys.shift),
    }


def _config_from_twist_control_reply(
    config: WebTwistConfig,
    reply: Any,
) -> WebTwistConfig:
    if not isinstance(reply, Mapping):
        return config

    values = {
        "max_vx": config.max_vx,
        "max_vy": config.max_vy,
        "max_yaw": config.max_yaw,
        "dash_max_vx": config.dash_max_vx,
        "dash_max_vy": config.dash_max_vy,
        "dash_max_yaw": config.dash_max_yaw,
    }
    changed = False
    for field in tuple(values):
        wire_value = reply.get(field)
        if isinstance(wire_value, bool) or not isinstance(wire_value, (int, float)):
            continue
        values[field] = max(0.0, float(wire_value))
        changed = True

    if not changed:
        return config
    return WebTwistConfig(**values)


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>URLab Web Control</title>
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      background: #111418;
      color: #eef2f7;
      font-family: system-ui, sans-serif;
    }
    body {
      display: grid;
      place-items: center;
      user-select: none;
      outline: none;
      min-height: 100dvh;
    }
    main {
      width: min(92vw, 560px);
      text-align: center;
    }
    h1 {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: 0;
      margin: 0 0 28px;
    }
    .control-pad {
      display: grid;
      grid-template-columns: repeat(3, minmax(86px, 1fr));
      grid-template-areas:
        "strafe-left forward strafe-right"
        "turn-left backward turn-right"
        "brake brake brake";
      gap: 12px;
    }
    .control-button {
      appearance: none;
      border: 1px solid #303946;
      border-radius: 8px;
      background: #1a2028;
      color: #eef2f7;
      min-height: 88px;
      padding: 12px 8px;
      display: grid;
      place-items: center;
      gap: 4px;
      font: inherit;
      cursor: pointer;
      touch-action: none;
      -webkit-tap-highlight-color: transparent;
      transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
    }
    .control-button:hover {
      border-color: #536274;
    }
    .control-button:focus-visible {
      outline: 3px solid #74a7ff;
      outline-offset: 3px;
    }
    .control-button.active {
      background: #1f7a5b;
      border-color: #42d49b;
      color: #ffffff;
      transform: translateY(1px);
    }
    .control-button.brake.active {
      background: #9a3030;
      border-color: #ff7777;
    }
    .key {
      font-size: 1.35rem;
      font-weight: 800;
      line-height: 1;
    }
    .label {
      font-size: 0.88rem;
      font-weight: 650;
      line-height: 1.1;
    }
    .forward { grid-area: forward; }
    .backward { grid-area: backward; }
    .strafe-left { grid-area: strafe-left; }
    .strafe-right { grid-area: strafe-right; }
    .turn-left { grid-area: turn-left; }
    .turn-right { grid-area: turn-right; }
    .brake { grid-area: brake; }
    @media (max-width: 420px) {
      main {
        width: min(94vw, 360px);
      }
      h1 {
        font-size: 1.5rem;
        margin-bottom: 18px;
      }
      .control-pad {
        grid-template-columns: repeat(3, minmax(74px, 1fr));
        gap: 8px;
      }
      .control-button {
        min-height: 76px;
        padding: 10px 6px;
      }
      .key {
        font-size: 1.15rem;
      }
      .label {
        font-size: 0.78rem;
      }
    }
  </style>
</head>
<body tabindex="0">
  <main>
    <h1>URLab Web Control</h1>
    <section class="control-pad" aria-label="Robot movement controls">
      <button class="control-button strafe-left" type="button" data-key="q" aria-pressed="false">
        <span class="key">Q</span>
        <span class="label">Strafe L</span>
      </button>
      <button class="control-button forward" type="button" data-key="w" aria-pressed="false">
        <span class="key">W</span>
        <span class="label">Forward</span>
      </button>
      <button class="control-button strafe-right" type="button" data-key="e" aria-pressed="false">
        <span class="key">E</span>
        <span class="label">Strafe R</span>
      </button>
      <button class="control-button turn-left" type="button" data-key="a" aria-pressed="false">
        <span class="key">A</span>
        <span class="label">Turn L</span>
      </button>
      <button class="control-button backward" type="button" data-key="s" aria-pressed="false">
        <span class="key">S</span>
        <span class="label">Back</span>
      </button>
      <button class="control-button turn-right" type="button" data-key="d" aria-pressed="false">
        <span class="key">D</span>
        <span class="label">Turn R</span>
      </button>
      <button class="control-button brake" type="button" data-key="space" aria-pressed="false">
        <span class="key">Space</span>
        <span class="label">Brake</span>
      </button>
    </section>
  </main>
  <script>
    const keys = {w:false,s:false,q:false,e:false,a:false,d:false,space:false,shift:false};
    const activeSources = new Map();
    const buttons = Array.from(document.querySelectorAll("[data-key]"));
    let timer = null;

    function keyName(event) {
      if (event.code === "Space") return "space";
      if (event.code === "ShiftLeft") return "shift";
      const k = event.key.toLowerCase();
      if (Object.prototype.hasOwnProperty.call(keys, k)) return k;
      return "";
    }

    function anyPressed() {
      return Object.values(keys).some(Boolean);
    }

    function syncButtons() {
      for (const button of buttons) {
        const active = !!keys[button.dataset.key];
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
      }
    }

    function recomputeKeys() {
      const next = {w:false,s:false,q:false,e:false,a:false,d:false,space:false,shift:false};
      for (const key of activeSources.values()) {
        next[key] = true;
      }
      let changed = false;
      for (const key of Object.keys(keys)) {
        if (keys[key] !== next[key]) changed = true;
        keys[key] = next[key];
      }
      syncButtons();
      if (changed) sendState();
      if (anyPressed()) ensureTimer();
    }

    function setSource(source, key, active) {
      if (!Object.prototype.hasOwnProperty.call(keys, key)) return;
      if (active) {
        activeSources.set(source, key);
      } else {
        activeSources.delete(source);
      }
      recomputeKeys();
    }

    function sendState(keepalive=false) {
      fetch("/api/control", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({keys}),
        keepalive
      }).catch(() => {});
    }

    function ensureTimer() {
      if (timer !== null) return;
      timer = setInterval(() => {
        if (anyPressed()) {
          sendState();
        } else {
          clearInterval(timer);
          timer = null;
        }
      }, 50);
    }

    function release() {
      activeSources.clear();
      for (const k of Object.keys(keys)) keys[k] = false;
      syncButtons();
      if (navigator.sendBeacon) {
        navigator.sendBeacon("/api/release", new Blob(["{}"], {type: "application/json"}));
      } else {
        fetch("/api/release", {method: "POST", body: "{}", keepalive: true}).catch(() => {});
      }
    }

    window.addEventListener("keydown", event => {
      const k = keyName(event);
      if (!k) return;
      event.preventDefault();
      setSource(`keyboard:${k}`, k, true);
    }, {passive: false});

    window.addEventListener("keyup", event => {
      const k = keyName(event);
      if (!k) return;
      event.preventDefault();
      setSource(`keyboard:${k}`, k, false);
    }, {passive: false});

    for (const button of buttons) {
      button.addEventListener("pointerdown", event => {
        if (event.button !== undefined && event.button !== 0) return;
        event.preventDefault();
        const key = button.dataset.key;
        const source = `pointer:${event.pointerId}`;
        try {
          button.setPointerCapture(event.pointerId);
        } catch (_) {}
        setSource(source, key, true);
      }, {passive: false});

      const releasePointer = event => {
        const source = `pointer:${event.pointerId}`;
        if (!activeSources.has(source)) return;
        event.preventDefault();
        try {
          if (button.hasPointerCapture(event.pointerId)) {
            button.releasePointerCapture(event.pointerId);
          }
        } catch (_) {}
        setSource(source, button.dataset.key, false);
      };

      button.addEventListener("pointerup", releasePointer, {passive: false});
      button.addEventListener("pointercancel", releasePointer, {passive: false});
      button.addEventListener("pointerleave", releasePointer, {passive: false});
      button.addEventListener("lostpointercapture", releasePointer, {passive: false});
      button.addEventListener("contextmenu", event => event.preventDefault());
    }

    window.addEventListener("blur", release);
    window.addEventListener("beforeunload", release);
    syncButtons();
    document.body.focus();
  </script>
</body>
</html>
"""
