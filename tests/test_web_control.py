from __future__ import annotations

import json
from io import BytesIO

import pytest


class FakeRuntime:
    def __init__(self, twist_control_state_reply: dict | None = None) -> None:
        self.calls: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = []
        self.twist_control_state_calls: list[tuple[str, dict]] = []
        self.twist_control_state_reply = dict(twist_control_state_reply or {})

    def set_twist(self, articulation: str, *, linear, angular) -> None:
        self.calls.append((articulation, tuple(linear), tuple(angular)))

    def set_twist_control_state(self, articulation: str, **kwargs) -> dict:
        self.twist_control_state_calls.append((articulation, dict(kwargs)))
        return {
            "op": "set_twist_control_state_ok",
            "articulation": articulation,
            "dash_active": bool(kwargs.get("dash_active", False)),
            **self.twist_control_state_reply,
        }


class FakeClient:
    def __init__(self) -> None:
        self.runtime = FakeRuntime()


class FakeClock:
    def __init__(self, now: float = 10.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class UncloseableBytesIO(BytesIO):
    def close(self) -> None:
        return None


class FakeSocket:
    def __init__(self, request: bytes) -> None:
        self.reader = BytesIO(request)
        self.writer = UncloseableBytesIO()

    def makefile(self, mode: str, _buffering: int | None = None):
        if "r" in mode:
            return self.reader
        return self.writer

    def sendall(self, data: bytes) -> None:
        self.writer.write(data)


def handle_raw_http(handler_cls, request: bytes) -> tuple[int, bytes]:
    fake_socket = FakeSocket(request)
    handler_cls(fake_socket, ("127.0.0.1", 12345), object())
    raw = fake_socket.writer.getvalue()
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(None, 2)[1])
    return status, body


def test_twist_from_keys_maps_browser_keys_to_body_twist_signs():
    from urlab_bridge.web_control import KeyState, twist_from_keys

    assert twist_from_keys(KeyState(w=True, q=True, a=True)) == pytest.approx(
        (1.0, 0.5, 1.57)
    )
    assert twist_from_keys(KeyState(s=True, e=True, d=True)) == pytest.approx(
        (-1.0, -0.5, -1.57)
    )


def test_twist_from_keys_brakes_and_cancels_opposite_keys():
    from urlab_bridge.web_control import KeyState, twist_from_keys

    assert twist_from_keys(
        KeyState(w=True, s=True, q=True, e=True, a=True, d=True)
    ) == pytest.approx((0.0, 0.0, 0.0))
    assert twist_from_keys(
        KeyState(w=True, e=True, d=True, space=True, shift=True)
    ) == pytest.approx((0.0, 0.0, 0.0))


def test_twist_from_keys_dash_doubles_and_caps_each_axis():
    from urlab_bridge.web_control import KeyState, WebTwistConfig, twist_from_keys

    config = WebTwistConfig(
        max_vx=0.8,
        max_vy=0.6,
        max_yaw=1.6,
        dash_max_vx=1.3,
        dash_max_vy=0.9,
        dash_max_yaw=3.0,
    )

    assert twist_from_keys(
        KeyState(w=True, e=True, d=True, shift=True),
        config,
    ) == pytest.approx((1.3, -0.9, -3.0))


def test_key_state_from_payload_ignores_unknown_and_falsey_values():
    from urlab_bridge.web_control import KeyState, twist_from_keys

    keys = KeyState.from_payload(
        {
            "w": True,
            "q": False,
            "e": "",
            "a": 0,
            "d": None,
            "space": [],
            "shift": False,
            "ignored": True,
        }
    )

    assert twist_from_keys(keys) == pytest.approx((1.0, 0.0, 0.0))


def test_web_control_broker_sends_twist_and_release_zero():
    from urlab_bridge.web_control import WebControlBroker

    client = FakeClient()
    broker = WebControlBroker(client, articulation="go2")

    reply = broker.apply_control({"keys": {"w": True, "e": True, "d": True}})

    assert reply["twist"] == pytest.approx([1.0, -0.5, -1.57])
    assert client.runtime.calls[-1] == (
        "go2",
        (1.0, -0.5, 0.0),
        (0.0, 0.0, -1.57),
    )

    broker.release()

    assert client.runtime.calls[-1] == (
        "go2",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )


def test_web_control_broker_stale_timeout_sends_zero_once():
    from urlab_bridge.web_control import WebControlBroker

    clock = FakeClock()
    client = FakeClient()
    broker = WebControlBroker(
        client,
        articulation="go2",
        stale_timeout_s=0.5,
        now_fn=clock,
    )

    broker.apply_control({"keys": {"w": True}})
    clock.now += 0.6

    assert broker.stop_if_stale() is True
    assert client.runtime.calls[-1] == (
        "go2",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    call_count = len(client.runtime.calls)

    assert broker.stop_if_stale() is False
    assert len(client.runtime.calls) == call_count


def test_web_command_source_polls_latest_command_and_stales_to_zero():
    from urlab_bridge.web_control import WebCommandSource

    clock = FakeClock()
    source = WebCommandSource(stale_timeout_s=0.5, now_fn=clock)

    source.apply_control({"keys": {"w": True, "d": True}})
    assert source.poll() == pytest.approx((1.0, 0.0, -1.57))
    assert source.quit_requested is False

    clock.now += 0.6

    assert source.poll() == pytest.approx((0.0, 0.0, 0.0))


def test_web_command_source_syncs_dash_display_state_to_runtime():
    from urlab_bridge.web_control import WebCommandSource, WebTwistConfig

    runtime = FakeRuntime(
        {
            "max_vx": 0.25,
            "max_vy": 0.2,
            "max_yaw": 0.4,
            "dash_max_vx": 2.0,
            "dash_max_vy": 1.0,
            "dash_max_yaw": 3.14,
        }
    )
    source = WebCommandSource(
        config=WebTwistConfig(
            max_vx=1.1,
            max_vy=0.7,
            max_yaw=1.8,
            dash_max_vx=2.2,
            dash_max_vy=1.4,
            dash_max_yaw=3.2,
        )
    )

    assert source.sync_runtime_ui(runtime, "go2") is True
    assert runtime.twist_control_state_calls[-1] == (
        "go2",
        {"dash_active": False},
    )

    source.apply_control({"keys": {"w": True, "shift": True}})

    assert source.sync_runtime_ui(runtime, "go2") is True
    assert runtime.twist_control_state_calls[-1][1] == {"dash_active": True}
    assert runtime.twist_control_state_calls[-1][1]["dash_active"] is True
    assert source.poll() == pytest.approx((0.5, 0.0, 0.0))

    source.release()

    assert source.sync_runtime_ui(runtime, "go2") is True
    assert runtime.twist_control_state_calls[-1][1]["dash_active"] is False


def test_web_command_source_stales_shift_only_dash_display_state():
    from urlab_bridge.web_control import WebCommandSource

    clock = FakeClock()
    runtime = FakeRuntime()
    source = WebCommandSource(stale_timeout_s=0.5, now_fn=clock)

    source.apply_control({"keys": {"shift": True}})
    assert source.sync_runtime_ui(runtime, "go2") is True
    assert runtime.twist_control_state_calls[-1][1]["dash_active"] is True

    clock.now += 0.6

    assert source.stop_if_stale() is True
    assert source.sync_runtime_ui(runtime, "go2") is True
    assert runtime.twist_control_state_calls[-1][1]["dash_active"] is False


def test_http_handler_can_drive_in_memory_command_source():
    from urlab_bridge.web_control import WebCommandSource, make_handler

    source = WebCommandSource()
    handler_cls = make_handler(source)
    payload_bytes = json.dumps({"keys": {"s": True, "q": True}}).encode("utf-8")

    status, body = handle_raw_http(
        handler_cls,
        (
            b"POST /api/control HTTP/1.1\r\n"
            b"Host: test\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(payload_bytes)}\r\n\r\n".encode("ascii")
            + payload_bytes
        ),
    )
    payload = json.loads(body)

    assert status == 200
    assert payload["twist"] == pytest.approx([-1.0, 0.5, 0.0])
    assert source.poll() == pytest.approx((-1.0, 0.5, 0.0))


def test_http_handler_serves_metrics_when_provider_is_configured():
    from urlab_bridge.web_control import WebCommandSource, make_handler

    source = WebCommandSource()
    handler_cls = make_handler(
        source,
        metrics_provider=lambda: {
            "tick_count": 7,
            "missed_deadlines": 1,
            "robots": {"go2": {"last_command_age_s": 0.12}},
        },
    )

    status, body = handle_raw_http(
        handler_cls,
        b"GET /metrics HTTP/1.1\r\nHost: test\r\n\r\n",
    )
    payload = json.loads(body)

    assert status == 200
    assert payload["tick_count"] == 7
    assert payload["missed_deadlines"] == 1
    assert payload["robots"]["go2"]["last_command_age_s"] == pytest.approx(0.12)


def test_http_handler_serves_page_and_applies_control():
    from urlab_bridge.web_control import WebControlBroker, make_handler

    client = FakeClient()
    broker = WebControlBroker(client, articulation="go2")
    handler_cls = make_handler(broker)

    status, body = handle_raw_http(
        handler_cls,
        b"GET / HTTP/1.1\r\nHost: test\r\n\r\n",
    )
    assert status == 200
    assert "URLab Web Control" in body.decode("utf-8")

    payload_bytes = json.dumps({"keys": {"q": True, "a": True}}).encode("utf-8")
    status, body = handle_raw_http(
        handler_cls,
        (
            b"POST /api/control HTTP/1.1\r\n"
            b"Host: test\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(payload_bytes)}\r\n\r\n".encode("ascii")
            + payload_bytes
        ),
    )
    payload = json.loads(body)

    assert status == 200
    assert payload["twist"] == pytest.approx([0.0, 0.5, 1.57])
    assert client.runtime.calls[-1] == (
        "go2",
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 1.57),
    )


def test_http_handler_serves_virtual_button_controls():
    from urlab_bridge.web_control import WebControlBroker, make_handler

    broker = WebControlBroker(FakeClient(), articulation="go2")
    handler_cls = make_handler(broker)

    status, body = handle_raw_http(
        handler_cls,
        b"GET / HTTP/1.1\r\nHost: test\r\n\r\n",
    )
    html = body.decode("utf-8")

    assert status == 200
    for key in ("w", "s", "q", "e", "a", "d", "space"):
        assert f'data-key="{key}"' in html
    assert '"strafe-left forward strafe-right"' in html
    assert '"turn-left backward turn-right"' in html
    assert '"brake brake brake"' in html
    assert 'aria-pressed="false"' in html
    assert "syncButtons" in html
    assert "pointerdown" in html


def test_http_handler_rejects_invalid_json():
    from urlab_bridge.web_control import WebControlBroker, make_handler

    broker = WebControlBroker(FakeClient(), articulation="go2")
    handler_cls = make_handler(broker)
    body = b"{not-json"

    status, _ = handle_raw_http(
        handler_cls,
        (
            b"POST /api/control HTTP/1.1\r\n"
            b"Host: test\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        ),
    )

    assert status == 400
