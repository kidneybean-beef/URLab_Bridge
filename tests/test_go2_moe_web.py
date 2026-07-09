from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_web_script():
    script_path = Path(__file__).parents[1] / "scripts" / "run_go2_moe_web.py"
    spec = importlib.util.spec_from_file_location("run_go2_moe_web", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_web_policy_parser_defaults_match_browser_control():
    mod = _load_web_script()

    args = mod.build_arg_parser().parse_args([])

    assert args.web_bind == "127.0.0.1"
    assert args.web_port == 8088
    assert args.web_stale_timeout_s == pytest.approx(0.5)
    assert args.max_vx == pytest.approx(1.0)
    assert args.max_vy == pytest.approx(0.5)
    assert args.max_yaw == pytest.approx(1.57)
    assert args.dash_max_vx == pytest.approx(2.0)
    assert args.dash_max_vy == pytest.approx(1.0)
    assert args.dash_max_yaw == pytest.approx(3.14)


def test_web_policy_runtime_ui_sync_uses_selected_articulation(monkeypatch):
    mod = _load_web_script()
    keyboard_globals = mod.run_keyboard_policy.__globals__

    class FakeRuntime:
        def __init__(self) -> None:
            self.control_sources: list[tuple[str, str]] = []

        def set_control_source(self, source: str, *, articulation: str) -> None:
            self.control_sources.append((source, articulation))

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.runtime = FakeRuntime()
            self.articulations = {
                "go2": SimpleNamespace(
                    joints=["joint"],
                    actuators=["actuator"],
                    has_free_base=True,
                    set_ctrl=lambda pose: None,
                )
            }
            self.sim_time = 0.0
            self.closed = False
            self.step_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def connect(self, *args, **kwargs) -> None:
            pass

        def step(self, *args, **kwargs) -> None:
            self.step_calls.append((args, kwargs))

        def close(self) -> None:
            self.closed = True

    class FakeCommandSource:
        quit_requested = True

        def __init__(self) -> None:
            self.synced_articulations: list[str] = []

        def poll(self) -> tuple[float, float, float]:
            return (0.0, 0.0, 0.0)

        def sync_runtime_ui(self, runtime: object, articulation: str) -> bool:
            self.synced_articulations.append(articulation)
            return True

        def release(self) -> None:
            pass

    fake_client_holder: dict[str, FakeClient] = {}

    def make_client(*args, **kwargs):
        client = FakeClient(*args, **kwargs)
        fake_client_holder["client"] = client
        return client

    monkeypatch.setitem(keyboard_globals, "URLabClient", make_client)
    monkeypatch.setitem(keyboard_globals, "_select_articulation", lambda client, name: "go2")
    monkeypatch.setitem(keyboard_globals, "load_go2_moe_policy", lambda *args, **kwargs: object())
    monkeypatch.setitem(keyboard_globals, "format_keyboard_help", lambda config: "help")
    monkeypatch.setitem(keyboard_globals, "resolve_torque_limits", lambda value: [1.0])
    monkeypatch.setitem(keyboard_globals, "_push_gains", lambda *args, **kwargs: 0)
    monkeypatch.setitem(
        keyboard_globals,
        "build_go2_moe_compatibility_report",
        lambda *args, **kwargs: SimpleNamespace(
            model_ok=True,
            stand_ready=True,
            summary=lambda: "ok",
        ),
    )
    monkeypatch.setitem(keyboard_globals, "format_go2_moe_report", lambda report: "")
    monkeypatch.setitem(keyboard_globals, "safety_abort_reason", lambda *args, **kwargs: None)
    monkeypatch.setitem(keyboard_globals, "capture_actuated_joint_pose", lambda art: {"joint": 0.0})
    monkeypatch.setitem(keyboard_globals, "validate_target_pose", lambda pose: None)
    monkeypatch.setitem(keyboard_globals, "build_go2_moe_observation", lambda *args, **kwargs: [0.0])
    monkeypatch.setitem(keyboard_globals, "reset_go2_moe_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(keyboard_globals["signal"], "signal", lambda *args, **kwargs: None)

    args = mod.build_arg_parser().parse_args(["--articulation", "go2"])
    command_source = FakeCommandSource()
    limit_mode = SimpleNamespace(raw_policy=False, action_limit_mode="abort", max_target_step=None)

    result = mod.run_keyboard_policy(args, command_source, limit_mode, SimpleNamespace())

    assert result == 0
    assert command_source.synced_articulations == ["go2", "go2"]
    assert fake_client_holder["client"].runtime.control_sources == [
        ("ui", "go2"),
        ("zmq", "go2"),
        ("ui", "go2"),
    ]
    assert fake_client_holder["client"].step_calls
    assert all(
        kwargs.get("control_articulations") == ("go2",)
        for _args, kwargs in fake_client_holder["client"].step_calls
    )
