from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from urlab_bridge.control_server import (
    RobotRegistry,
    SessionManager,
    WebGateway,
    WebPolicyTarget,
)
from urlab_bridge.control_server.go2_moe import (
    Go2MoeControlLoop,
    Go2MoeDependencies,
)


def _load_multi_web_script():
    script_path = Path(__file__).parents[1] / "scripts" / "run_go2_moe_multi_web.py"
    spec = importlib.util.spec_from_file_location("run_go2_moe_multi_web", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multi_web_parser_accepts_two_articulation_port_targets():
    mod = _load_multi_web_script()

    args = mod.build_arg_parser().parse_args([
        "--web-target", "go2_go2_rl_gym_C_1:8099",
        "--web-target", "go2_go2_rl_gym_C2:8100",
    ])

    targets = mod.parse_web_targets(args)
    assert [(target.articulation, target.port) for target in targets] == [
        ("go2_go2_rl_gym_C_1", 8099),
        ("go2_go2_rl_gym_C2", 8100),
    ]
    assert args.web_bind == "127.0.0.1"
    assert args.max_vx == pytest.approx(1.0)
    assert args.max_vy == pytest.approx(0.5)
    assert args.max_yaw == pytest.approx(1.57)


def test_multi_web_parser_fallback_articulation_port_target():
    mod = _load_multi_web_script()

    args = mod.build_arg_parser().parse_args([
        "--articulation", "dog_a",
        "--web-port", "8099",
    ])

    targets = mod.parse_web_targets(args)
    assert targets == [WebPolicyTarget(articulation="dog_a", port=8099)]


def test_robot_registry_rejects_duplicate_articulation():
    with pytest.raises(SystemExit, match="duplicate articulation"):
        RobotRegistry.from_arg_strings(["dog_a:8099", "dog_a:8100"])


def test_robot_registry_rejects_duplicate_port():
    with pytest.raises(SystemExit, match="duplicate web port"):
        RobotRegistry.from_arg_strings(["dog_a:8099", "dog_b:8099"])


def test_robot_registry_rejects_invalid_target():
    with pytest.raises(SystemExit, match="expected ARTICULATION:PORT"):
        RobotRegistry.from_arg_strings(["dog_a"])


def test_robot_registry_preserves_ordered_control_articulations():
    registry = RobotRegistry.from_arg_strings(["dog_b:8100", "dog_a:8099"])

    assert registry.control_articulations == ("dog_b", "dog_a")

    resolved = registry.resolve(object(), lambda _client, name: f"{name}_resolved")
    assert resolved == ("dog_b_resolved", "dog_a_resolved")
    assert registry.control_articulations == ("dog_b_resolved", "dog_a_resolved")


def test_web_gateway_starts_and_closes_one_server_per_target():
    class FakeCommandSource:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.closed = False
            self.stale_checks = 0

        def stop_if_stale(self) -> None:
            self.stale_checks += 1

        def close(self) -> None:
            self.closed = True

    class FakeServer:
        instances: list["FakeServer"] = []

        def __init__(self, address, handler) -> None:
            self.address = address
            self.handler = handler
            self.shutdown_called = False
            self.server_close_called = False
            FakeServer.instances.append(self)

        def serve_forever(self) -> None:
            pass

        def shutdown(self) -> None:
            self.shutdown_called = True

        def server_close(self) -> None:
            self.server_close_called = True

    class FakeThread:
        instances: list["FakeThread"] = []

        def __init__(self, *, target, daemon) -> None:
            self.target = target
            self.daemon = daemon
            self.started = False
            self.joined = False
            FakeThread.instances.append(self)

        def start(self) -> None:
            self.started = True

        def join(self, timeout=None) -> None:
            self.joined = True

    class FakeEvent:
        def __init__(self) -> None:
            self.set_called = False

        def wait(self, _interval) -> bool:
            return True

        def set(self) -> None:
            self.set_called = True

    targets = [
        WebPolicyTarget("dog_a", 8099),
        WebPolicyTarget("dog_b", 8100),
    ]
    gateway = WebGateway(
        targets=targets,
        bind="127.0.0.1",
        web_config=object(),
        stale_timeout_s=0.5,
        command_source_factory=FakeCommandSource,
        server_factory=FakeServer,
        handler_factory=lambda source: ("handler", source),
        thread_factory=FakeThread,
        event_factory=FakeEvent,
    )

    gateway.start()

    assert len(gateway.handles) == 2
    assert len(gateway.target_sources) == 2
    assert gateway.target_sources[0][1] is not gateway.target_sources[1][1]
    assert [server.address for server in FakeServer.instances] == [
        ("127.0.0.1", 8099),
        ("127.0.0.1", 8100),
    ]
    assert all(thread.started for thread in FakeThread.instances)

    gateway.close()

    assert all(server.shutdown_called for server in FakeServer.instances)
    assert all(server.server_close_called for server in FakeServer.instances)
    assert all(source.closed for _target, source in gateway.target_sources)


def test_session_manager_constructs_one_client_and_closes():
    class FakeClient:
        instances: list["FakeClient"] = []

        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self.runtime = object()
            self.connect_calls: list[dict[str, object]] = []
            self.closed = False
            FakeClient.instances.append(self)

        def connect(self, **kwargs) -> None:
            self.connect_calls.append(kwargs)

        def close(self) -> None:
            self.closed = True

    session = SessionManager(
        address="tcp://127.0.0.1",
        step_port=5559,
        state_port=5555,
        client_factory=FakeClient,
    )

    first = session.connect()
    second = session.connect()
    session.close()

    assert first is second
    assert session.client is first
    assert session.runtime is first.runtime
    assert len(FakeClient.instances) == 1
    assert first.kwargs == {
        "step_mode": "live",
        "step_port": 5559,
        "state_port": 5555,
    }
    assert first.connect_calls == [{"observations": "standard"}]
    assert first.closed is True


def test_multi_web_policy_uses_one_client_and_steps_selected_articulations(monkeypatch):
    mod = _load_multi_web_script()

    class FakeRuntime:
        def __init__(self) -> None:
            self.control_sources: list[tuple[str, str]] = []

        def set_control_source(self, source: str, *, articulation: str) -> None:
            self.control_sources.append((source, articulation))

    class FakeArt:
        def __init__(self, name: str) -> None:
            self.name = name
            self.joints = ["joint"]
            self.actuators = ["actuator"]
            self.has_free_base = True
            self.ctrl_targets: list[object] = []

        def set_ctrl(self, pose: object) -> None:
            self.ctrl_targets.append(pose)

    class FakeClient:
        instances: list["FakeClient"] = []

        def __init__(self, *args, **kwargs) -> None:
            self.runtime = FakeRuntime()
            self.articulations = {
                "dog_a": FakeArt("dog_a"),
                "dog_b": FakeArt("dog_b"),
            }
            self.sim_time = 0.0
            self.step_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            self.closed = False
            FakeClient.instances.append(self)

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
            self.released = False

        def poll(self):
            return (0.0, 0.0, 0.0)

        def sync_runtime_ui(self, runtime: object, articulation: str) -> bool:
            self.synced_articulations.append(articulation)
            return True

        def release(self) -> None:
            self.released = True

    args = mod.build_arg_parser().parse_args([
        "--web-target", "dog_a:8099",
        "--web-target", "dog_b:8100",
    ])
    sources = [FakeCommandSource(), FakeCommandSource()]
    session = SessionManager(
        address=args.address,
        step_port=args.step_port,
        state_port=args.state_port,
        client_factory=FakeClient,
    )
    client = session.connect()
    deps = Go2MoeDependencies(
        select_articulation=lambda _client, name: name,
        format_pose_sample=lambda pose: pose,
        push_gains=lambda *args, **kwargs: 0,
        resolve_torque_limits=lambda value: [1.0],
        sync_command_source_runtime_ui=lambda source, _client, prefix: source.sync_runtime_ui(
            _client.runtime,
            prefix,
        ),
        build_compatibility_report=lambda *args, **kwargs: SimpleNamespace(
            model_ok=True,
            stand_ready=True,
            summary=lambda: "ok",
        ),
        format_compatibility_report=lambda report: "",
        safety_abort_reason=lambda *args, **kwargs: None,
        capture_actuated_joint_pose=lambda art: {"joint": 0.0},
        validate_target_pose=lambda pose: None,
        load_policy=lambda *args, **kwargs: object(),
        build_observation=lambda *args, **kwargs: [0.0],
        reset_history=lambda *args, **kwargs: None,
        infer_action=lambda *args, **kwargs: (
            [0.0],
            SimpleNamespace(expert_weights=[0.0]),
        ),
        action_abort_reason=lambda *args, **kwargs: None,
        apply_action_limit=lambda action, **kwargs: (action, False),
        action_to_target_pose=lambda art, action: {"joint": 0.0},
        maybe_rate_limit_target_pose=lambda previous, desired, **kwargs: desired,
        target_pose_to_action=lambda target: [0.0],
        target_delta_abs_max=lambda previous, target: 0.0,
        signal_handler=lambda *args, **kwargs: None,
    )

    result = Go2MoeControlLoop(
        args,
        list(zip(mod.parse_web_targets(args), sources)),
        mod.resolve_limit_mode(args),
        deps,
    ).run(client)
    session.close()

    assert result == 0
    assert len(FakeClient.instances) == 1
    client = FakeClient.instances[0]
    assert client.runtime.control_sources == [
        ("ui", "dog_a"),
        ("ui", "dog_b"),
        ("zmq", "dog_a"),
        ("zmq", "dog_b"),
        ("ui", "dog_a"),
        ("ui", "dog_b"),
    ]
    assert client.step_calls
    assert all(
        kwargs.get("control_articulations") == ("dog_a", "dog_b")
        for _args, kwargs in client.step_calls
    )
    assert sources[0].synced_articulations == ["dog_a", "dog_a"]
    assert sources[1].synced_articulations == ["dog_b", "dog_b"]
