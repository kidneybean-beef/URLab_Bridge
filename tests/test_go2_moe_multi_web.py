from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from urlab_bridge.control_server import (
    RobotRegistry,
    SessionManager,
    WebGateway,
    WebPolicyTarget,
)
from urlab_bridge.control_server.commands import CommandHub, RobotCommandPort
from urlab_bridge.control_server.go2_moe import (
    Go2MoeControlLoop,
    Go2MoeDependencies,
)
from urlab_bridge.control_server.metrics import ControlLoopMetrics


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
    assert args.cmd_accel_vx == pytest.approx(2.0)
    assert args.cmd_accel_vy == pytest.approx(1.0)
    assert args.cmd_accel_yaw == pytest.approx(3.14)
    assert args.cmd_decel_vx == pytest.approx(3.0)
    assert args.cmd_decel_vy == pytest.approx(1.5)
    assert args.cmd_decel_yaw == pytest.approx(4.71)


def test_multi_web_parser_accepts_optional_ros2_cmd_vel_gateway():
    mod = _load_multi_web_script()

    defaults = mod.build_arg_parser().parse_args([])
    args = mod.build_arg_parser().parse_args([
        "--ros2-cmd-vel",
        "--ros2-node-name",
        "urlab_test_ros2",
        "--ros2-publish-state",
        "--ros2-publish-sensors",
        "--ros2-publish-cameras",
        "--ros2-publish-compressed-cameras",
        "--ros2-state-hz",
        "25",
        "--ros2-camera-fps",
        "12",
        "--ros2-camera-jpeg-quality",
        "70",
        "--ros2-camera-log-interval-s",
        "3",
    ])

    assert defaults.ros2_cmd_vel is False
    assert defaults.ros2_publish_state is False
    assert defaults.ros2_publish_sensors is False
    assert defaults.ros2_publish_cameras is False
    assert defaults.ros2_publish_compressed_cameras is False
    assert defaults.ros2_state_hz is None
    assert defaults.ros2_camera_fps is None
    assert defaults.ros2_camera_jpeg_quality is None
    assert defaults.ros2_camera_log_interval_s == pytest.approx(2.0)
    assert defaults.ros2_node_name == "urlab_go2_control_server"
    assert args.ros2_cmd_vel is True
    assert args.ros2_publish_state is True
    assert args.ros2_publish_sensors is True
    assert args.ros2_publish_cameras is True
    assert args.ros2_publish_compressed_cameras is True
    assert args.ros2_state_hz == pytest.approx(25.0)
    assert args.ros2_camera_fps == pytest.approx(12.0)
    assert args.ros2_camera_jpeg_quality == 70
    assert args.ros2_camera_log_interval_s == pytest.approx(3.0)
    assert args.ros2_node_name == "urlab_test_ros2"


def test_multi_web_parser_maps_rgb_camera_to_web_target():
    mod = _load_multi_web_script()

    args = mod.build_arg_parser().parse_args([
        "--web-target", "dog_a:8099",
        "--web-camera", "dog_a:front_rgb",
        "--camera-fps", "18",
        "--camera-jpeg-quality", "76",
    ])
    targets = mod.parse_web_targets(args)

    assert mod.parse_web_cameras(args, targets) == [
        mod.WebCameraTarget("dog_a", "front_rgb")
    ]
    assert args.camera_fps == pytest.approx(18.0)
    assert args.camera_jpeg_quality == 76


def test_multi_web_parser_maps_same_camera_name_for_two_robots():
    mod = _load_multi_web_script()
    args = mod.build_arg_parser().parse_args([
        "--web-target", "dog_a:8099",
        "--web-target", "dog_b:8100",
        "--web-camera", "dog_a:front_rgb",
        "--web-camera", "dog_b:front_rgb",
    ])

    assert mod.parse_web_cameras(args, mod.parse_web_targets(args)) == [
        mod.WebCameraTarget("dog_a", "front_rgb"),
        mod.WebCameraTarget("dog_b", "front_rgb"),
    ]


def test_multi_web_parser_rejects_camera_for_uncontrolled_robot():
    mod = _load_multi_web_script()
    args = mod.build_arg_parser().parse_args([
        "--web-target", "dog_a:8099",
        "--web-camera", "dog_b:front_rgb",
    ])

    with pytest.raises(SystemExit, match="not a configured --web-target"):
        mod.parse_web_cameras(args, mod.parse_web_targets(args))


def test_multi_web_parser_fallback_articulation_port_target():
    mod = _load_multi_web_script()

    args = mod.build_arg_parser().parse_args([
        "--articulation", "dog_a",
        "--web-port", "8099",
    ])

    targets = mod.parse_web_targets(args)
    assert targets == [WebPolicyTarget(articulation="dog_a", port=8099)]


def test_multi_web_policy_benchmark_uses_report_mode_without_starting_server(monkeypatch):
    mod = _load_multi_web_script()
    benchmark_calls: list[tuple[str, str]] = []

    def fake_benchmark(args) -> int:
        benchmark_calls.append((args.policy, args.device))
        return 0

    class ExplodingServer:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("policy benchmark must not start URLabControlServer")

    monkeypatch.setattr(mod, "run_policy_benchmark", fake_benchmark, raising=False)
    monkeypatch.setattr(mod, "URLabControlServer", ExplodingServer)

    result = mod.main(["--policy-benchmark", "--device", "cpu"])

    assert result == 0
    assert benchmark_calls == [(mod.build_arg_parser().parse_args([]).policy, "cpu")]


def test_policy_benchmark_runs_with_injected_policy_helpers(monkeypatch, capsys):
    mod = _load_multi_web_script()
    infer_shapes: list[tuple[int, int]] = []

    monkeypatch.setattr(mod, "load_go2_moe_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        mod,
        "reset_go2_moe_history",
        lambda *args, **kwargs: True,
    )

    def fake_infer(_policy, observations, *, device):
        infer_shapes.append(tuple(observations.shape))
        return (
            np.zeros((observations.shape[0], 12), dtype=np.float32),
            [SimpleNamespace(expert_weights=np.zeros(8)) for _ in range(observations.shape[0])],
        )

    monkeypatch.setattr(mod, "infer_go2_moe_action", fake_infer)
    args = mod.build_arg_parser().parse_args(["--policy-benchmark", "--device", "cpu"])

    result = mod.run_policy_benchmark(args)

    assert result == 0
    assert (1, 45) in infer_shapes
    assert (16, 45) in infer_shapes
    assert "batch_size,mean_ms,p50_ms,p95_ms" in capsys.readouterr().out


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
        handler_factory=lambda source, metrics_provider=None: (
            "handler",
            source,
            metrics_provider,
        ),
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


def test_web_gateway_creates_robot_command_ports_backed_by_one_hub():
    class FakeServer:
        def __init__(self, address, handler) -> None:
            self.address = address
            self.handler = handler

        def serve_forever(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

        def server_close(self) -> None:
            pass

    class FakeThread:
        def __init__(self, *, target, daemon) -> None:
            self.target = target
            self.daemon = daemon

        def start(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            pass

    class FakeEvent:
        def wait(self, _interval) -> bool:
            return True

        def set(self) -> None:
            pass

    targets = [
        WebPolicyTarget("dog_a", 8099),
        WebPolicyTarget("dog_b", 8100),
    ]
    hub = CommandHub([target.articulation for target in targets])
    metrics = ControlLoopMetrics(freq_hz=50.0)
    gateway = WebGateway(
        targets=targets,
        bind="127.0.0.1",
        web_config=object(),
        stale_timeout_s=0.5,
        command_hub=hub,
        metrics_provider=metrics.snapshot,
        server_factory=FakeServer,
        handler_factory=lambda source, metrics_provider=None: (
            "handler",
            source,
            metrics_provider,
        ),
        thread_factory=FakeThread,
        event_factory=FakeEvent,
    )

    gateway.start()

    sources = [source for _target, source in gateway.target_sources]
    assert all(isinstance(source, RobotCommandPort) for source in sources)
    assert sources[0].hub is hub
    assert sources[1].hub is hub

    sources[0].apply_control({"keys": {"w": True}})

    assert sources[0].poll() == pytest.approx((1.0, 0.0, 0.0))
    assert sources[1].poll() == pytest.approx((0.0, 0.0, 0.0))


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
        infer_action=lambda _policy, observations, **kwargs: (
            np.zeros((len(observations), 1), dtype=np.float32),
            [
                SimpleNamespace(expert_weights=np.zeros(1, dtype=np.float32))
                for _ in range(len(observations))
            ],
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


def test_multi_web_policy_records_control_loop_metrics(monkeypatch):
    mod = _load_multi_web_script()

    class FakeRuntime:
        def __init__(self) -> None:
            self.control_sources: list[tuple[str, str]] = []

        def set_control_source(self, source: str, *, articulation: str) -> None:
            self.control_sources.append((source, articulation))

    class FakeArt:
        joints = ["joint"]
        actuators = ["actuator"]
        has_free_base = True
        root_pos_w = [0.0, 0.0, 0.3]

        def __init__(self) -> None:
            self.ctrl_targets: list[object] = []

        def set_ctrl(self, pose: object) -> None:
            self.ctrl_targets.append(pose)

    class FakeClient:
        def __init__(self) -> None:
            self.runtime = FakeRuntime()
            self.articulations = {"dog_a": FakeArt(), "dog_b": FakeArt()}
            self.sim_time = 0.0
            self.step_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def step(self, *args, **kwargs) -> None:
            self.step_calls.append((args, kwargs))

    class OneTickCommandSource:
        def __init__(self, command) -> None:
            self.command = command
            self.poll_count = 0
            self.released = False

        @property
        def quit_requested(self) -> bool:
            return self.poll_count >= 2

        def poll(self):
            self.poll_count += 1
            return self.command

        def status(self) -> dict:
            return {
                "articulation": "",
                "source": "web",
                "active": any(self.command),
                "stale": False,
                "last_command_age_s": 0.05,
                "twist": list(self.command),
            }

        def stop_if_stale(self) -> bool:
            return False

        def sync_runtime_ui(self, runtime: object, articulation: str) -> bool:
            return True

        def release(self) -> None:
            self.released = True

    args = mod.build_arg_parser().parse_args([
        "--web-target",
        "dog_a:8099",
        "--web-target",
        "dog_b:8100",
    ])
    sources = [
        OneTickCommandSource((1.0, 0.0, 0.0)),
        OneTickCommandSource((0.0, 0.0, 0.0)),
    ]
    metrics = ControlLoopMetrics(freq_hz=args.freq)
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
        infer_action=lambda _policy, observations, **kwargs: (
            np.zeros((len(observations), 1), dtype=np.float32),
            [
                SimpleNamespace(expert_weights=np.zeros(1, dtype=np.float32))
                for _ in range(len(observations))
            ],
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
        metrics=metrics,
        metrics_log_interval_s=0.0,
    ).run(FakeClient())

    snapshot = metrics.snapshot()
    assert result == 0
    assert snapshot["tick_count"] == 1
    assert snapshot["active_robot_count"] == 1
    assert set(snapshot["robots"]) == {"dog_a", "dog_b"}


def test_multi_web_policy_loads_one_policy_and_stacks_two_robot_inference(monkeypatch):
    mod = _load_multi_web_script()

    class FakeRuntime:
        def set_control_source(self, source: str, *, articulation: str) -> None:
            pass

    class FakeArt:
        joints = ["joint"]
        actuators = ["actuator"]
        has_free_base = True
        root_pos_w = [0.0, 0.0, 0.3]

        def __init__(self, name: str) -> None:
            self.name = name
            self.ctrl_targets: list[dict[str, float]] = []

        def set_ctrl(self, pose: dict[str, float]) -> None:
            self.ctrl_targets.append(dict(pose))

    class FakeClient:
        def __init__(self) -> None:
            self.runtime = FakeRuntime()
            self.articulations = {
                "dog_a": FakeArt("dog_a"),
                "dog_b": FakeArt("dog_b"),
            }
            self.sim_time = 0.0

        def step(self, *args, **kwargs) -> None:
            pass

    class OneTickCommandSource:
        def __init__(self, command) -> None:
            self.command = command
            self.poll_count = 0

        @property
        def quit_requested(self) -> bool:
            return self.poll_count >= 2

        def stop_if_stale(self) -> bool:
            return False

        def poll(self):
            self.poll_count += 1
            return self.command

        def status(self) -> dict:
            return {
                "active": any(self.command),
                "stale": False,
                "last_command_age_s": 0.0,
                "twist": list(self.command),
            }

        def sync_runtime_ui(self, runtime: object, articulation: str) -> bool:
            return True

        def release(self) -> None:
            pass

    args = mod.build_arg_parser().parse_args([
        "--web-target",
        "dog_a:8099",
        "--web-target",
        "dog_b:8100",
    ])
    loaded_policies: list[tuple[str, str]] = []
    reset_inputs: list[np.ndarray] = []
    infer_inputs: list[np.ndarray] = []

    def build_observation(art, *, command, last_action):
        return np.array(
            [
                1.0 if art.name == "dog_a" else 2.0,
                float(command[0]),
                float(np.asarray(last_action, dtype=np.float32)[0]),
            ],
            dtype=np.float32,
        )

    def load_policy(path, *, device):
        loaded_policies.append((path, device))
        return "shared-policy"

    def reset_history(policy, observations, *, device):
        assert policy == "shared-policy"
        reset_inputs.append(np.asarray(observations, dtype=np.float32).copy())
        return True

    def infer_action(policy, observations, *, device):
        assert policy == "shared-policy"
        infer_inputs.append(np.asarray(observations, dtype=np.float32).copy())
        return (
            np.array([[0.11], [0.22]], dtype=np.float32),
            [
                SimpleNamespace(expert_weights=np.array([0.1], dtype=np.float32)),
                SimpleNamespace(expert_weights=np.array([0.2], dtype=np.float32)),
            ],
        )

    deps = Go2MoeDependencies(
        action_size=1,
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
        load_policy=load_policy,
        build_observation=build_observation,
        reset_history=reset_history,
        infer_action=infer_action,
        action_abort_reason=lambda *args, **kwargs: None,
        apply_action_limit=lambda action, **kwargs: (action, False),
        action_to_target_pose=lambda art, action: {"joint": float(action[0])},
        maybe_rate_limit_target_pose=lambda previous, desired, **kwargs: desired,
        target_pose_to_action=lambda target: [target["joint"]],
        target_delta_abs_max=lambda previous, target: 0.0,
        signal_handler=lambda *args, **kwargs: None,
    )
    sources = [
        OneTickCommandSource((1.0, 0.0, 0.0)),
        OneTickCommandSource((0.0, 0.0, 0.0)),
    ]
    client = FakeClient()

    result = Go2MoeControlLoop(
        args,
        list(zip(mod.parse_web_targets(args), sources)),
        mod.resolve_limit_mode(args),
        deps,
        metrics_log_interval_s=0.0,
    ).run(client)

    assert result == 0
    assert loaded_policies == [(args.policy, args.device)]
    assert len(reset_inputs) == 1
    assert reset_inputs[0].shape == (2, 3)
    assert np.allclose(reset_inputs[0][:, 0], [1.0, 2.0])
    assert np.allclose(reset_inputs[0][:, 1], [0.0, 0.0])
    assert len(infer_inputs) == 1
    assert infer_inputs[0].shape == (2, 3)
    assert np.allclose(infer_inputs[0][:, 1], [0.04, 0.0])
    assert client.articulations["dog_a"].ctrl_targets[-1] == {
        "joint": pytest.approx(0.11)
    }
    assert client.articulations["dog_b"].ctrl_targets[-1] == {
        "joint": pytest.approx(0.22)
    }
