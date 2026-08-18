#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
for _path in (_HERE, _SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from run_go2_moe_apply import (  # noqa: E402
    _format_pose_sample,
    _push_gains,
    _select_articulation,
    action_abort_reason,
    apply_action_limit,
    maybe_rate_limit_target_pose,
    resolve_limit_mode,
    resolve_torque_limits,
    safety_abort_reason,
    target_delta_abs_max,
    target_pose_to_action,
    validate_target_pose,
)
from run_go2_moe_keyboard import (  # noqa: E402
    build_arg_parser as build_keyboard_arg_parser,
    sync_command_source_runtime_ui,
)
from urlab_bridge.control_server import (  # noqa: E402
    RobotRegistry,
    SessionManager,
    URLabControlServer,
    WebCameraTarget,
    WebPolicyTarget,
)
from urlab_bridge.control_server.models import parse_web_camera_target  # noqa: E402
from urlab_bridge.control_server.go2_moe import Go2MoeControlLoop, Go2MoeDependencies  # noqa: E402
from urlab_bridge.web_control import WebTwistConfig  # noqa: E402
from urlab_policy.go2.pose import capture_actuated_joint_pose  # noqa: E402
from urlab_policy.go2.unitree_rl_gym_moe import (  # noqa: E402
    GO2_MOE_ACTION_SIZE,
    GO2_MOE_OBS_SIZE,
    action_to_moe_target_pose,
    build_go2_moe_observation,
    infer_go2_moe_action,
    load_go2_moe_policy,
    reset_go2_moe_history,
)
from urlab_policy.go2.unitree_rl_gym_moe_compat import (  # noqa: E402
    build_go2_moe_compatibility_report,
    format_go2_moe_report,
)

logger = logging.getLogger("run_go2_moe_multi_web")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_keyboard_arg_parser()
    parser.description = (
        "Drive multiple Unitree RL Gym Go2 CTS/MoE policies from multiple "
        "browser control pages using one URLab RPC session. Use one "
        "--web-target ARTICULATION:PORT per dog."
    )
    parser.set_defaults(max_vx=1.0, max_vy=0.5, max_yaw=1.57)
    parser.add_argument(
        "--web-target",
        action="append",
        default=[],
        metavar="ARTICULATION:PORT",
        help="serve one web controller for this articulation on this port; repeatable",
    )
    parser.add_argument("--web-bind", default="127.0.0.1")
    parser.add_argument(
        "--web-port",
        type=int,
        default=8088,
        help="fallback port used only with single --articulation and no --web-target",
    )
    parser.add_argument("--web-stale-timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--web-camera",
        action="append",
        default=[],
        metavar="ARTICULATION:CAMERA",
        help=(
            "enable camera controls for this articulation and select the initial "
            "camera; the page discovers all cameras URLab advertises; repeatable"
        ),
    )
    parser.add_argument(
        "--camera-fps",
        type=_positive_float,
        default=20.0,
        help="maximum browser camera encoding rate",
    )
    parser.add_argument(
        "--camera-jpeg-quality",
        type=_jpeg_quality,
        default=80,
        help="browser camera JPEG quality from 1 to 100",
    )
    parser.add_argument("--dash-max-vx", type=float, default=2.0)
    parser.add_argument("--dash-max-vy", type=float, default=1.0)
    parser.add_argument("--dash-max-yaw", type=float, default=3.14)
    smoothing = parser.add_argument_group("command smoothing")
    smoothing.add_argument(
        "--cmd-accel-vx",
        type=_positive_float,
        default=2.0,
        help="forward/backward command acceleration in m/s^2",
    )
    smoothing.add_argument(
        "--cmd-accel-vy",
        type=_positive_float,
        default=1.0,
        help="lateral command acceleration in m/s^2",
    )
    smoothing.add_argument(
        "--cmd-accel-yaw",
        type=_positive_float,
        default=3.14,
        help="yaw command acceleration in rad/s^2",
    )
    smoothing.add_argument(
        "--cmd-decel-vx",
        type=_positive_float,
        default=3.0,
        help="forward/backward command deceleration in m/s^2",
    )
    smoothing.add_argument(
        "--cmd-decel-vy",
        type=_positive_float,
        default=1.5,
        help="lateral command deceleration in m/s^2",
    )
    smoothing.add_argument(
        "--cmd-decel-yaw",
        type=_positive_float,
        default=4.71,
        help="yaw command deceleration in rad/s^2",
    )
    parser.add_argument(
        "--metrics-log-interval-s",
        type=float,
        default=1.0,
        help="seconds between periodic control-loop metric logs; 0 disables",
    )
    parser.add_argument(
        "--policy-benchmark",
        action="store_true",
        help="benchmark Go2 MoE policy inference and exit without connecting to UE",
    )
    ros2 = parser.add_argument_group("ROS2 command gateway")
    ros2.add_argument(
        "--ros2-cmd-vel",
        action="store_true",
        help="subscribe to /<articulation>/cmd_vel and feed Twist commands into the Go2 policy loop",
    )
    ros2.add_argument(
        "--ros2-publish-state",
        action="store_true",
        help="publish /<articulation>/joint_states and /<articulation>/odom",
    )
    ros2.add_argument(
        "--ros2-state-hz",
        type=_positive_float,
        default=None,
        help="optional ROS2 state publish rate; default publishes every control tick",
    )
    ros2.add_argument(
        "--ros2-publish-sensors",
        action="store_true",
        help="publish generic MuJoCo sensors as Float64MultiArray topics",
    )
    ros2.add_argument(
        "--ros2-publish-cameras",
        action="store_true",
        help="publish URLab camera frames for configured --web-camera articulations",
    )
    ros2.add_argument(
        "--ros2-publish-compressed-cameras",
        action="store_true",
        help=(
            "publish JPEG sensor_msgs/CompressedImage preview topics at "
            "/<articulation>/camera/<camera>/image_raw/compressed"
        ),
    )
    ros2.add_argument(
        "--ros2-camera-fps",
        type=_positive_float,
        default=None,
        help="optional ROS2 raw camera publish rate; default follows --camera-fps",
    )
    ros2.add_argument(
        "--ros2-camera-jpeg-quality",
        type=_jpeg_quality,
        default=None,
        help="ROS2 compressed camera JPEG quality from 1 to 100; default follows --camera-jpeg-quality",
    )
    ros2.add_argument(
        "--ros2-camera-log-interval-s",
        type=float,
        default=2.0,
        help="seconds between ROS2 raw camera publish metric logs; 0 disables",
    )
    ros2.add_argument(
        "--ros2-node-name",
        default="urlab_go2_control_server",
        help="ROS2 node name used when any ROS2 bridge feature is enabled",
    )
    return parser


def parse_web_targets(args: argparse.Namespace) -> list[WebPolicyTarget]:
    return list(RobotRegistry.from_args(args).targets)


def parse_web_cameras(
    args: argparse.Namespace,
    targets: list[WebPolicyTarget],
) -> list[WebCameraTarget]:
    cameras = [parse_web_camera_target(raw) for raw in (args.web_camera or [])]
    controlled = {target.articulation for target in targets}
    seen: set[str] = set()
    for camera in cameras:
        if camera.articulation not in controlled:
            raise SystemExit(
                f"--web-camera articulation {camera.articulation!r} is not a "
                "configured --web-target"
            )
        if camera.articulation in seen:
            raise SystemExit(
                f"duplicate --web-camera for articulation {camera.articulation!r}"
            )
        seen.add(camera.articulation)
    return cameras


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _jpeg_quality(raw: str) -> int:
    value = int(raw)
    if value < 1 or value > 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.policy_benchmark:
        return run_policy_benchmark(args)

    limit_mode = resolve_limit_mode(args)
    targets = parse_web_targets(args)
    camera_targets = parse_web_cameras(args, targets)

    web_config = WebTwistConfig(
        max_vx=args.max_vx,
        max_vy=args.max_vy,
        max_yaw=args.max_yaw,
        dash_max_vx=args.dash_max_vx,
        dash_max_vy=args.dash_max_vy,
        dash_max_yaw=args.dash_max_yaw,
    )
    server = URLabControlServer(
        args=args,
        targets=targets,
        limit_mode=limit_mode,
        web_config=web_config,
        dependencies=_build_go2_moe_dependencies(),
        camera_targets=camera_targets,
        camera_fps=args.camera_fps,
        camera_jpeg_quality=args.camera_jpeg_quality,
        log=logger,
    )
    return server.run()


def run_policy_benchmark(args: argparse.Namespace) -> int:
    import torch

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested for --policy-benchmark, but CUDA is unavailable")

    policy = load_go2_moe_policy(args.policy, device=device)
    batch_sizes = (1, 2, 4, 8, 16)
    warmup_iters = 10
    measure_iters = 100
    logger.info(
        "Go2 MoE policy benchmark: policy=%s device=%s warmup=%d iters=%d",
        args.policy,
        device,
        warmup_iters,
        measure_iters,
    )
    print("batch_size,mean_ms,p50_ms,p95_ms")

    for batch_size in batch_sizes:
        obs = np.zeros((batch_size, GO2_MOE_OBS_SIZE), dtype=np.float32)
        reset_go2_moe_history(policy, obs, device=device)
        for _ in range(warmup_iters):
            infer_go2_moe_action(policy, obs, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        durations_ms: list[float] = []
        for _ in range(measure_iters):
            start = time.perf_counter()
            infer_go2_moe_action(policy, obs, device=device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            durations_ms.append((time.perf_counter() - start) * 1000.0)

        values = np.asarray(durations_ms, dtype=np.float64)
        print(
            f"{batch_size},{float(np.mean(values)):.4f},"
            f"{float(np.percentile(values, 50)):.4f},"
            f"{float(np.percentile(values, 95)):.4f}"
        )
    return 0


def run_multi_web_policy(
    args: argparse.Namespace,
    target_sources: list[tuple[WebPolicyTarget, object]],
    limit_mode: object,
) -> int:
    session = SessionManager(
        address=args.address,
        step_port=args.step_port,
        state_port=args.state_port,
    )
    try:
        client = session.connect()
        return Go2MoeControlLoop(
            args,
            target_sources,
            limit_mode,
            _build_go2_moe_dependencies(),
            log=logger,
        ).run(client)
    finally:
        session.close()


def _build_go2_moe_dependencies() -> Go2MoeDependencies:
    return Go2MoeDependencies(
        action_size=GO2_MOE_ACTION_SIZE,
        select_articulation=_select_articulation,
        format_pose_sample=_format_pose_sample,
        push_gains=_push_gains,
        resolve_torque_limits=resolve_torque_limits,
        sync_command_source_runtime_ui=sync_command_source_runtime_ui,
        build_compatibility_report=build_go2_moe_compatibility_report,
        format_compatibility_report=format_go2_moe_report,
        safety_abort_reason=safety_abort_reason,
        capture_actuated_joint_pose=capture_actuated_joint_pose,
        validate_target_pose=validate_target_pose,
        load_policy=load_go2_moe_policy,
        build_observation=build_go2_moe_observation,
        reset_history=reset_go2_moe_history,
        infer_action=infer_go2_moe_action,
        action_abort_reason=action_abort_reason,
        apply_action_limit=apply_action_limit,
        action_to_target_pose=action_to_moe_target_pose,
        maybe_rate_limit_target_pose=maybe_rate_limit_target_pose,
        target_pose_to_action=target_pose_to_action,
        target_delta_abs_max=target_delta_abs_max,
    )


if __name__ == "__main__":
    sys.exit(main())
