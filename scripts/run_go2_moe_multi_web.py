#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys

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
    WebPolicyTarget,
)
from urlab_bridge.control_server.go2_moe import Go2MoeControlLoop, Go2MoeDependencies  # noqa: E402
from urlab_bridge.web_control import WebTwistConfig  # noqa: E402
from urlab_policy.go2.pose import capture_actuated_joint_pose  # noqa: E402
from urlab_policy.go2.unitree_rl_gym_moe import (  # noqa: E402
    GO2_MOE_ACTION_SIZE,
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
    parser.add_argument("--dash-max-vx", type=float, default=2.0)
    parser.add_argument("--dash-max-vy", type=float, default=1.0)
    parser.add_argument("--dash-max-yaw", type=float, default=3.14)
    parser.add_argument(
        "--metrics-log-interval-s",
        type=float,
        default=1.0,
        help="seconds between periodic control-loop metric logs; 0 disables",
    )
    return parser


def parse_web_targets(args: argparse.Namespace) -> list[WebPolicyTarget]:
    return list(RobotRegistry.from_args(args).targets)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    limit_mode = resolve_limit_mode(args)
    targets = parse_web_targets(args)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

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
        log=logger,
    )
    return server.run()


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
