#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_client import URLabClient  # noqa: E402
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

logger = logging.getLogger("run_go2_moe_preflight")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the live URLab Go2 articulation matches the "
            "Unitree RL Gym Go2 CTS/MoE policy contract. This script does "
            "not send controls and does not switch to ZMQ."
        )
    )
    parser.add_argument(
        "--policy",
        default=os.path.join(_ROOT, "policies", "go2-rl-gym", "moe_cts.pt"),
        help="Unitree RL Gym Go2 CTS/MoE TorchScript policy path",
    )
    parser.add_argument("--address", default="tcp://127.0.0.1")
    parser.add_argument("--step-port", type=int, default=5559)
    parser.add_argument("--state-port", type=int, default=5555)
    parser.add_argument("--articulation", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cmd-vx", type=float, default=0.5)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument("--max-stand-error", type=float, default=0.35)
    parser.add_argument(
        "--control-source",
        choices=("ui", "unchanged"),
        default="ui",
        help="set UI control while checking so policy outputs cannot actuate",
    )
    parser.add_argument(
        "--skip-policy",
        action="store_true",
        help="only check model/pose compatibility; do not load the policy artifact",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="return exit code 0 even when the compatibility report is not ok",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _select_articulation(client: URLabClient, requested: str) -> str:
    if requested:
        if requested not in client.articulations:
            raise SystemExit(
                f"articulation {requested!r} not found; available: "
                f"{list(client.articulations)}"
            )
        return requested
    if len(client.articulations) == 1:
        return next(iter(client.articulations))
    raise SystemExit(
        f"multiple articulations available {list(client.articulations)}; "
        "pass --articulation"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    client = URLabClient(
        args.address,
        step_mode="live",
        step_port=args.step_port,
        state_port=args.state_port,
    )
    client.connect(observations="standard")

    try:
        prefix = _select_articulation(client, args.articulation)
        art = client.articulations[prefix]
        logger.info(
            "connected: prefix=%s joints=%d actuators=%d free_base=%s",
            prefix,
            len(art.joints),
            len(art.actuators),
            art.has_free_base,
        )

        if args.control_source == "ui":
            client.runtime.set_control_source("ui", articulation=prefix)
            logger.info("control source set to UI; preflight will not actuate")

        client.step(n_steps=1, observations="standard")
        report = build_go2_moe_compatibility_report(
            art,
            max_stand_error=args.max_stand_error,
        )
        for line in format_go2_moe_report(report).splitlines():
            logger.info(line)

        if report.model_ok and not args.skip_policy:
            _run_policy_smoke(args, art)

        if report.ok or args.allow_mismatch:
            return 0
        return 2
    finally:
        client.close()


def _run_policy_smoke(args: argparse.Namespace, art) -> None:
    policy = load_go2_moe_policy(args.policy, device=args.device)
    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)
    last_action = np.zeros(GO2_MOE_ACTION_SIZE, dtype=np.float32)
    obs = build_go2_moe_observation(
        art,
        command=command,
        last_action=last_action,
    )
    reset_go2_moe_history(policy, obs, device=args.device)
    action, diagnostics = infer_go2_moe_action(policy, obs, device=args.device)
    target_pose = action_to_moe_target_pose(art, action)
    target_values = np.fromiter(target_pose.values(), dtype=np.float32)
    logger.info(
        "policy_smoke: obs_norm=%.3f action[min,max]=[%.3f, %.3f] "
        "target[min,max]=[%.3f, %.3f] weights=%s latent_norm=%.3f",
        float(np.linalg.norm(obs)),
        float(np.min(action)),
        float(np.max(action)),
        float(np.min(target_values)),
        float(np.max(target_values)),
        np.round(diagnostics.expert_weights, 3).tolist(),
        float(np.linalg.norm(diagnostics.latent)) if diagnostics.latent.size else 0.0,
    )


if __name__ == "__main__":
    sys.exit(main())
