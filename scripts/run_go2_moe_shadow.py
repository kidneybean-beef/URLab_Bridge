#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_client import URLabClient  # noqa: E402
from urlab_policy.go2.unitree_rl_gym_moe import (  # noqa: E402
    GO2_MOE_ACTION_SIZE,
    GO2_MOE_JOINT_NAMES,
    action_to_moe_target_pose,
    build_go2_moe_observation,
    infer_go2_moe_action,
    load_go2_moe_policy,
    reset_go2_moe_history,
)

logger = logging.getLogger("run_go2_moe_shadow")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Unitree RL Gym Go2 CTS/MoE TorchScript policy in shadow "
            "mode against URLab. The script builds live observations and "
            "prints policy outputs, but never sends actions to the robot."
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
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--cmd-vx", type=float, default=0.5)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument(
        "--control-source",
        choices=("ui", "unchanged"),
        default="ui",
        help="set UI control while shadowing so policy outputs cannot actuate",
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

    policy = load_go2_moe_policy(args.policy, device=args.device)
    logger.info("loaded policy checkpoint: %s", args.policy)

    client = URLabClient(
        args.address,
        step_mode="live",
        step_port=args.step_port,
        state_port=args.state_port,
    )
    client.connect(observations="standard")

    prefix = _select_articulation(client, args.articulation)
    art = client.articulations[prefix]
    logger.info(
        "connected: prefix=%s joints=%d actuators=%d free_base=%s",
        prefix,
        len(art.joints),
        len(art.actuators),
        art.has_free_base,
    )
    logger.info("policy joint order: %s", list(GO2_MOE_JOINT_NAMES))

    if args.control_source == "ui":
        client.runtime.set_control_source("ui", articulation=prefix)
        logger.info("control source set to UI; shadow mode will not actuate")

    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)
    last_action = np.zeros(GO2_MOE_ACTION_SIZE, dtype=np.float32)
    stop = False

    def _on_sigint(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    deadline = time.perf_counter() + args.duration
    iters = 0
    try:
        client.step(n_steps=1, observations="standard", target_hz=args.freq)
        obs = build_go2_moe_observation(
            art,
            command=command,
            last_action=last_action,
        )
        reset_go2_moe_history(policy, obs, device=args.device)
        logger.info("primed MoE history from current URLab observation")

        while not stop and time.perf_counter() < deadline:
            client.step(n_steps=1, observations="standard", target_hz=args.freq)
            obs = build_go2_moe_observation(
                art,
                command=command,
                last_action=last_action,
            )
            action, diagnostics = infer_go2_moe_action(
                policy,
                obs,
                device=args.device,
            )
            target_pose = action_to_moe_target_pose(art, action)
            last_action = action

            iters += 1
            if iters == 1 or iters % max(1, int(args.freq)) == 0:
                target_values = np.fromiter(target_pose.values(), dtype=np.float32)
                weights = np.round(diagnostics.expert_weights, 3).tolist()
                latent_norm = (
                    float(np.linalg.norm(diagnostics.latent))
                    if diagnostics.latent.size
                    else 0.0
                )
                logger.info(
                    "step=%d sim=%.2fs obs_norm=%.3f action[min,max]=[%.3f, %.3f] "
                    "target[min,max]=[%.3f, %.3f] weights=%s latent_norm=%.3f "
                    "base_pos=%s",
                    iters,
                    client.sim_time,
                    float(np.linalg.norm(obs)),
                    float(np.min(action)),
                    float(np.max(action)),
                    float(np.min(target_values)),
                    float(np.max(target_values)),
                    weights,
                    latent_norm,
                    np.round(art.root_pos_w, 3).tolist(),
                )
                logger.info(
                    "sample target pose: %s",
                    {k: round(v, 4) for k, v in list(target_pose.items())[:4]},
                )
    finally:
        client.close()

    logger.info("shadow run done after %d iterations; no actions were sent", iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
