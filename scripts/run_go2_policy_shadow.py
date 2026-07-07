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
from urlab_policy.go2.unitree_policy import (  # noqa: E402
    GO2_UNITREE_JOINT_NAMES,
    action_to_target_pose,
    build_unitree_go2_observation,
    infer_unitree_go2_action,
    load_unitree_go2_actor,
)

logger = logging.getLogger("run_go2_policy_shadow")


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Genesis locomotion Go2 checkpoint in shadow mode against URLab. "
            "The script builds live observations and prints policy outputs, "
            "but never sends actions to the robot."
        )
    )
    parser.add_argument(
        "--policy",
        default=os.path.join(_ROOT, "policies", "model_80.pt"),
        help="Genesis locomotion PPO checkpoint path",
    )
    parser.add_argument("--address", default="tcp://127.0.0.1")
    parser.add_argument("--step-port", type=int, default=5559)
    parser.add_argument("--state-port", type=int, default=5555)
    parser.add_argument("--articulation", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--cmd-vx", type=float, default=0.0)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument(
        "--control-source",
        choices=("ui", "unchanged"),
        default="ui",
        help="set UI control while shadowing so policy outputs cannot actuate",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    policy = load_unitree_go2_actor(args.policy, device=args.device)
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
    logger.info("policy joint order: %s", list(GO2_UNITREE_JOINT_NAMES))

    if args.control_source == "ui":
        client.runtime.set_control_source("ui", articulation=prefix)
        logger.info("control source set to UI; shadow mode will not actuate")

    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)
    last_action = np.zeros(12, dtype=np.float32)
    stop = False

    def _on_sigint(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    deadline = time.perf_counter() + args.duration
    iters = 0
    try:
        while not stop and time.perf_counter() < deadline:
            client.step(n_steps=1, observations="standard", target_hz=args.freq)
            obs = build_unitree_go2_observation(
                art,
                command=command,
                last_action=last_action,
            )
            action = infer_unitree_go2_action(policy, obs, device=args.device)
            target_pose = action_to_target_pose(art, action)
            last_action = action

            iters += 1
            if iters == 1 or iters % max(1, int(args.freq)) == 0:
                target_values = np.fromiter(target_pose.values(), dtype=np.float32)
                logger.info(
                    "step=%d sim=%.2fs obs_norm=%.3f action[min,max]=[%.3f, %.3f] "
                    "target[min,max]=[%.3f, %.3f] base_pos=%s",
                    iters,
                    client.sim_time,
                    float(np.linalg.norm(obs)),
                    float(np.min(action)),
                    float(np.max(action)),
                    float(np.min(target_values)),
                    float(np.max(target_values)),
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
