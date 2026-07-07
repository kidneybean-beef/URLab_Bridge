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
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_client import URLabClient  # noqa: E402
from urlab_policy.go2 import (  # noqa: E402
    capture_actuated_joint_pose,
    pose_with_sine_offset,
)

logger = logging.getLogger("run_go2_capture_hold")


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
            "Capture the current Go2 actuated joint pose and hold it from "
            "Python through URLab Bridge. Intended as a safe ZMQ takeover test "
            "after the robot has already stood up in UE."
        )
    )
    parser.add_argument("--address", default="tcp://127.0.0.1")
    parser.add_argument("--step-port", type=int, default=5559)
    parser.add_argument("--state-port", type=int, default=5555)
    parser.add_argument("--step-mode", default="live", choices=["live", "direct"])
    parser.add_argument("--articulation", default="")
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sim-decimation", type=int, default=1)
    parser.add_argument(
        "--sine-actuator",
        default="",
        help="optional local actuator name to move around the captured pose",
    )
    parser.add_argument(
        "--sine-amplitude",
        type=float,
        default=0.0,
        help="sinusoidal offset amplitude in radians",
    )
    parser.add_argument(
        "--sine-frequency",
        type=float,
        default=0.5,
        help="sinusoidal offset frequency in Hz",
    )
    parser.add_argument(
        "--observations",
        default="standard",
        choices=["minimal", "standard", "full"],
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    client = URLabClient(
        args.address,
        step_mode=args.step_mode,
        step_port=args.step_port,
        state_port=args.state_port,
    )
    client.connect(observations=args.observations)

    prefix = _select_articulation(client, args.articulation)
    art = client.articulations[prefix]
    logger.info(
        "connected: prefix=%s joints=%d actuators=%d free_base=%s",
        prefix,
        len(art.joints),
        len(art.actuators),
        art.has_free_base,
    )

    stop = False

    def _on_sigint(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        # Start from the UE/InternalValue side while reading the current pose.
        # This makes the script safe even if a previous experiment left the
        # articulation on ZMQ with stale network controls.
        client.runtime.set_control_source("ui", articulation=prefix)
        logger.info("control source set to UI for capture")

        # Read the current MuJoCo state before changing the articulation's
        # control source. While UE is still using UI/InternalValue, this step
        # should not disturb the stand-up controller's hold.
        client.step(n_steps=1, observations=args.observations)
        hold_pose = capture_actuated_joint_pose(art)

        logger.info("captured %d actuator targets from current qpos", len(hold_pose))
        logger.info("sample targets: %s", {
            k: round(v, 4) for k, v in list(hold_pose.items())[:4]
        })
        if args.sine_actuator:
            if args.sine_actuator not in hold_pose:
                raise SystemExit(
                    f"sine actuator {args.sine_actuator!r} not in captured pose; "
                    f"available: {list(hold_pose)}"
                )
            logger.info(
                "sine offset enabled: actuator=%s amplitude=%.4f rad frequency=%.3f Hz",
                args.sine_actuator,
                args.sine_amplitude,
                args.sine_frequency,
            )

        # Prime NetworkValue with the captured pose before switching to ZMQ.
        # This avoids a one-frame jump to stale zero/random network controls.
        art.set_ctrl(hold_pose)
        client.step(n_steps=1, observations=args.observations)
        client.runtime.set_control_source("zmq", articulation=prefix)
        logger.info("control source set to ZMQ for %s", prefix)

        deadline = time.perf_counter() + args.duration
        motion_start = time.perf_counter()
        iters = 0
        while not stop and time.perf_counter() < deadline:
            if args.sine_actuator and args.sine_amplitude != 0.0:
                ctrl_pose = pose_with_sine_offset(
                    hold_pose,
                    actuator_name=args.sine_actuator,
                    amplitude=args.sine_amplitude,
                    frequency_hz=args.sine_frequency,
                    elapsed_s=time.perf_counter() - motion_start,
                )
            else:
                ctrl_pose = hold_pose
            art.set_ctrl(ctrl_pose)
            client.step(
                n_steps=args.sim_decimation,
                observations=args.observations,
                target_hz=args.freq,
            )
            iters += 1
            if iters % max(1, int(args.freq)) == 0 and art.has_free_base:
                logger.info(
                    "t=%.2fs base_pos=%s base_lin_vel=%s",
                    client.sim_time,
                    np.round(art.root_pos_w, 3).tolist(),
                    np.round(art.root_lin_vel_w, 3).tolist(),
                )
    finally:
        client.close()

    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
