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
for _path in (_HERE, _SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from run_go2_moe_apply import (  # noqa: E402
    UNITREE_GO2_ACTION_CLIP,
    UNITREE_GO2_KP,
    UNITREE_GO2_KV,
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
from urlab_client import URLabClient  # noqa: E402
from urlab_policy.go2.pose import capture_actuated_joint_pose  # noqa: E402
from urlab_policy.go2.twist_commands import URLabTwistCommandSource  # noqa: E402
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

logger = logging.getLogger("run_go2_moe_ue_twist")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drive the Unitree RL Gym Go2 CTS/MoE policy from UE viewport "
            "input. Possess the Go2 in UE, focus the viewport, and this "
            "script reads URLab's published twist command each policy tick."
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
    parser.add_argument("--twist-vx-sign", type=float, default=1.0)
    parser.add_argument("--twist-vy-sign", type=float, default=-1.0)
    parser.add_argument("--twist-yaw-sign", type=float, default=-1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="run duration in seconds; 0 means until Ctrl+C",
    )
    parser.add_argument("--min-base-z", type=float, default=0.18)
    parser.add_argument("--max-stand-error", type=float, default=0.35)
    parser.add_argument("--max-action-abs", type=float, default=UNITREE_GO2_ACTION_CLIP)
    parser.add_argument(
        "--action-limit-mode",
        choices=("clip", "abort", "none"),
        default="clip",
    )
    parser.add_argument("--max-target-step", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--raw-policy", action="store_true")
    parser.add_argument("--kp", type=float, default=UNITREE_GO2_KP)
    parser.add_argument("--kv", type=float, default=UNITREE_GO2_KV)
    parser.add_argument("--torque-limit", type=float, default=None)
    parser.add_argument("--push-gains", action="store_true", default=True)
    parser.add_argument("--no-push-gains", dest="push_gains", action="store_false")
    parser.add_argument(
        "--allow-stand-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--strict-stand", action="store_true")
    parser.add_argument("--leave-zmq", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    limit_mode = resolve_limit_mode(args)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    policy = load_go2_moe_policy(args.policy, device=args.device)
    logger.info("loaded policy checkpoint: %s", args.policy)
    logger.info(
        "UE input mode: click Possess for the Go2, focus the UE viewport, "
        "then use the URLab Twist input mapping. Stop this script with Ctrl+C."
    )
    if limit_mode.raw_policy:
        logger.warning("RAW POLICY MODE: no action clipping and no target slew limiting")

    client = URLabClient(
        args.address,
        step_mode="live",
        step_port=args.step_port,
        state_port=args.state_port,
    )
    client.connect(observations="standard")

    prefix = _select_articulation(client, args.articulation)
    art = client.articulations[prefix]
    twist_axis_signs = (
        args.twist_vx_sign,
        args.twist_vy_sign,
        args.twist_yaw_sign,
    )
    command_source = URLabTwistCommandSource(art, axis_signs=twist_axis_signs)
    logger.info(
        "connected: prefix=%s joints=%d actuators=%d free_base=%s",
        prefix,
        len(art.joints),
        len(art.actuators),
        art.has_free_base,
    )

    torque_limits = resolve_torque_limits(args.torque_limit)
    logger.info(
        "UE-twist MoE params: freq=%.1fHz twist_axis_signs=%s "
        "action_clip=%.1f target_slew=%s warmup_steps=%d push_gains=%s "
        "kp=%.3f kv=%.3f torque_limits=%s",
        args.freq,
        np.round(np.asarray(twist_axis_signs, dtype=np.float32), 3).tolist(),
        args.max_action_abs,
        "none" if limit_mode.max_target_step is None else limit_mode.max_target_step,
        args.warmup_steps,
        bool(args.push_gains),
        args.kp,
        args.kv,
        np.round(torque_limits, 3).tolist(),
    )

    last_action = np.zeros(GO2_MOE_ACTION_SIZE, dtype=np.float32)
    stop = False
    switched_to_zmq = False
    iters = 0

    def _on_sigint(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        client.runtime.set_control_source("ui", articulation=prefix)
        logger.info("control source set to UI for preflight")
        client.step(n_steps=1, observations="standard")

        report = build_go2_moe_compatibility_report(
            art,
            max_stand_error=args.max_stand_error,
        )
        for line in format_go2_moe_report(report).splitlines():
            logger.info(line)
        if not report.model_ok:
            raise SystemExit(f"preflight abort: {report.summary()}")
        if not report.stand_ready:
            if args.strict_stand or not args.allow_stand_mismatch:
                raise SystemExit(f"stand preflight abort: {report.summary()}")
            logger.warning("continuing despite stand mismatch: %s", report.summary())

        reason = safety_abort_reason(art, min_base_z=args.min_base_z)
        if reason is not None:
            raise SystemExit(f"preflight safety abort: {reason}")

        if args.push_gains:
            pushed = _push_gains(
                art,
                kp=args.kp,
                kv=args.kv,
                torque_limit=args.torque_limit,
            )
            logger.info("pushed Unitree RL Gym PD gains for %d joints", pushed)
        else:
            logger.info("leaving existing UE PD gains unchanged")

        current_pose = capture_actuated_joint_pose(art)
        reason = validate_target_pose(current_pose)
        if reason is not None:
            raise SystemExit(f"captured pose abort: {reason}")

        art.set_ctrl(current_pose)
        client.step(n_steps=1, observations="standard")
        client.runtime.set_control_source("zmq", articulation=prefix)
        switched_to_zmq = True
        logger.info("control source set to ZMQ for %s", prefix)
        logger.info("staged current pose sample: %s", _format_pose_sample(current_pose))

        command = command_source.poll()
        obs = build_go2_moe_observation(
            art,
            command=command,
            last_action=last_action,
        )
        reset_go2_moe_history(policy, obs, device=args.device)
        logger.info("primed MoE history; command=%s", np.round(command, 4).tolist())

        for warmup_idx in range(max(0, int(args.warmup_steps))):
            if stop:
                break
            art.set_ctrl(current_pose)
            client.step(n_steps=1, observations="standard", target_hz=args.freq)
            if warmup_idx == 0:
                logger.info("holding captured pose for %d warmup steps", args.warmup_steps)

        previous_target = dict(current_pose)
        deadline = (
            None
            if float(args.duration) <= 0.0
            else time.perf_counter() + float(args.duration)
        )
        while not stop:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            reason = safety_abort_reason(art, min_base_z=args.min_base_z)
            if reason is not None:
                raise SystemExit(f"safety abort: {reason}")

            command = command_source.poll()
            obs = build_go2_moe_observation(
                art,
                command=command,
                last_action=last_action,
            )
            action, diagnostics = infer_go2_moe_action(policy, obs, device=args.device)
            reason = action_abort_reason(action, max_abs=args.max_action_abs)
            if reason is not None:
                if limit_mode.action_limit_mode == "abort":
                    raise SystemExit(f"safety abort: {reason}")
                if limit_mode.action_limit_mode == "clip":
                    logger.warning("clipping policy action: %s", reason)
                action, _ = apply_action_limit(
                    action,
                    max_abs=args.max_action_abs,
                    mode=limit_mode.action_limit_mode,
                )

            desired_target = action_to_moe_target_pose(art, action)
            reason = validate_target_pose(desired_target)
            if reason is not None:
                raise SystemExit(f"safety abort: {reason}")
            applied_target = maybe_rate_limit_target_pose(
                previous_target,
                desired_target,
                max_step=limit_mode.max_target_step,
            )

            art.set_ctrl(applied_target)
            client.step(n_steps=1, observations="standard", target_hz=args.freq)
            iters += 1
            applied_action = (
                action if limit_mode.raw_policy else target_pose_to_action(applied_target)
            )

            if iters == 1 or iters % max(1, int(args.freq)) == 0:
                logger.info(
                    "step=%d sim=%.2fs cmd=%s obs_norm=%.3f "
                    "action[min,max]=[%.3f, %.3f] raw_delta=%.3f "
                    "applied_delta=%.3f weights=%s base_pos=%s",
                    iters,
                    client.sim_time,
                    np.round(command, 3).tolist(),
                    float(np.linalg.norm(obs)),
                    float(np.min(action)),
                    float(np.max(action)),
                    target_delta_abs_max(previous_target, desired_target),
                    target_delta_abs_max(previous_target, applied_target),
                    np.round(diagnostics.expert_weights, 3).tolist(),
                    np.round(art.root_pos_w, 3).tolist(),
                )

            previous_target = applied_target
            last_action = applied_action
    finally:
        if switched_to_zmq and not args.leave_zmq:
            try:
                client.runtime.set_control_source("ui", articulation=prefix)
                logger.info("control source restored to UI")
            except Exception as exc:  # pragma: no cover - teardown best effort
                logger.warning("failed to restore UI control source: %s", exc)
        client.close()

    logger.info("UE-twist MoE run done after %d iterations", iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
