#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_client import URLabClient  # noqa: E402
from urlab_policy.go2.pose import capture_actuated_joint_pose  # noqa: E402
from urlab_policy.go2.unitree_rl_gym_moe import (  # noqa: E402
    GO2_MOE_ACTION_SIZE,
    GO2_MOE_ACTION_SCALE,
    GO2_MOE_DEFAULT_DOF_POS,
    GO2_MOE_JOINT_NAMES,
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

logger = logging.getLogger("run_go2_moe_apply")

UNITREE_GO2_POLICY_FREQ_HZ = 50.0
UNITREE_GO2_TEST_DURATION_S = 10.0
UNITREE_GO2_TEST_COMMAND = np.array([0.5, 0.0, 0.0], dtype=np.float32)
UNITREE_GO2_KP = 20.0
UNITREE_GO2_KV = 0.5
UNITREE_GO2_TORQUE_LIMITS = np.array(
    [23.7, 23.7, 35.55] * 4,
    dtype=np.float32,
)
UNITREE_GO2_ACTION_CLIP = 100.0
UNITREE_GO2_COMMAND_DEADZONE = 0.2


class LimitMode:
    def __init__(
        self,
        *,
        raw_policy: bool,
        action_limit_mode: str,
        max_target_step: float | None,
    ) -> None:
        self.raw_policy = raw_policy
        self.action_limit_mode = action_limit_mode
        self.max_target_step = max_target_step


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the Unitree RL Gym Go2 CTS/MoE policy to URLab with a "
            "Unitree-style test setup. The current joint pose is staged "
            "before switching to ZMQ, then the policy runs at the RL Gym "
            "control rate."
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
    parser.add_argument("--freq", type=float, default=UNITREE_GO2_POLICY_FREQ_HZ)
    parser.add_argument("--duration", type=float, default=UNITREE_GO2_TEST_DURATION_S)
    parser.add_argument("--cmd-vx", type=float, default=float(UNITREE_GO2_TEST_COMMAND[0]))
    parser.add_argument("--cmd-vy", type=float, default=float(UNITREE_GO2_TEST_COMMAND[1]))
    parser.add_argument("--cmd-yaw", type=float, default=float(UNITREE_GO2_TEST_COMMAND[2]))
    parser.add_argument("--min-base-z", type=float, default=0.18)
    parser.add_argument("--max-stand-error", type=float, default=0.35)
    parser.add_argument("--max-action-abs", type=float, default=UNITREE_GO2_ACTION_CLIP)
    parser.add_argument(
        "--action-limit-mode",
        choices=("clip", "abort", "none"),
        default="clip",
        help=(
            "clip actions like RL Gym; use abort for strict diagnostics; "
            "use none to observe raw policy output"
        ),
    )
    parser.add_argument(
        "--max-target-step",
        type=float,
        default=None,
        help=(
            "optional maximum per-actuator target change per policy tick, "
            "in radians; omitted by default to match RL Gym"
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--raw-policy",
        action="store_true",
        help=(
            "send policy targets directly: no action clipping and no target "
            "slew limiting. This shows the policy/model behavior honestly."
        ),
    )
    parser.add_argument("--kp", type=float, default=UNITREE_GO2_KP)
    parser.add_argument("--kv", type=float, default=UNITREE_GO2_KV)
    parser.add_argument(
        "--torque-limit",
        type=float,
        default=None,
        help=(
            "override pushed torque limits with one uniform value; omitted "
            "by default to use Unitree Go2 URDF efforts"
        ),
    )
    parser.add_argument(
        "--push-gains",
        action="store_true",
        default=True,
        help="push Unitree RL Gym PD gains before switching to ZMQ",
    )
    parser.add_argument(
        "--no-push-gains",
        dest="push_gains",
        action="store_false",
        help="leave the existing UE PD gains unchanged",
    )
    parser.add_argument(
        "--allow-stand-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="warn, instead of aborting, when current stand pose is not near default",
    )
    parser.add_argument(
        "--strict-stand",
        action="store_true",
        help="abort when current stand pose exceeds --max-stand-error",
    )
    parser.add_argument("--leave-zmq", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_limit_mode(args: argparse.Namespace) -> LimitMode:
    if bool(args.raw_policy):
        return LimitMode(
            raw_policy=True,
            action_limit_mode="none",
            max_target_step=None,
        )
    return LimitMode(
        raw_policy=False,
        action_limit_mode=str(args.action_limit_mode),
        max_target_step=(
            None if args.max_target_step is None else float(args.max_target_step)
        ),
    )


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


def validate_target_pose(target_pose: Mapping[str, float]) -> str | None:
    if not target_pose:
        return "empty target pose"
    for name, value in target_pose.items():
        if not math.isfinite(float(value)):
            return f"non-finite target for {name}: {value!r}"
    return None


def action_abort_reason(action: Sequence[float], *, max_abs: float) -> str | None:
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape != (GO2_MOE_ACTION_SIZE,):
        return f"policy action has shape {arr.shape}, expected (12,)"
    if not np.all(np.isfinite(arr)):
        return f"policy action is non-finite: {arr}"
    peak = float(np.max(np.abs(arr)))
    if peak > float(max_abs):
        return f"policy action abs max {peak:.3f} exceeds {float(max_abs):.3f}"
    return None


def clip_action(action: Sequence[float], *, max_abs: float) -> tuple[np.ndarray, bool]:
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape != (GO2_MOE_ACTION_SIZE,):
        raise ValueError(f"action must have shape (12,), got {arr.shape}")
    clipped = np.clip(arr, -float(max_abs), float(max_abs)).astype(np.float32)
    return clipped, bool(not np.allclose(arr, clipped))


def apply_action_limit(
    action: Sequence[float],
    *,
    max_abs: float,
    mode: str,
) -> tuple[np.ndarray, bool]:
    arr = np.asarray(action, dtype=np.float32)
    if mode == "none":
        if arr.shape != (GO2_MOE_ACTION_SIZE,):
            raise ValueError(f"action must have shape (12,), got {arr.shape}")
        return arr, False
    if mode == "clip":
        return clip_action(arr, max_abs=max_abs)
    if mode == "abort":
        reason = action_abort_reason(arr, max_abs=max_abs)
        if reason is not None:
            raise ValueError(reason)
        return arr, False
    raise ValueError(f"unknown action limit mode {mode!r}")


def command_deadzone_warning(command: Sequence[float]) -> str | None:
    arr = np.asarray(command, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"command must have shape (3,), got {arr.shape}")
    xy_norm = float(np.linalg.norm(arr[:2]))
    if 0.0 < xy_norm <= UNITREE_GO2_COMMAND_DEADZONE + 1e-6:
        return (
            "Unitree RL Gym zeros XY commands with norm <= "
            f"{UNITREE_GO2_COMMAND_DEADZONE:.1f}; command "
            f"{np.round(arr, 4).tolist()} would be zeroed there"
        )
    return None


def safety_abort_reason(art: Any, *, min_base_z: float) -> str | None:
    root_pos = np.asarray(getattr(art, "root_pos_w"), dtype=np.float32)
    if root_pos.shape != (3,):
        return f"root_pos_w has shape {root_pos.shape}, expected (3,)"
    if not np.all(np.isfinite(root_pos)):
        return f"root_pos_w is non-finite: {root_pos}"
    if float(root_pos[2]) < float(min_base_z):
        return f"base z {float(root_pos[2]):.3f} is below {float(min_base_z):.3f}"
    return None


def rate_limit_target_pose(
    previous: Mapping[str, float],
    desired: Mapping[str, float],
    *,
    max_step: float,
) -> dict[str, float]:
    limited: dict[str, float] = {}
    step = abs(float(max_step))
    for name, desired_value in desired.items():
        desired_float = float(desired_value)
        if name not in previous:
            limited[str(name)] = desired_float
            continue
        prev_float = float(previous[name])
        delta = float(np.clip(desired_float - prev_float, -step, step))
        limited[str(name)] = prev_float + delta
    return limited


def maybe_rate_limit_target_pose(
    previous: Mapping[str, float],
    desired: Mapping[str, float],
    *,
    max_step: float | None,
) -> dict[str, float]:
    if max_step is None:
        return {str(name): float(value) for name, value in desired.items()}
    return rate_limit_target_pose(previous, desired, max_step=max_step)


def target_delta_abs_max(
    previous: Mapping[str, float],
    desired: Mapping[str, float],
) -> float:
    deltas = [
        abs(float(value) - float(previous[name]))
        for name, value in desired.items()
        if name in previous
    ]
    return max(deltas) if deltas else 0.0


def target_pose_to_action(
    target_pose: Mapping[str, float],
    *,
    action_scale: float = GO2_MOE_ACTION_SCALE,
) -> np.ndarray:
    action = np.zeros(GO2_MOE_ACTION_SIZE, dtype=np.float32)
    for i, (joint_name, default_qpos) in enumerate(
        zip(GO2_MOE_JOINT_NAMES, GO2_MOE_DEFAULT_DOF_POS)
    ):
        actuator_name = joint_name.removesuffix("_joint")
        if actuator_name not in target_pose:
            raise KeyError(f"target pose missing actuator {actuator_name!r}")
        action[i] = (float(target_pose[actuator_name]) - float(default_qpos)) / float(
            action_scale
        )
    return action


def _format_pose_sample(target_pose: Mapping[str, float], *, count: int = 4) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in list(target_pose.items())[:count]}


def resolve_torque_limits(torque_limit: float | None) -> np.ndarray:
    if torque_limit is None:
        return UNITREE_GO2_TORQUE_LIMITS.copy()
    return np.full(GO2_MOE_ACTION_SIZE, float(torque_limit), dtype=np.float32)


def _push_gains(art: Any, *, kp: float, kv: float, torque_limit: float | None) -> int:
    kp_arr = np.full(GO2_MOE_ACTION_SIZE, kp, dtype=np.float32)
    kv_arr = np.full(GO2_MOE_ACTION_SIZE, kv, dtype=np.float32)
    torque_arr = resolve_torque_limits(torque_limit)
    return int(art.push_gains(GO2_MOE_JOINT_NAMES, kp_arr, kv_arr, torque_arr))


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
    if limit_mode.raw_policy:
        logger.warning(
            "RAW POLICY MODE: no action clipping and no target slew limiting"
        )

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

    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)
    warning = command_deadzone_warning(command)
    if warning is not None:
        logger.warning(warning)
    torque_limits = resolve_torque_limits(args.torque_limit)
    logger.info(
        "Unitree RL Gym test params: freq=%.1fHz command=%s action_clip=%.1f "
        "target_slew=%s warmup_steps=%d push_gains=%s kp=%.3f kv=%.3f "
        "torque_limits=%s",
        args.freq,
        np.round(command, 4).tolist(),
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

        # Stage the live pose while UI is still driving the articulation, so
        # the ZMQ handoff target equals the current Matrix-Go2 stand pose.
        art.set_ctrl(current_pose)
        client.step(n_steps=1, observations="standard")
        client.runtime.set_control_source("zmq", articulation=prefix)
        switched_to_zmq = True
        logger.info("control source set to ZMQ for %s", prefix)
        logger.info("staged current pose sample: %s", _format_pose_sample(current_pose))

        obs = build_go2_moe_observation(
            art,
            command=command,
            last_action=last_action,
        )
        reset_go2_moe_history(policy, obs, device=args.device)
        logger.info(
            "primed MoE history from current observation; command=%s",
            np.round(command, 4).tolist(),
        )

        for warmup_idx in range(max(0, int(args.warmup_steps))):
            if stop:
                break
            art.set_ctrl(current_pose)
            client.step(n_steps=1, observations="standard", target_hz=args.freq)
            if warmup_idx == 0:
                logger.info("holding captured pose for %d warmup steps", args.warmup_steps)

        previous_target = dict(current_pose)
        deadline = time.perf_counter() + args.duration
        while not stop and time.perf_counter() < deadline:
            reason = safety_abort_reason(art, min_base_z=args.min_base_z)
            if reason is not None:
                raise SystemExit(f"safety abort: {reason}")

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
                action
                if limit_mode.raw_policy
                else target_pose_to_action(applied_target)
            )

            if iters == 1 or iters % max(1, int(args.freq)) == 0:
                logger.info(
                    "step=%d sim=%.2fs obs_norm=%.3f action[min,max]=[%.3f, %.3f] "
                    "raw_delta=%.3f applied_delta=%.3f weights=%s base_pos=%s",
                    iters,
                    client.sim_time,
                    float(np.linalg.norm(obs)),
                    float(np.min(action)),
                    float(np.max(action)),
                    target_delta_abs_max(previous_target, desired_target),
                    target_delta_abs_max(previous_target, applied_target),
                    np.round(diagnostics.expert_weights, 3).tolist(),
                    np.round(art.root_pos_w, 3).tolist(),
                )

            previous_target = applied_target
            # The Unitree env feeds the action that was actually applied into
            # the next observation. With target slew limiting, that is the
            # equivalent action represented by the limited target, not the raw
            # network output.
            last_action = applied_action
    finally:
        if switched_to_zmq and not args.leave_zmq:
            try:
                client.runtime.set_control_source("ui", articulation=prefix)
                logger.info("control source restored to UI")
            except Exception as exc:  # pragma: no cover - teardown best effort
                logger.warning("failed to restore UI control source: %s", exc)
        client.close()

    logger.info("MoE apply run done after %d iterations", iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
