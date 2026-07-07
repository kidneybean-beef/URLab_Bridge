#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_client import URLabClient  # noqa: E402
from urlab_policy.go2 import (  # noqa: E402
    GO2_UNITREE_ACTION_SIZE,
    GO2_UNITREE_DEFAULT_DOF_POS,
    GO2_UNITREE_JOINT_NAMES,
    action_to_target_pose,
    build_unitree_go2_observation,
    genesis_default_target_pose,
    infer_unitree_go2_action,
    load_unitree_go2_actor,
)

logger = logging.getLogger("run_go2_policy_apply")

GENESIS_TRAINING_COMMAND = np.array([0.5, 0.0, 0.0], dtype=np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a Genesis locomotion Go2 policy to URLab with no target "
            "blending. UE should already be holding the Genesis default stand "
            "pose before this script switches the articulation to ZMQ."
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
    parser.add_argument(
        "--cmd-vx",
        type=float,
        default=float(GENESIS_TRAINING_COMMAND[0]),
        help="forward velocity command; Genesis Go2 example is trained at 0.5",
    )
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument("--min-base-z", type=float, default=0.20)
    parser.add_argument("--max-action-abs", type=float, default=10.0)
    parser.add_argument("--kp", type=float, default=20.0)
    parser.add_argument("--kv", type=float, default=0.5)
    parser.add_argument("--torque-limit", type=float, default=45.0)
    parser.add_argument(
        "--push-gains",
        action="store_true",
        help="push --kp/--kv/--torque-limit to URLab before switching to ZMQ",
    )
    parser.add_argument("--no-push-gains", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--hold-default-only",
        action="store_true",
        help="switch to ZMQ and hold the Genesis default pose without policy actions",
    )
    parser.add_argument("--leave-zmq", action="store_true")
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


def validate_target_pose(target_pose: Mapping[str, float]) -> str | None:
    if not target_pose:
        return "empty target pose"
    for name, value in target_pose.items():
        if not math.isfinite(float(value)):
            return f"non-finite target for {name}: {value!r}"
    return None


def safety_abort_reason(art: Any, *, min_base_z: float) -> str | None:
    root_pos = np.asarray(getattr(art, "root_pos_w"), dtype=np.float32)
    if root_pos.shape != (3,):
        return f"root_pos_w has shape {root_pos.shape}, expected (3,)"
    if not np.all(np.isfinite(root_pos)):
        return f"root_pos_w is non-finite: {root_pos}"
    if float(root_pos[2]) < float(min_base_z):
        return f"base z {float(root_pos[2]):.3f} is below {float(min_base_z):.3f}"

    quat = np.asarray(getattr(art, "root_quat_xyzw"), dtype=np.float32)
    if quat.shape != (4,):
        return f"root_quat_xyzw has shape {quat.shape}, expected (4,)"
    if not np.all(np.isfinite(quat)):
        return f"root_quat_xyzw is non-finite: {quat}"
    return None


def _action_abort_reason(action: np.ndarray, *, max_abs: float) -> str | None:
    if action.shape != (GO2_UNITREE_ACTION_SIZE,):
        return f"policy action has shape {action.shape}, expected (12,)"
    if not np.all(np.isfinite(action)):
        return f"policy action is non-finite: {action}"
    peak = float(np.max(np.abs(action)))
    if peak > float(max_abs):
        return f"policy action abs max {peak:.3f} exceeds {float(max_abs):.3f}"
    return None


def _format_top_abs(
    label: str,
    names: list[str] | tuple[str, ...],
    values: np.ndarray,
    *,
    count: int = 4,
) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return f"{label}: <empty>"
    order = np.argsort(-np.abs(arr))[:count]
    pairs = ", ".join(f"{names[i]}={float(arr[i]):.4f}" for i in order)
    return f"{label}: {pairs}"


def _resolve_joint_key(art: Any, joint_name: str) -> str | None:
    resolver = getattr(art, "resolve_joint", None)
    if callable(resolver):
        resolved = resolver(joint_name)
        if resolved is not None:
            return resolved
    joints = getattr(art, "joints", {})
    return joint_name if isinstance(joints, Mapping) and joint_name in joints else None


def _ordered_joint_state_for_diagnostics(art: Any) -> tuple[np.ndarray, np.ndarray]:
    qpos_src = np.asarray(getattr(art, "qpos_array"), dtype=np.float32)
    qvel_src = np.asarray(getattr(art, "qvel_array"), dtype=np.float32)
    joints = getattr(art, "joints", {})
    qpos = np.zeros(GO2_UNITREE_ACTION_SIZE, dtype=np.float32)
    qvel = np.zeros(GO2_UNITREE_ACTION_SIZE, dtype=np.float32)

    for i, joint_name in enumerate(GO2_UNITREE_JOINT_NAMES):
        key = _resolve_joint_key(art, joint_name)
        if key is None or key not in joints:
            raise KeyError(f"joint {joint_name!r} not found")
        joint = joints[key]
        qpos[i] = qpos_src[int(getattr(joint, "qpos_local_offset"))]
        qvel[i] = qvel_src[int(getattr(joint, "qvel_local_offset"))]
    return qpos, qvel


def _format_abort_diagnostics(
    art: Any,
    *,
    obs: np.ndarray,
    last_action: np.ndarray,
    action: np.ndarray,
) -> str:
    names = tuple(name.removesuffix("_joint") for name in GO2_UNITREE_JOINT_NAMES)
    parts = [
        _format_top_abs("action", names, action),
        _format_top_abs("last_action", names, last_action),
    ]
    try:
        qpos, qvel = _ordered_joint_state_for_diagnostics(art)
        parts.extend([
            _format_top_abs("qpos_err", names, qpos - GO2_UNITREE_DEFAULT_DOF_POS),
            _format_top_abs("qvel", names, qvel),
        ])
    except Exception as exc:
        parts.append(f"joint diagnostics unavailable: {exc}")

    parts.append(f"base_pos={np.round(getattr(art, 'root_pos_w'), 4).tolist()}")
    parts.append(f"root_ang_vel_b={np.round(getattr(art, 'root_ang_vel_b'), 4).tolist()}")
    parts.append(f"root_quat_xyzw={np.round(getattr(art, 'root_quat_xyzw'), 4).tolist()}")
    parts.append(f"obs_head={np.round(obs[:9], 4).tolist()}")
    parts.append(f"obs_norm={float(np.linalg.norm(obs)):.4f}")
    return "; ".join(parts)


def main() -> int:
    parser = build_arg_parser()
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

    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)
    if not np.allclose(command, GENESIS_TRAINING_COMMAND, atol=1e-6):
        logger.warning(
            "command %s differs from the Genesis example command %s; start "
            "with --cmd-vx 0.5 before testing other commands",
            np.round(command, 4).tolist(),
            GENESIS_TRAINING_COMMAND.tolist(),
        )
    previous_action = np.zeros(GO2_UNITREE_ACTION_SIZE, dtype=np.float32)
    stop = False
    switched_to_zmq = False

    def _on_sigint(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        client.runtime.set_control_source("ui", articulation=prefix)
        logger.info("control source set to UI for preflight")
        client.step(n_steps=1, observations="standard")

        reason = safety_abort_reason(art, min_base_z=args.min_base_z)
        if reason is not None:
            raise SystemExit(f"preflight safety abort: {reason}")

        if args.push_gains and not args.no_push_gains:
            kp = np.full(GO2_UNITREE_ACTION_SIZE, args.kp, dtype=np.float32)
            kv = np.full(GO2_UNITREE_ACTION_SIZE, args.kv, dtype=np.float32)
            torque = np.full(GO2_UNITREE_ACTION_SIZE, args.torque_limit, dtype=np.float32)
            pushed = art.push_gains(GO2_UNITREE_JOINT_NAMES, kp, kv, torque)
            logger.info("pushed Genesis PD gains for %d joints", pushed)
        else:
            logger.info("leaving existing UE PD gains unchanged")

        default_pose = genesis_default_target_pose(art)
        reason = validate_target_pose(default_pose)
        if reason is not None:
            raise SystemExit(f"default pose safety abort: {reason}")

        # Stage NetworkValue while UI is still in control. The first ZMQ frame
        # will therefore match the UE-held Genesis stand pose.
        art.set_ctrl(default_pose)
        client.step(n_steps=1, observations="standard")
        client.runtime.set_control_source("zmq", articulation=prefix)
        switched_to_zmq = True
        logger.info("control source set to ZMQ for %s", prefix)
        logger.info("primed ZMQ with Genesis default pose: %s", {
            k: round(v, 4) for k, v in list(default_pose.items())[:4]
        })

        deadline = time.perf_counter() + args.duration
        iters = 0
        if args.hold_default_only:
            while not stop and time.perf_counter() < deadline:
                reason = safety_abort_reason(art, min_base_z=args.min_base_z)
                if reason is not None:
                    raise SystemExit(f"safety abort: {reason}")

                art.set_ctrl(default_pose)
                client.step(n_steps=1, observations="standard", target_hz=args.freq)
                iters += 1

                if iters == 1 or iters % max(1, int(args.freq)) == 0:
                    logger.info(
                        "hold step=%d sim=%.2fs base_pos=%s",
                        iters,
                        client.sim_time,
                        np.round(art.root_pos_w, 3).tolist(),
                    )

            logger.info("default hold run done after %d iterations", iters)
            return 0

        while not stop and time.perf_counter() < deadline:
            reason = safety_abort_reason(art, min_base_z=args.min_base_z)
            if reason is not None:
                raise SystemExit(f"safety abort: {reason}")

            obs = build_unitree_go2_observation(
                art,
                command=command,
                last_action=previous_action,
            )
            next_action = infer_unitree_go2_action(policy, obs, device=args.device)
            reason = _action_abort_reason(next_action, max_abs=args.max_action_abs)
            if reason is not None:
                detail = _format_abort_diagnostics(
                    art,
                    obs=obs,
                    last_action=previous_action,
                    action=next_action,
                )
                raise SystemExit(f"safety abort: {reason}; {detail}")

            # Genesis training used simulated action latency, so execute the
            # previous policy action. At startup this is zero, i.e. the default
            # stand pose, matching the UE handover pose with no blend.
            target_pose = action_to_target_pose(art, previous_action)
            reason = validate_target_pose(target_pose)
            if reason is not None:
                raise SystemExit(f"safety abort: {reason}")

            art.set_ctrl(target_pose)
            client.step(n_steps=1, observations="standard", target_hz=args.freq)
            iters += 1

            if iters == 1 or iters % max(1, int(args.freq)) == 0:
                target_values = np.fromiter(target_pose.values(), dtype=np.float32)
                logger.info(
                    "step=%d sim=%.2fs obs_norm=%.3f exec_action[min,max]=[%.3f, %.3f] "
                    "next_action[min,max]=[%.3f, %.3f] target[min,max]=[%.3f, %.3f] base_pos=%s",
                    iters,
                    client.sim_time,
                    float(np.linalg.norm(obs)),
                    float(np.min(previous_action)),
                    float(np.max(previous_action)),
                    float(np.min(next_action)),
                    float(np.max(next_action)),
                    float(np.min(target_values)),
                    float(np.max(target_values)),
                    np.round(art.root_pos_w, 3).tolist(),
                )

            previous_action = next_action
    finally:
        if switched_to_zmq and not args.leave_zmq:
            try:
                client.runtime.set_control_source("ui", articulation=prefix)
                logger.info("control source restored to UI")
            except Exception as exc:  # pragma: no cover - teardown best effort
                logger.warning("failed to restore UI control source: %s", exc)
        client.close()

    logger.info("apply run done after %d iterations", iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
