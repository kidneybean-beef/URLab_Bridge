from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


GO2_MOE_JOINT_NAMES: tuple[str, ...] = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

GO2_MOE_DEFAULT_DOF_POS = np.array(
    [
        0.1,
        0.8,
        -1.5,
        -0.1,
        0.8,
        -1.5,
        0.1,
        1.0,
        -1.5,
        -0.1,
        1.0,
        -1.5,
    ],
    dtype=np.float32,
)

GO2_MOE_CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
GO2_MOE_ANG_VEL_SCALE = 0.25
GO2_MOE_DOF_POS_SCALE = 1.0
GO2_MOE_DOF_VEL_SCALE = 0.05
GO2_MOE_ACTION_SCALE = 0.25
GO2_MOE_OBS_SIZE = 45
GO2_MOE_ACTION_SIZE = 12
GO2_MOE_HISTORY_LENGTH = 5


@dataclass(frozen=True)
class Go2MoeDiagnostics:
    expert_weights: np.ndarray
    latent: np.ndarray


def projected_gravity_from_xyzw(quat_xyzw: Sequence[float]) -> np.ndarray:
    """Return projected gravity from an xyzw root quaternion."""
    q = np.asarray(quat_xyzw, dtype=np.float32)
    if q.shape != (4,):
        raise ValueError(f"quat_xyzw must have shape (4,), got {q.shape}")
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("quat_xyzw has zero norm")
    x, y, z, w = q / norm
    return np.array(
        [
            2.0 * (-z * x + w * y),
            -2.0 * (z * y + w * x),
            1.0 - 2.0 * (w * w + z * z),
        ],
        dtype=np.float32,
    )


def build_go2_moe_observation(
    art: Any,
    *,
    command: Sequence[float] = (0.0, 0.0, 0.0),
    last_action: Sequence[float] | None = None,
    joint_names: Sequence[str] = GO2_MOE_JOINT_NAMES,
    default_dof_pos: Sequence[float] = GO2_MOE_DEFAULT_DOF_POS,
) -> np.ndarray:
    """Build the 45-value Unitree RL Gym Go2 CTS/MoE observation."""
    command_arr = _checked_array(command, "command", 3)
    last_action_arr = (
        np.zeros(GO2_MOE_ACTION_SIZE, dtype=np.float32)
        if last_action is None
        else _checked_array(last_action, "last_action", GO2_MOE_ACTION_SIZE)
    )
    default_arr = _checked_array(default_dof_pos, "default_dof_pos", len(joint_names))

    qpos, qvel = _ordered_joint_state(art, joint_names)
    dof_pos_err = (qpos - default_arr) * GO2_MOE_DOF_POS_SCALE
    dof_vel = qvel * GO2_MOE_DOF_VEL_SCALE
    ang_vel = _checked_array(getattr(art, "root_ang_vel_b"), "root_ang_vel_b", 3)
    projected_gravity = projected_gravity_from_xyzw(getattr(art, "root_quat_xyzw"))

    obs = np.concatenate(
        [
            ang_vel * GO2_MOE_ANG_VEL_SCALE,
            projected_gravity,
            command_arr * GO2_MOE_CMD_SCALE,
            dof_pos_err,
            dof_vel,
            last_action_arr,
        ]
    ).astype(np.float32)
    if obs.shape != (GO2_MOE_OBS_SIZE,):
        raise RuntimeError(f"internal observation size mismatch: {obs.shape}")
    return obs


def action_to_moe_target_pose(
    art: Any,
    action: Sequence[float],
    *,
    joint_names: Sequence[str] = GO2_MOE_JOINT_NAMES,
    default_dof_pos: Sequence[float] = GO2_MOE_DEFAULT_DOF_POS,
    action_scale: float = GO2_MOE_ACTION_SCALE,
) -> dict[str, float]:
    """Convert a CTS/MoE action to URLab actuator position targets."""
    action_arr = _checked_array(action, "action", len(joint_names))
    default_arr = _checked_array(default_dof_pos, "default_dof_pos", len(joint_names))
    target_q = default_arr + action_arr * float(action_scale)

    targets: dict[str, float] = {}
    for joint_name, target in zip(joint_names, target_q):
        actuator_name = _resolve_actuator_for_joint(art, joint_name)
        targets[actuator_name] = float(target)
    return targets


def moe_default_target_pose(art: Any) -> dict[str, float]:
    """Return the Unitree RL Gym Go2 zero-action/default target pose."""
    return action_to_moe_target_pose(
        art,
        np.zeros(GO2_MOE_ACTION_SIZE, dtype=np.float32),
    )


def load_go2_moe_policy(policy_path: str | Path, *, device: str = "cpu") -> Any:
    """Load the exported Unitree RL Gym Go2 CTS/MoE TorchScript policy."""
    import torch

    policy = torch.jit.load(str(Path(policy_path)), map_location=device)
    return policy.eval()


def reset_go2_moe_history(
    policy: Any,
    observation: Sequence[float] | None = None,
    *,
    device: str = "cpu",
) -> bool:
    """Prime a TorchScript MoE policy history with repeated observations."""
    import torch

    obs = (
        np.zeros(GO2_MOE_OBS_SIZE, dtype=np.float32)
        if observation is None
        else _checked_array(observation, "observation", GO2_MOE_OBS_SIZE)
    )
    history_length = int(getattr(policy, "history_length", GO2_MOE_HISTORY_LENGTH))
    current = getattr(policy, "history", None)
    target_device = getattr(current, "device", torch.device(device))
    history = (
        torch.from_numpy(obs)
        .to(device=target_device, dtype=torch.float32)
        .view(1, 1, GO2_MOE_OBS_SIZE)
        .repeat(1, history_length, 1)
    )

    if current is not None and hasattr(current, "copy_"):
        if tuple(current.shape) == tuple(history.shape):
            current.copy_(history)
            return True

    setattr(policy, "history", history)
    return True


def infer_go2_moe_action(
    policy: Any,
    observation: Sequence[float],
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, Go2MoeDiagnostics]:
    """Run the MoE policy and return its action plus expert diagnostics."""
    import torch

    obs = _checked_array(observation, "observation", GO2_MOE_OBS_SIZE)
    with torch.no_grad():
        tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)
        raw_output = policy(tensor)

    action_tensor = raw_output
    extra = None
    if isinstance(raw_output, tuple):
        if not raw_output:
            raise RuntimeError("policy returned an empty tuple")
        action_tensor = raw_output[0]
        extra = raw_output[1] if len(raw_output) > 1 else None

    action = _torch_output_to_numpy(action_tensor, "action")
    if action.shape != (GO2_MOE_ACTION_SIZE,):
        raise RuntimeError(
            f"policy returned action shape {action.shape}, expected (12,)"
        )

    weights = np.zeros(0, dtype=np.float32)
    latent = np.zeros(0, dtype=np.float32)
    if isinstance(extra, tuple) and len(extra) >= 2:
        weights = _torch_output_to_numpy(extra[0], "expert_weights")
        latent = _torch_output_to_numpy(extra[1], "latent")

    return action, Go2MoeDiagnostics(expert_weights=weights, latent=latent)


def _ordered_joint_state(
    art: Any,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    joints = _as_mapping(getattr(art, "joints", None), "joints")
    qpos_src = np.asarray(getattr(art, "qpos_array"), dtype=np.float32)
    qvel_src = np.asarray(getattr(art, "qvel_array"), dtype=np.float32)
    qpos = np.zeros(len(joint_names), dtype=np.float32)
    qvel = np.zeros(len(joint_names), dtype=np.float32)

    for i, joint_name in enumerate(joint_names):
        live_name = _resolve_joint(art, joint_name)
        if live_name is None or live_name not in joints:
            raise KeyError(
                f"joint {joint_name!r} not found on articulation; have {sorted(joints)}"
            )
        joint = joints[live_name]
        if int(getattr(joint, "qpos_dim", 1)) != 1:
            raise ValueError(f"joint {joint_name!r} is not scalar")
        qpos[i] = qpos_src[int(getattr(joint, "qpos_local_offset"))]
        qvel[i] = qvel_src[int(getattr(joint, "qvel_local_offset"))]
    return qpos, qvel


def _resolve_joint(art: Any, joint_name: str) -> str | None:
    resolver = getattr(art, "resolve_joint", None)
    if callable(resolver):
        resolved = resolver(joint_name)
        if resolved is not None:
            return resolved
    joints = getattr(art, "joints", {})
    return joint_name if isinstance(joints, Mapping) and joint_name in joints else None


def _resolve_actuator_for_joint(art: Any, joint_name: str) -> str:
    resolver = getattr(art, "resolve_actuator", None)
    if callable(resolver):
        resolved = resolver(joint_name)
        if resolved is not None:
            return resolved
    actuator_name = joint_name.removesuffix("_joint")
    actuators = getattr(art, "actuators", {})
    if isinstance(actuators, Mapping) and actuator_name in actuators:
        return actuator_name
    raise KeyError(
        f"actuator for joint {joint_name!r} not found; have {sorted(actuators)}"
    )


def _checked_array(values: Sequence[float], name: str, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    return arr


def _torch_output_to_numpy(value: Any, name: str) -> np.ndarray:
    if not hasattr(value, "detach"):
        raise RuntimeError(f"policy returned non-tensor {name}: {type(value)!r}")
    arr = value.detach().cpu().numpy().astype(np.float32)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    return np.asarray(arr, dtype=np.float32)


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"articulation.{field_name} must be a mapping")
    return value
