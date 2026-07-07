from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


GO2_UNITREE_JOINT_NAMES: tuple[str, ...] = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)

GO2_UNITREE_DEFAULT_DOF_POS = np.array(
    [
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
        0.0, 1.0, -1.5,
        0.0, 1.0, -1.5,
    ],
    dtype=np.float32,
)

GO2_UNITREE_CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
GO2_UNITREE_ANG_VEL_SCALE = 0.25
GO2_UNITREE_DOF_POS_SCALE = 1.0
GO2_UNITREE_DOF_VEL_SCALE = 0.05
GO2_UNITREE_ACTION_SCALE = 0.25
GO2_UNITREE_OBS_SIZE = 45
GO2_UNITREE_ACTION_SIZE = 12


def projected_gravity_from_xyzw(quat_xyzw: Sequence[float]) -> np.ndarray:
    """Return Genesis Go2 projected gravity from an xyzw quaternion."""
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


def build_unitree_go2_observation(
    art: Any,
    *,
    command: Sequence[float] = (0.0, 0.0, 0.0),
    last_action: Sequence[float] | None = None,
    joint_names: Sequence[str] = GO2_UNITREE_JOINT_NAMES,
    default_dof_pos: Sequence[float] = GO2_UNITREE_DEFAULT_DOF_POS,
) -> np.ndarray:
    """Build the 45-dim Genesis locomotion Go2 actor observation from URLab state."""
    command_arr = _checked_array(command, "command", 3)
    last_action_arr = (
        np.zeros(GO2_UNITREE_ACTION_SIZE, dtype=np.float32)
        if last_action is None
        else _checked_array(last_action, "last_action", GO2_UNITREE_ACTION_SIZE)
    )
    default_arr = _checked_array(default_dof_pos, "default_dof_pos", len(joint_names))

    qpos, qvel = _ordered_joint_state(art, joint_names)
    dof_pos_err = (qpos - default_arr) * GO2_UNITREE_DOF_POS_SCALE
    dof_vel = qvel * GO2_UNITREE_DOF_VEL_SCALE
    ang_vel = _checked_array(getattr(art, "root_ang_vel_b"), "root_ang_vel_b", 3)
    projected_gravity = projected_gravity_from_xyzw(getattr(art, "root_quat_xyzw"))

    obs = np.concatenate(
        [
            ang_vel * GO2_UNITREE_ANG_VEL_SCALE,
            projected_gravity,
            command_arr * GO2_UNITREE_CMD_SCALE,
            dof_pos_err,
            dof_vel,
            last_action_arr,
        ]
    ).astype(np.float32)
    if obs.shape != (GO2_UNITREE_OBS_SIZE,):
        raise RuntimeError(f"internal observation size mismatch: {obs.shape}")
    return obs


def action_to_target_pose(
    art: Any,
    action: Sequence[float],
    *,
    joint_names: Sequence[str] = GO2_UNITREE_JOINT_NAMES,
    default_dof_pos: Sequence[float] = GO2_UNITREE_DEFAULT_DOF_POS,
    action_scale: float = GO2_UNITREE_ACTION_SCALE,
) -> dict[str, float]:
    """Convert Genesis actor actions to URLab actuator position targets."""
    action_arr = _checked_array(action, "action", len(joint_names))
    default_arr = _checked_array(default_dof_pos, "default_dof_pos", len(joint_names))
    target_q = default_arr + action_arr * float(action_scale)

    targets: dict[str, float] = {}
    for joint_name, target in zip(joint_names, target_q):
        actuator_name = _resolve_actuator_for_joint(art, joint_name)
        targets[actuator_name] = float(target)
    return targets


def genesis_default_target_pose(art: Any) -> dict[str, float]:
    """Return Genesis Go2 zero-action/default target pose by actuator name."""
    return action_to_target_pose(
        art,
        np.zeros(GO2_UNITREE_ACTION_SIZE, dtype=np.float32),
    )


def load_unitree_go2_actor(checkpoint_path: str | Path, *, device: str = "cpu"):
    """Load the actor MLP from a Genesis locomotion PPO checkpoint."""
    import torch

    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, Mapping) else None
    if state is None:
        raise ValueError(f"{path} does not contain 'model_state_dict'")

    model = torch.nn.Sequential(
        torch.nn.Linear(GO2_UNITREE_OBS_SIZE, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, GO2_UNITREE_ACTION_SIZE),
    )
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    missing, unexpected = model.load_state_dict(actor_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"actor checkpoint shape mismatch; missing={missing}, unexpected={unexpected}"
        )
    return model.to(device).eval()


def infer_unitree_go2_action(model: Any, observation: Sequence[float], *, device: str = "cpu") -> np.ndarray:
    """Run a loaded Genesis Go2 actor and return the 12-dim action."""
    import torch

    obs = _checked_array(observation, "observation", GO2_UNITREE_OBS_SIZE)
    with torch.no_grad():
        tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32).unsqueeze(0)
        action = model(tensor).detach().cpu().numpy().squeeze(0)
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (GO2_UNITREE_ACTION_SIZE,):
        raise RuntimeError(f"policy returned shape {action.shape}, expected (12,)")
    return action


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


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"articulation.{field_name} must be a mapping")
    return value
