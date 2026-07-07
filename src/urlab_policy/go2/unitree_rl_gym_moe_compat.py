from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .unitree_rl_gym_moe import (
    GO2_MOE_DEFAULT_DOF_POS,
    GO2_MOE_JOINT_NAMES,
)


@dataclass(frozen=True)
class Go2MoeJointCheck:
    policy_joint_name: str
    resolved_joint_name: str | None
    resolved_actuator_name: str | None
    qpos: float | None
    qvel: float | None
    default_qpos: float
    qpos_error: float | None
    ok: bool
    issue: str | None = None


@dataclass(frozen=True)
class Go2MoeCompatibilityReport:
    has_free_base: bool
    base_z: float | None
    max_stand_error: float
    joint_checks: tuple[Go2MoeJointCheck, ...]
    missing_joints: tuple[str, ...]
    missing_actuators: tuple[str, ...]
    max_abs_qpos_error: float
    rms_qpos_error: float

    @property
    def model_ok(self) -> bool:
        return self.has_free_base and not self.missing_joints and not self.missing_actuators

    @property
    def stand_ready(self) -> bool:
        return self.max_abs_qpos_error <= self.max_stand_error

    @property
    def ok(self) -> bool:
        return self.model_ok and self.stand_ready

    def top_pose_errors(self, count: int = 4) -> tuple[Go2MoeJointCheck, ...]:
        valid = [check for check in self.joint_checks if check.qpos_error is not None]
        valid.sort(key=lambda check: abs(float(check.qpos_error)), reverse=True)
        return tuple(valid[:count])

    def summary(self) -> str:
        parts: list[str] = []
        if not self.has_free_base:
            parts.append("missing free base")
        if self.missing_joints:
            parts.append(f"missing joints: {', '.join(self.missing_joints)}")
        if self.missing_actuators:
            parts.append(f"missing actuators: {', '.join(self.missing_actuators)}")
        if not self.stand_ready:
            parts.append(
                "stand pose error "
                f"{self.max_abs_qpos_error:.4f} > {self.max_stand_error:.4f}"
            )
        if not parts:
            return "Go2 MoE preflight ok"
        return "; ".join(parts)


def build_go2_moe_compatibility_report(
    art: Any,
    *,
    joint_names: Sequence[str] = GO2_MOE_JOINT_NAMES,
    default_dof_pos: Sequence[float] = GO2_MOE_DEFAULT_DOF_POS,
    max_stand_error: float = 0.35,
) -> Go2MoeCompatibilityReport:
    """Inspect a URLab articulation against the Unitree RL Gym Go2 contract."""
    default_arr = np.asarray(default_dof_pos, dtype=np.float32)
    if default_arr.shape != (len(joint_names),):
        raise ValueError(
            f"default_dof_pos must have shape ({len(joint_names)},), "
            f"got {default_arr.shape}"
        )

    joints = _as_mapping(getattr(art, "joints", None), "joints")
    qpos_src = np.asarray(getattr(art, "qpos_array"), dtype=np.float32)
    qvel_src = np.asarray(getattr(art, "qvel_array"), dtype=np.float32)
    checks: list[Go2MoeJointCheck] = []
    missing_joints: list[str] = []
    missing_actuators: list[str] = []
    qpos_errors: list[float] = []

    for i, joint_name in enumerate(joint_names):
        default_qpos = _clean_float(default_arr[i])
        resolved_joint = _resolve_joint(art, joint_name)
        resolved_actuator = _resolve_actuator(art, joint_name)
        actuator_name = joint_name.removesuffix("_joint")
        issue: str | None = None
        qpos: float | None = None
        qvel: float | None = None
        qpos_error: float | None = None

        if resolved_joint is None or resolved_joint not in joints:
            missing_joints.append(joint_name)
            issue = "missing joint"
        else:
            joint = joints[resolved_joint]
            if int(getattr(joint, "qpos_dim", 1)) != 1:
                missing_joints.append(joint_name)
                issue = "joint is not scalar"
            else:
                qpos = _read_indexed_value(
                    qpos_src,
                    int(getattr(joint, "qpos_local_offset")),
                    f"{joint_name}.qpos",
                )
                qvel = _read_indexed_value(
                    qvel_src,
                    int(getattr(joint, "qvel_local_offset")),
                    f"{joint_name}.qvel",
                )
                qpos_error = _clean_float(float(qpos) - default_qpos)
                qpos_errors.append(float(qpos_error))

        if resolved_actuator is None:
            missing_actuators.append(actuator_name)
            issue = "missing actuator" if issue is None else f"{issue}; missing actuator"

        checks.append(
            Go2MoeJointCheck(
                policy_joint_name=joint_name,
                resolved_joint_name=resolved_joint,
                resolved_actuator_name=resolved_actuator,
                qpos=qpos,
                qvel=qvel,
                default_qpos=default_qpos,
                qpos_error=qpos_error,
                ok=issue is None,
                issue=issue,
            )
        )

    errors = np.asarray(qpos_errors, dtype=np.float32)
    max_abs_qpos_error = (
        _clean_float(np.max(np.abs(errors))) if errors.size else float("inf")
    )
    rms_qpos_error = (
        _clean_float(np.sqrt(np.mean(np.square(errors)))) if errors.size else float("inf")
    )

    return Go2MoeCompatibilityReport(
        has_free_base=bool(getattr(art, "has_free_base", False)),
        base_z=_base_z(art),
        max_stand_error=_clean_float(max_stand_error),
        joint_checks=tuple(checks),
        missing_joints=tuple(missing_joints),
        missing_actuators=tuple(missing_actuators),
        max_abs_qpos_error=max_abs_qpos_error,
        rms_qpos_error=rms_qpos_error,
    )


def format_go2_moe_report(report: Go2MoeCompatibilityReport) -> str:
    lines = [
        f"model_ok={report.model_ok} stand_ready={report.stand_ready} ok={report.ok}",
        f"summary: {report.summary()}",
        (
            "pose_error: "
            f"max={report.max_abs_qpos_error:.4f} "
            f"rms={report.rms_qpos_error:.4f} "
            f"threshold={report.max_stand_error:.4f}"
        ),
    ]
    if report.base_z is not None:
        lines.append(f"base_z={report.base_z:.4f}")
    if report.missing_joints:
        lines.append(f"missing_joints={list(report.missing_joints)}")
    if report.missing_actuators:
        lines.append(f"missing_actuators={list(report.missing_actuators)}")

    top = report.top_pose_errors(4)
    if top:
        formatted = ", ".join(
            f"{check.policy_joint_name}={check.qpos_error:.4f}" for check in top
        )
        lines.append(f"top_pose_errors: {formatted}")
    return "\n".join(lines)


def _resolve_joint(art: Any, joint_name: str) -> str | None:
    resolver = getattr(art, "resolve_joint", None)
    if callable(resolver):
        resolved = resolver(joint_name)
        if resolved is not None:
            return resolved
    joints = getattr(art, "joints", {})
    return joint_name if isinstance(joints, Mapping) and joint_name in joints else None


def _resolve_actuator(art: Any, joint_name: str) -> str | None:
    resolver = getattr(art, "resolve_actuator", None)
    if callable(resolver):
        resolved = resolver(joint_name)
        if resolved is not None:
            return resolved
    actuator_name = joint_name.removesuffix("_joint")
    actuators = getattr(art, "actuators", {})
    return (
        actuator_name
        if isinstance(actuators, Mapping) and actuator_name in actuators
        else None
    )


def _read_indexed_value(values: np.ndarray, index: int, label: str) -> float:
    if index < 0 or index >= values.size:
        raise ValueError(f"{label} offset {index} outside array of size {values.size}")
    return _clean_float(values[index])


def _base_z(art: Any) -> float | None:
    root_pos = getattr(art, "root_pos_w", None)
    if root_pos is None:
        return None
    arr = np.asarray(root_pos, dtype=np.float32)
    if arr.shape != (3,):
        return None
    return _clean_float(arr[2])


def _clean_float(value: float) -> float:
    return float(np.round(np.float32(value), 6))


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"articulation.{field_name} must be a mapping")
    return value
