from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from urlab_policy.go2.unitree_rl_gym_moe import (
    GO2_MOE_DEFAULT_DOF_POS,
    GO2_MOE_JOINT_NAMES,
)
from urlab_policy.go2.unitree_rl_gym_moe_compat import (
    build_go2_moe_compatibility_report,
)


@dataclass
class FakeJoint:
    name: str
    qpos_local_offset: int
    qvel_local_offset: int
    qpos_dim: int = 1


@dataclass
class FakeActuator:
    name: str
    joint: str


class FakeArticulation:
    def __init__(
        self,
        *,
        has_free_base: bool = True,
        drop_joint: str | None = None,
        drop_actuator: str | None = None,
        qpos_offset: float = 0.0,
    ) -> None:
        self.has_free_base = has_free_base
        self.joints = {
            name: FakeJoint(name, i, i)
            for i, name in enumerate(GO2_MOE_JOINT_NAMES)
            if name != drop_joint
        }
        self.actuators = {
            name.removesuffix("_joint"): FakeActuator(
                name.removesuffix("_joint"), name
            )
            for name in GO2_MOE_JOINT_NAMES
            if name.removesuffix("_joint") != drop_actuator
        }
        self.qpos_array = GO2_MOE_DEFAULT_DOF_POS + qpos_offset
        self.qvel_array = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
        self.root_pos_w = np.array([0.0, 0.0, 0.31], dtype=np.float32)

    def resolve_joint(self, name: str) -> str | None:
        return name if name in self.joints else None

    def resolve_actuator(self, name: str) -> str | None:
        key = name.removesuffix("_joint")
        return key if key in self.actuators else None


def test_compatibility_report_accepts_matching_free_base_articulation():
    report = build_go2_moe_compatibility_report(
        FakeArticulation(qpos_offset=0.02),
        max_stand_error=0.05,
    )

    assert report.model_ok is True
    assert report.stand_ready is True
    assert report.ok is True
    assert report.missing_joints == ()
    assert report.missing_actuators == ()
    assert report.max_abs_qpos_error == np.float32(0.02)
    assert report.base_z == np.float32(0.31)
    assert report.top_pose_errors(2)[0].policy_joint_name == "FL_hip_joint"


def test_compatibility_report_flags_missing_actuator_and_fixed_base():
    report = build_go2_moe_compatibility_report(
        FakeArticulation(
            has_free_base=False,
            drop_actuator="RR_calf",
        ),
    )

    assert report.model_ok is False
    assert report.ok is False
    assert report.has_free_base is False
    assert report.missing_actuators == ("RR_calf",)
    assert "free base" in report.summary()
    assert "RR_calf" in report.summary()


def test_compatibility_report_marks_stand_not_ready_without_model_failure():
    report = build_go2_moe_compatibility_report(
        FakeArticulation(qpos_offset=0.2),
        max_stand_error=0.05,
    )

    assert report.model_ok is True
    assert report.stand_ready is False
    assert report.ok is False
    assert report.max_abs_qpos_error == np.float32(0.2)
    assert "stand pose error" in report.summary()
