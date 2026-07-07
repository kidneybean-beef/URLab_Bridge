from __future__ import annotations

from dataclasses import dataclass

import pytest

from urlab_policy.go2.pose import (
    capture_actuated_joint_pose,
    pose_with_sine_offset,
)


@dataclass
class FakeActuator:
    name: str
    joint: str | None


@dataclass
class FakeJoint:
    name: str
    qpos_local_offset: int
    qpos_dim: int = 1


class FakeArticulation:
    def __init__(self) -> None:
        self.actuators = {
            "FR_hip": FakeActuator("FR_hip", "FR_hip_joint"),
            "FR_thigh": FakeActuator("FR_thigh", "FR_thigh_joint"),
            "FR_calf": FakeActuator("FR_calf", "FR_calf_joint"),
        }
        self.joints = {
            "floating_base": FakeJoint("floating_base", 0, qpos_dim=7),
            "FR_hip_joint": FakeJoint("FR_hip_joint", 7),
            "FR_thigh_joint": FakeJoint("FR_thigh_joint", 8),
            "FR_calf_joint": FakeJoint("FR_calf_joint", 9),
        }
        self.qpos_array = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.12, 0.93, -1.71]


def test_capture_actuated_joint_pose_uses_live_joint_qpos():
    art = FakeArticulation()

    pose = capture_actuated_joint_pose(art)

    assert pose == {
        "FR_hip": 0.12,
        "FR_thigh": 0.93,
        "FR_calf": -1.71,
    }


def test_capture_actuated_joint_pose_fails_loudly_for_missing_joint():
    art = FakeArticulation()
    art.actuators["bad"] = FakeActuator("bad", "missing_joint")

    with pytest.raises(KeyError, match="missing_joint"):
        capture_actuated_joint_pose(art)


def test_capture_actuated_joint_pose_rejects_actuator_without_joint():
    art = FakeArticulation()
    art.actuators["free_motor"] = FakeActuator("free_motor", None)

    with pytest.raises(ValueError, match="free_motor"):
        capture_actuated_joint_pose(art)


def test_pose_with_sine_offset_moves_one_actuator_without_mutating_base():
    base = {"FR_hip": 0.12, "FR_thigh": 0.93}

    pose = pose_with_sine_offset(
        base,
        actuator_name="FR_hip",
        amplitude=0.02,
        frequency_hz=1.0,
        elapsed_s=0.25,
    )

    assert pose == {"FR_hip": pytest.approx(0.14), "FR_thigh": 0.93}
    assert base == {"FR_hip": 0.12, "FR_thigh": 0.93}


def test_pose_with_sine_offset_rejects_unknown_actuator():
    with pytest.raises(KeyError, match="bad_actuator"):
        pose_with_sine_offset(
            {"FR_hip": 0.12},
            actuator_name="bad_actuator",
            amplitude=0.02,
            frequency_hz=1.0,
            elapsed_s=0.25,
        )
