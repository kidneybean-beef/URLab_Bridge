from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from urlab_policy.go2.unitree_policy import (
    GO2_UNITREE_DEFAULT_DOF_POS,
    GO2_UNITREE_JOINT_NAMES,
    action_to_target_pose,
    build_unitree_go2_observation,
    genesis_default_target_pose,
    projected_gravity_from_xyzw,
)


def test_genesis_go2_policy_constants_match_training_example():
    assert GO2_UNITREE_JOINT_NAMES == (
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    )
    assert np.allclose(
        GO2_UNITREE_DEFAULT_DOF_POS,
        [
            0.0, 0.8, -1.5,
            0.0, 0.8, -1.5,
            0.0, 1.0, -1.5,
            0.0, 1.0, -1.5,
        ],
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
    def __init__(self) -> None:
        self.joints = {
            name: FakeJoint(name, i, i)
            for i, name in enumerate(GO2_UNITREE_JOINT_NAMES)
        }
        self.actuators = {
            name.removesuffix("_joint"): FakeActuator(
                name.removesuffix("_joint"), name
            )
            for name in GO2_UNITREE_JOINT_NAMES
        }
        self.qpos_array = GO2_UNITREE_DEFAULT_DOF_POS + np.arange(12) * 0.01
        self.qvel_array = np.arange(12) * 0.1
        self.root_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        self.root_ang_vel_b = np.array([1.0, 2.0, 3.0])

    def resolve_joint(self, name: str) -> str | None:
        return name if name in self.joints else None

    def resolve_actuator(self, name: str) -> str | None:
        key = name.removesuffix("_joint")
        return key if key in self.actuators else None


def test_projected_gravity_from_xyzw_identity_points_down():
    gravity = projected_gravity_from_xyzw(np.array([0.0, 0.0, 0.0, 1.0]))

    assert np.allclose(gravity, [0.0, 0.0, -1.0])


def test_build_unitree_go2_observation_packs_45_values():
    art = FakeArticulation()
    last_action = np.linspace(-0.1, 0.1, 12)

    obs = build_unitree_go2_observation(
        art,
        command=np.array([0.5, -0.2, 0.4]),
        last_action=last_action,
    )

    assert obs.shape == (45,)
    assert np.allclose(obs[0:3], [0.25, 0.5, 0.75])
    assert np.allclose(obs[3:6], [0.0, 0.0, -1.0])
    assert np.allclose(obs[6:9], [1.0, -0.4, 0.1])
    assert np.allclose(obs[9:21], np.arange(12) * 0.01)
    assert np.allclose(obs[21:33], np.arange(12) * 0.1 * 0.05)
    assert np.allclose(obs[33:45], last_action)


def test_action_to_target_pose_maps_policy_order_to_actuators():
    art = FakeArticulation()
    action = np.ones(12)

    targets = action_to_target_pose(art, action)

    assert targets["FR_hip"] == pytest.approx(GO2_UNITREE_DEFAULT_DOF_POS[0] + 0.25)
    assert targets["FL_hip"] == pytest.approx(GO2_UNITREE_DEFAULT_DOF_POS[3] + 0.25)
    assert targets["RR_calf"] == pytest.approx(GO2_UNITREE_DEFAULT_DOF_POS[8] + 0.25)
    assert targets["RL_calf"] == pytest.approx(GO2_UNITREE_DEFAULT_DOF_POS[-1] + 0.25)
    assert len(targets) == 12


def test_genesis_default_target_pose_uses_zero_action():
    targets = genesis_default_target_pose(FakeArticulation())

    assert targets == pytest.approx({
        "FR_hip": 0.0,
        "FR_thigh": 0.8,
        "FR_calf": -1.5,
        "FL_hip": 0.0,
        "FL_thigh": 0.8,
        "FL_calf": -1.5,
        "RR_hip": 0.0,
        "RR_thigh": 1.0,
        "RR_calf": -1.5,
        "RL_hip": 0.0,
        "RL_thigh": 1.0,
        "RL_calf": -1.5,
    })


def test_genesis_default_target_pose_is_exported_from_go2_package():
    from urlab_policy.go2 import genesis_default_target_pose as exported

    assert exported is genesis_default_target_pose
