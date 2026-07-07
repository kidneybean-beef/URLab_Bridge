from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from urlab_policy.go2.unitree_rl_gym_moe import (
    GO2_MOE_ACTION_SCALE,
    GO2_MOE_ACTION_SIZE,
    GO2_MOE_DEFAULT_DOF_POS,
    GO2_MOE_JOINT_NAMES,
    GO2_MOE_OBS_SIZE,
    action_to_moe_target_pose,
    build_go2_moe_observation,
    infer_go2_moe_action,
    reset_go2_moe_history,
)


def test_moe_constants_match_unitree_rl_gym_go2_contract():
    assert GO2_MOE_JOINT_NAMES == (
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    )
    assert GO2_MOE_OBS_SIZE == 45
    assert GO2_MOE_ACTION_SIZE == 12
    assert GO2_MOE_ACTION_SCALE == pytest.approx(0.25)
    assert np.allclose(
        GO2_MOE_DEFAULT_DOF_POS,
        [
            0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1, 1.0, -1.5,
            -0.1, 1.0, -1.5,
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
            for i, name in enumerate(GO2_MOE_JOINT_NAMES)
        }
        self.actuators = {
            name.removesuffix("_joint"): FakeActuator(
                name.removesuffix("_joint"), name
            )
            for name in GO2_MOE_JOINT_NAMES
        }
        self.qpos_array = (
            GO2_MOE_DEFAULT_DOF_POS + np.arange(12, dtype=np.float32) * 0.02
        )
        self.qvel_array = np.arange(12, dtype=np.float32) * -0.1
        self.root_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.root_ang_vel_b = np.array([1.0, -2.0, 3.0], dtype=np.float32)

    def resolve_joint(self, name: str) -> str | None:
        return name if name in self.joints else None

    def resolve_actuator(self, name: str) -> str | None:
        key = name.removesuffix("_joint")
        return key if key in self.actuators else None


def test_build_go2_moe_observation_packs_45_values_without_phase_terms():
    art = FakeArticulation()
    last_action = np.linspace(-0.2, 0.2, 12, dtype=np.float32)

    obs = build_go2_moe_observation(
        art,
        command=np.array([0.5, -0.25, 0.4], dtype=np.float32),
        last_action=last_action,
    )

    assert obs.shape == (45,)
    assert np.allclose(obs[0:3], [0.25, -0.5, 0.75])
    assert np.allclose(obs[3:6], [0.0, 0.0, -1.0])
    assert np.allclose(obs[6:9], [1.0, -0.5, 0.1])
    assert np.allclose(obs[9:21], np.arange(12, dtype=np.float32) * 0.02)
    assert np.allclose(obs[21:33], np.arange(12, dtype=np.float32) * -0.1 * 0.05)
    assert np.allclose(obs[33:45], last_action)


def test_action_to_moe_target_pose_uses_unitree_rl_gym_joint_order():
    art = FakeArticulation()

    targets = action_to_moe_target_pose(art, np.ones(12, dtype=np.float32))

    assert targets["FL_hip"] == pytest.approx(0.1 + 0.25)
    assert targets["FR_hip"] == pytest.approx(-0.1 + 0.25)
    assert targets["RL_thigh"] == pytest.approx(1.0 + 0.25)
    assert targets["RR_calf"] == pytest.approx(-1.5 + 0.25)
    assert list(targets) == [
        name.removesuffix("_joint") for name in GO2_MOE_JOINT_NAMES
    ]


class TuplePolicy(torch.nn.Module):
    def forward(self, obs: torch.Tensor):
        action = torch.arange(12, dtype=torch.float32).unsqueeze(0)
        weights = torch.ones(1, 8, dtype=torch.float32) / 8.0
        latent = torch.zeros(1, 32, dtype=torch.float32)
        return action, (weights, latent)


def test_infer_go2_moe_action_unwraps_torchscript_tuple_shape():
    action, diagnostics = infer_go2_moe_action(
        TuplePolicy(),
        np.zeros(45, dtype=np.float32),
    )

    assert np.allclose(action, np.arange(12, dtype=np.float32))
    assert diagnostics.expert_weights.shape == (8,)
    assert diagnostics.latent.shape == (32,)


def test_reset_go2_moe_history_primes_all_history_frames():
    class FakePolicy:
        history_length = 5
        history = torch.zeros(1, 5, 45, dtype=torch.float32)

    policy = FakePolicy()
    obs = np.linspace(-1.0, 1.0, 45, dtype=np.float32)

    reset_go2_moe_history(policy, obs)

    assert tuple(policy.history.shape) == (1, 5, 45)
    assert torch.allclose(policy.history[0, 0], torch.from_numpy(obs))
    assert torch.allclose(policy.history[0, -1], torch.from_numpy(obs))
