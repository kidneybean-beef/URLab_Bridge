from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_apply_script():
    script_path = Path(__file__).parents[1] / "scripts" / "run_go2_moe_apply.py"
    spec = importlib.util.spec_from_file_location("run_go2_moe_apply", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_match_unitree_rl_gym_play_parameters():
    mod = _load_apply_script()

    args = mod.build_arg_parser().parse_args([])

    assert args.policy.endswith("policies/go2-rl-gym/moe_cts.pt")
    assert args.duration == pytest.approx(10.0)
    assert args.freq == pytest.approx(50.0)
    assert args.cmd_vx == pytest.approx(0.5)
    assert args.cmd_vy == pytest.approx(0.0)
    assert args.cmd_yaw == pytest.approx(0.0)
    assert args.allow_stand_mismatch is True
    assert args.strict_stand is False
    assert args.max_action_abs == pytest.approx(100.0)
    assert args.action_limit_mode == "clip"
    assert args.max_target_step is None
    assert args.warmup_steps == 0
    assert args.push_gains is True
    assert args.kp == pytest.approx(20.0)
    assert args.kv == pytest.approx(0.5)
    assert args.torque_limit is None
    assert args.raw_policy is False
    assert args.leave_zmq is False


def test_raw_policy_flag_disables_manual_policy_limits():
    mod = _load_apply_script()

    args = mod.build_arg_parser().parse_args(["--raw-policy"])
    mode = mod.resolve_limit_mode(args)

    assert mode.raw_policy is True
    assert mode.action_limit_mode == "none"
    assert mode.max_target_step is None


def test_action_abort_reason_rejects_nonfinite_and_large_actions():
    mod = _load_apply_script()

    assert mod.action_abort_reason([0.0] * 12, max_abs=4.0) is None
    assert "non-finite" in mod.action_abort_reason([float("nan")] + [0.0] * 11, max_abs=4.0)
    assert "exceeds" in mod.action_abort_reason([4.1] + [0.0] * 11, max_abs=4.0)
    assert "shape" in mod.action_abort_reason([0.0] * 11, max_abs=4.0)


def test_validate_target_pose_rejects_empty_and_nonfinite_values():
    mod = _load_apply_script()

    assert "empty" in mod.validate_target_pose({})
    assert "non-finite" in mod.validate_target_pose({"FR_hip": float("inf")})
    assert mod.validate_target_pose({"FR_hip": 0.0}) is None


def test_rate_limit_target_pose_caps_each_actuator_step():
    mod = _load_apply_script()

    limited = mod.rate_limit_target_pose(
        {"a": 0.0, "b": 1.0},
        {"a": 0.3, "b": 0.97},
        max_step=0.05,
    )

    assert limited == {"a": pytest.approx(0.05), "b": pytest.approx(0.97)}
    assert mod.target_delta_abs_max({"a": 0.0}, {"a": 0.05}) == pytest.approx(0.05)


def test_clip_action_caps_values_and_reports_when_clipped():
    mod = _load_apply_script()

    action, clipped = mod.clip_action([5.0, -4.5] + [0.0] * 10, max_abs=4.0)

    assert clipped is True
    assert action[0] == pytest.approx(4.0)
    assert action[1] == pytest.approx(-4.0)


def test_apply_action_limit_none_leaves_action_unchanged():
    mod = _load_apply_script()

    action, clipped = mod.apply_action_limit(
        [5.0, -4.5] + [0.0] * 10,
        max_abs=4.0,
        mode="none",
    )

    assert clipped is False
    assert action[0] == pytest.approx(5.0)
    assert action[1] == pytest.approx(-4.5)


def test_maybe_rate_limit_target_pose_none_returns_desired_target():
    mod = _load_apply_script()

    desired = {"a": 0.3, "b": 0.97}

    assert mod.maybe_rate_limit_target_pose(
        {"a": 0.0, "b": 1.0},
        desired,
        max_step=None,
    ) == desired


def test_unitree_command_deadzone_warning_matches_training_threshold():
    mod = _load_apply_script()

    assert "zeroed" in mod.command_deadzone_warning([0.2, 0.0, 0.0])
    assert mod.command_deadzone_warning([0.5, 0.0, 0.0]) is None


def test_default_torque_limits_match_unitree_go2_urdf_efforts():
    mod = _load_apply_script()

    limits = mod.resolve_torque_limits(None)

    assert limits.shape == (12,)
    assert limits[:3].tolist() == pytest.approx([23.7, 23.7, 35.55])
    assert limits[3:6].tolist() == pytest.approx([23.7, 23.7, 35.55])


def test_explicit_torque_limit_overrides_unitree_urdf_efforts():
    mod = _load_apply_script()

    limits = mod.resolve_torque_limits(45.0)

    assert limits.tolist() == pytest.approx([45.0] * 12)


def test_target_pose_to_action_returns_policy_order_feedback():
    mod = _load_apply_script()

    target_pose = {
        "FL_hip": 0.15,
        "FL_thigh": 0.75,
        "FL_calf": -1.5,
        "FR_hip": -0.1,
        "FR_thigh": 0.8,
        "FR_calf": -1.25,
        "RL_hip": 0.1,
        "RL_thigh": 1.0,
        "RL_calf": -1.5,
        "RR_hip": -0.1,
        "RR_thigh": 1.0,
        "RR_calf": -1.5,
    }

    action = mod.target_pose_to_action(target_pose)

    assert action.shape == (12,)
    assert action[0] == pytest.approx(0.2)
    assert action[1] == pytest.approx(-0.2)
    assert action[5] == pytest.approx(1.0)


def test_select_articulation_requires_name_when_multiple_available():
    mod = _load_apply_script()

    class Client:
        articulations = {"go2_a": object(), "go2_b": object()}

    try:
        mod._select_articulation(Client(), "")
    except SystemExit as exc:
        assert "multiple articulations" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
