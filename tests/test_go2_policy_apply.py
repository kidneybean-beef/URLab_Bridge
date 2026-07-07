from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_apply_script():
    script_path = Path(__file__).parents[1] / "scripts" / "run_go2_policy_apply.py"
    spec = importlib.util.spec_from_file_location("run_go2_policy_apply", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeArt:
    def __init__(self, root_z: float) -> None:
        self.root_pos_w = np.array([0.0, 0.0, root_z], dtype=np.float32)
        self.root_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def test_validate_target_pose_rejects_nonfinite_values():
    mod = _load_apply_script()

    assert mod.validate_target_pose({"FR_hip": 0.0, "FR_thigh": float("nan")}) is not None
    assert mod.validate_target_pose({"FR_hip": 0.0, "FR_thigh": 0.8}) is None


def test_safety_abort_reason_checks_base_height():
    mod = _load_apply_script()

    assert mod.safety_abort_reason(FakeArt(0.18), min_base_z=0.2) is not None
    assert mod.safety_abort_reason(FakeArt(0.31), min_base_z=0.2) is None


def test_parser_keeps_existing_gains_and_genesis_command_by_default():
    mod = _load_apply_script()

    args = mod.build_arg_parser().parse_args([])

    assert args.push_gains is False
    assert args.hold_default_only is False
    assert args.cmd_vx == pytest.approx(0.5)
    assert args.max_action_abs == pytest.approx(10.0)


def test_format_top_abs_names_largest_values():
    mod = _load_apply_script()

    text = mod._format_top_abs(
        "action",
        ["FR_hip", "FR_thigh", "FR_calf"],
        np.array([0.1, -2.0, 1.0]),
        count=2,
    )

    assert text == "action: FR_thigh=-2.0000, FR_calf=1.0000"
