from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_ue_twist_script():
    script_path = Path(__file__).parents[1] / "scripts" / "run_go2_moe_ue_twist.py"
    spec = importlib.util.spec_from_file_location("run_go2_moe_ue_twist", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ue_twist_parser_uses_unitree_live_defaults():
    mod = _load_ue_twist_script()

    args = mod.build_arg_parser().parse_args([])

    assert args.freq == pytest.approx(50.0)
    assert args.duration == pytest.approx(0.0)
    assert args.max_action_abs == pytest.approx(100.0)
    assert args.max_target_step is None
    assert args.warmup_steps == 0
    assert args.push_gains is True
    assert args.torque_limit is None
    assert args.twist_vx_sign == pytest.approx(1.0)
    assert args.twist_vy_sign == pytest.approx(-1.0)
    assert args.twist_yaw_sign == pytest.approx(-1.0)
