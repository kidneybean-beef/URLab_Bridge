from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_keyboard_script():
    script_path = Path(__file__).parents[1] / "scripts" / "run_go2_moe_keyboard.py"
    spec = importlib.util.spec_from_file_location("run_go2_moe_keyboard", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keyboard_parser_defaults_start_stationary_with_unitree_limits():
    mod = _load_keyboard_script()

    args = mod.build_arg_parser().parse_args([])

    assert args.initial_cmd_vx == pytest.approx(0.0)
    assert args.initial_cmd_vy == pytest.approx(0.0)
    assert args.initial_cmd_yaw == pytest.approx(0.0)
    assert args.max_vx == pytest.approx(0.8)
    assert args.max_vy == pytest.approx(0.6)
    assert args.max_yaw == pytest.approx(1.0)
    assert args.step_vx == pytest.approx(0.1)
    assert args.step_vy == pytest.approx(0.1)
    assert args.step_yaw == pytest.approx(0.2)
    assert args.duration == pytest.approx(0.0)
    assert args.push_gains is True
