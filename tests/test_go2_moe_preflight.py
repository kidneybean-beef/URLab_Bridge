from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_preflight_script():
    script_path = Path(__file__).parents[1] / "scripts" / "run_go2_moe_preflight.py"
    spec = importlib.util.spec_from_file_location("run_go2_moe_preflight", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_to_shadow_safe_preflight():
    mod = _load_preflight_script()

    args = mod.build_arg_parser().parse_args([])

    assert args.policy.endswith("policies/go2-rl-gym/moe_cts.pt")
    assert args.control_source == "ui"
    assert args.max_stand_error == 0.35
    assert args.cmd_vx == 0.5
    assert args.skip_policy is False


def test_select_articulation_requires_name_when_multiple_available():
    mod = _load_preflight_script()

    class Client:
        articulations = {"go2_a": object(), "go2_b": object()}

    try:
        mod._select_articulation(Client(), "")
    except SystemExit as exc:
        assert "multiple articulations" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
