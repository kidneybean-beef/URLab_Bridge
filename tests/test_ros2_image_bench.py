from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script():
    script_path = Path(__file__).parents[1] / "scripts" / "ros2_image_bench.py"
    spec = importlib.util.spec_from_file_location("ros2_image_bench", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_to_sensor_data_qos() -> None:
    mod = _load_script()

    args = mod.build_arg_parser().parse_args(["/dog/camera/front_rgb/image_raw"])

    assert args.topic == "/dog/camera/front_rgb/image_raw"
    assert args.qos_reliability == "best_effort"
    assert args.qos_history == "keep_last"
    assert args.qos_depth == 1
    assert args.report_interval_s == pytest.approx(1.0)
    assert args.message_type == "auto"


def test_parser_can_select_compressed_image_messages() -> None:
    mod = _load_script()

    args = mod.build_arg_parser().parse_args([
        "/dog/camera/front_rgb/image_raw/compressed",
        "--message-type",
        "compressed_image",
    ])

    assert args.message_type == "compressed_image"


def test_auto_message_type_uses_compressed_suffix() -> None:
    mod = _load_script()

    assert mod._resolve_message_type("auto", "/dog/camera/image_raw") == "image"
    assert mod._resolve_message_type("auto", "/dog/camera/image_raw/compressed") == "compressed_image"


def test_image_stats_reports_hz_and_bandwidth_from_same_callbacks() -> None:
    mod = _load_script()
    stats = mod.ImageTrafficStats(started_at=10.0)

    stats.record(now=10.00, byte_count=100)
    stats.record(now=10.25, byte_count=100)
    stats.record(now=10.50, byte_count=100)

    report = stats.report(now=10.50)

    assert report.messages == 3
    assert report.bytes_total == 300
    assert report.hz == pytest.approx(6.0)
    assert report.mb_s == pytest.approx(300.0 / (1024.0 * 1024.0) / 0.5)
    assert report.mean_message_mb == pytest.approx(100.0 / (1024.0 * 1024.0))
    assert report.min_period_s == pytest.approx(0.25)
    assert report.max_period_s == pytest.approx(0.25)


def test_image_stats_reset_starts_a_new_window() -> None:
    mod = _load_script()
    stats = mod.ImageTrafficStats(started_at=10.0)

    stats.record(now=10.0, byte_count=100)
    stats.record(now=10.1, byte_count=100)
    stats.reset(now=11.0)
    stats.record(now=11.2, byte_count=50)

    report = stats.report(now=11.5)

    assert report.messages == 1
    assert report.bytes_total == 50
    assert report.hz == pytest.approx(2.0)
    assert report.min_period_s is None
