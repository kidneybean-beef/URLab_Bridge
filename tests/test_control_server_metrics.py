from __future__ import annotations

import json

import pytest


def test_control_loop_metrics_records_timing_and_deadline_miss():
    from urlab_bridge.control_server.metrics import ControlLoopMetrics

    metrics = ControlLoopMetrics(freq_hz=50.0)

    metrics.record_tick(
        tick_duration_s=0.026,
        policy_duration_s=0.003,
        step_duration_s=0.012,
        active_robot_count=1,
        command_statuses={
            "dog_a": {
                "articulation": "dog_a",
                "active": True,
                "stale": False,
                "last_command_age_s": 0.04,
                "twist": [1.0, 0.0, 0.0],
            }
        },
    )

    snapshot = metrics.snapshot()

    assert snapshot["tick_count"] == 1
    assert snapshot["missed_deadlines"] == 1
    assert snapshot["last_tick_duration_s"] == pytest.approx(0.026)
    assert snapshot["last_policy_duration_s"] == pytest.approx(0.003)
    assert snapshot["last_step_duration_s"] == pytest.approx(0.012)
    assert snapshot["last_deadline_overrun_s"] == pytest.approx(0.006)
    assert snapshot["max_tick_duration_s"] == pytest.approx(0.026)
    assert snapshot["active_robot_count"] == 1
    assert snapshot["robots"]["dog_a"]["last_command_age_s"] == pytest.approx(0.04)


def test_control_loop_metrics_does_not_count_tick_inside_deadline():
    from urlab_bridge.control_server.metrics import ControlLoopMetrics

    metrics = ControlLoopMetrics(freq_hz=50.0)

    metrics.record_tick(
        tick_duration_s=0.019,
        policy_duration_s=0.002,
        step_duration_s=0.010,
        active_robot_count=2,
        command_statuses={},
    )

    snapshot = metrics.snapshot()

    assert snapshot["missed_deadlines"] == 0
    assert snapshot["last_deadline_overrun_s"] == pytest.approx(0.0)
    assert snapshot["active_robot_count"] == 2


def test_control_loop_metrics_snapshot_is_json_safe():
    from urlab_bridge.control_server.metrics import ControlLoopMetrics

    metrics = ControlLoopMetrics(freq_hz=20.0)
    metrics.record_tick(
        tick_duration_s=0.01,
        policy_duration_s=0.004,
        step_duration_s=0.003,
        active_robot_count=0,
        command_statuses={
            "dog_a": {
                "articulation": "dog_a",
                "source": "web",
                "active": False,
                "stale": False,
                "last_command_age_s": None,
                "twist": (0.0, 0.0, 0.0),
            }
        },
    )

    payload = metrics.snapshot()

    assert json.loads(json.dumps(payload))["robots"]["dog_a"]["twist"] == [
        0.0,
        0.0,
        0.0,
    ]
