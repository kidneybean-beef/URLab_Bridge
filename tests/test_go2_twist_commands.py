from __future__ import annotations

import numpy as np
import pytest


def test_fixed_command_source_returns_defensive_copy():
    from urlab_policy.go2.twist_commands import FixedCommandSource

    source = FixedCommandSource([0.3, -0.2, 0.1])

    first = source.poll()
    first[0] = 99.0

    assert source.poll().tolist() == pytest.approx([0.3, -0.2, 0.1])


def test_keyboard_command_source_maps_remapped_keys_to_body_twist():
    from urlab_policy.go2.twist_commands import (
        KeyboardCommandConfig,
        KeyboardCommandSource,
    )

    source = KeyboardCommandSource(
        config=KeyboardCommandConfig(
            step_vx=0.1,
            step_vy=0.2,
            step_yaw=0.3,
            max_vx=0.5,
            max_vy=0.5,
            max_yaw=1.0,
        )
    )

    source.handle_key("w")
    source.handle_key("q")
    source.handle_key("a")

    assert source.poll().tolist() == pytest.approx([0.1, 0.2, 0.3])

    source.handle_key("s")
    source.handle_key("e")
    source.handle_key("d")

    assert source.poll().tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_keyboard_command_source_clips_to_configured_limits():
    from urlab_policy.go2.twist_commands import (
        KeyboardCommandConfig,
        KeyboardCommandSource,
    )

    source = KeyboardCommandSource(
        config=KeyboardCommandConfig(
            step_vx=0.3,
            step_vy=0.4,
            step_yaw=0.7,
            max_vx=0.5,
            max_vy=0.5,
            max_yaw=1.0,
        )
    )

    for key in ["w", "w", "q", "q", "a", "a"]:
        source.handle_key(key)

    assert source.poll().tolist() == pytest.approx([0.5, 0.5, 1.0])


def test_keyboard_command_source_zero_and_quit_keys():
    from urlab_policy.go2.twist_commands import KeyboardCommandSource

    source = KeyboardCommandSource(initial_command=[0.3, 0.2, -0.1])

    source.handle_key(" ")
    assert source.poll().tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert source.quit_requested is False

    source.handle_key("\x1b")
    assert source.quit_requested is True


def test_keyboard_command_source_polls_reader_keys():
    from urlab_policy.go2.twist_commands import KeyboardCommandSource

    class Reader:
        def poll_keys(self):
            return ["w", "w", "d"]

    source = KeyboardCommandSource(reader=Reader())

    command = source.poll()

    assert isinstance(command, np.ndarray)
    assert command.tolist() == pytest.approx([0.2, 0.0, -0.2])


def test_urlab_twist_command_source_reads_articulation_twist():
    from urlab_policy.go2.twist_commands import URLabTwistCommandSource

    class Articulation:
        twist_linear = np.array([0.4, -0.2, 9.0], dtype=np.float64)
        twist_angular = np.array([1.0, 2.0, 0.7], dtype=np.float64)

    source = URLabTwistCommandSource(Articulation())

    command = source.poll()
    command[0] = 99.0

    assert source.poll().tolist() == pytest.approx([0.4, 0.2, -0.7])
    assert source.quit_requested is False


def test_urlab_twist_command_source_allows_explicit_axis_signs():
    from urlab_policy.go2.twist_commands import URLabTwistCommandSource

    class Articulation:
        twist_linear = np.array([0.4, -0.2, 9.0], dtype=np.float64)
        twist_angular = np.array([1.0, 2.0, 0.7], dtype=np.float64)

    source = URLabTwistCommandSource(Articulation(), axis_signs=(1.0, 1.0, 1.0))

    assert source.poll().tolist() == pytest.approx([0.4, -0.2, 0.7])
