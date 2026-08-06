from __future__ import annotations

import pytest


class FakeClock:
    def __init__(self, now: float = 10.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def set_twist_control_state(self, articulation: str, **kwargs) -> dict:
        self.calls.append((articulation, dict(kwargs)))
        return {
            "op": "set_twist_control_state_ok",
            "articulation": articulation,
            "dash_active": bool(kwargs.get("dash_active", False)),
            "max_vx": 0.25,
            "max_vy": 0.2,
            "max_yaw": 0.4,
            "dash_max_vx": 2.0,
            "dash_max_vy": 1.0,
            "dash_max_yaw": 3.14,
        }


def test_twist_slew_limiter_uses_axis_rates_at_policy_tick() -> None:
    from urlab_bridge.control_server.commands import TwistSlewLimiter

    limiter = TwistSlewLimiter()

    assert limiter.update((1.0, 0.5, 1.57), dt=0.02) == pytest.approx(
        (0.04, 0.02, 0.0628)
    )


def test_twist_slew_limiter_brakes_immediately() -> None:
    from urlab_bridge.control_server.commands import TwistSlewLimiter

    limiter = TwistSlewLimiter(initial=(0.8, -0.4, 1.2))

    assert limiter.update((1.0, -0.5, 1.57), dt=0.02, brake=True) == (
        0.0,
        0.0,
        0.0,
    )
    assert limiter.current == (0.0, 0.0, 0.0)


def test_twist_slew_limiter_decelerates_before_reversing() -> None:
    from urlab_bridge.control_server.commands import (
        TwistSlewConfig,
        TwistSlewLimiter,
    )

    limiter = TwistSlewLimiter(
        TwistSlewConfig(
            accel_vx=1.0,
            accel_vy=1.0,
            accel_yaw=1.0,
            decel_vx=2.0,
            decel_vy=2.0,
            decel_yaw=2.0,
        ),
        initial=(0.1, 0.0, 0.0),
    )

    assert limiter.update((-1.0, 0.0, 0.0), dt=0.04)[0] == pytest.approx(0.02)
    assert limiter.update((-1.0, 0.0, 0.0), dt=0.04)[0] == pytest.approx(-0.03)


def test_command_hub_updates_only_selected_robot():
    from urlab_bridge.control_server.commands import CommandHub

    hub = CommandHub(["dog_a", "dog_b"])
    dog_a = hub.port("dog_a")
    dog_b = hub.port("dog_b")

    dog_a.apply_control({"keys": {"w": True, "shift": True}})

    assert dog_a.poll() == pytest.approx((2.0, 0.0, 0.0))
    assert dog_b.poll() == pytest.approx((0.0, 0.0, 0.0))
    assert dog_a.status()["active"] is True
    assert dog_b.status()["active"] is False


def test_robot_command_port_exposes_space_as_immediate_brake() -> None:
    from urlab_bridge.control_server.commands import CommandHub

    dog = CommandHub(["dog_a"]).port("dog_a")

    dog.apply_control({"keys": {"w": True}})
    assert dog.brake_requested is False

    dog.apply_control({"keys": {"space": True}})
    assert dog.brake_requested is True
    assert dog.poll() == (0.0, 0.0, 0.0)


def test_command_hub_release_zeros_one_robot_without_touching_other_robot():
    from urlab_bridge.control_server.commands import CommandHub

    hub = CommandHub(["dog_a", "dog_b"])
    dog_a = hub.port("dog_a")
    dog_b = hub.port("dog_b")

    dog_a.apply_control({"keys": {"w": True}})
    dog_b.apply_control({"keys": {"s": True}})
    dog_a.release()

    assert dog_a.poll() == pytest.approx((0.0, 0.0, 0.0))
    assert dog_b.poll() == pytest.approx((-1.0, 0.0, 0.0))
    assert dog_a.status()["active"] is False
    assert dog_b.status()["active"] is True


def test_command_hub_stales_only_the_robot_whose_command_is_old():
    from urlab_bridge.control_server.commands import CommandHub

    clock = FakeClock()
    hub = CommandHub(["dog_a", "dog_b"], stale_timeout_s=0.5, now_fn=clock)
    dog_a = hub.port("dog_a")
    dog_b = hub.port("dog_b")

    dog_a.apply_control({"keys": {"w": True}})
    clock.now += 0.3
    dog_b.apply_control({"keys": {"s": True}})
    clock.now += 0.3

    assert dog_a.stop_if_stale() is True
    assert dog_b.stop_if_stale() is False
    assert dog_a.poll() == pytest.approx((0.0, 0.0, 0.0))
    assert dog_b.poll() == pytest.approx((-1.0, 0.0, 0.0))
    assert dog_a.status()["stale"] is True
    assert dog_b.status()["last_command_age_s"] == pytest.approx(0.3)


def test_command_hub_reports_command_age_from_fake_clock():
    from urlab_bridge.control_server.commands import CommandHub

    clock = FakeClock()
    hub = CommandHub(["dog_a"], stale_timeout_s=1.0, now_fn=clock)
    dog_a = hub.port("dog_a")

    dog_a.apply_control({"keys": {"q": True}})
    clock.now += 0.25

    status = dog_a.status()

    assert status["last_command_age_s"] == pytest.approx(0.25)
    assert status["twist"] == pytest.approx([0.0, 0.5, 0.0])
    assert status["source"] == "web"


def test_command_hub_accepts_ros2_twist_for_selected_robot():
    from urlab_bridge.control_server.commands import CommandHub

    hub = CommandHub(["dog_a", "dog_b"])
    dog_a_ros = hub.port("dog_a", source="ros2")
    dog_b_web = hub.port("dog_b")

    dog_a_ros.apply_twist((0.2, -0.1, 0.35))

    assert dog_a_ros.poll() == pytest.approx((0.2, -0.1, 0.35))
    assert dog_b_web.poll() == pytest.approx((0.0, 0.0, 0.0))
    assert dog_a_ros.status()["source"] == "ros2"
    assert dog_a_ros.status()["active"] is True


def test_command_hub_latest_source_wins_between_web_and_ros2():
    from urlab_bridge.control_server.commands import CommandHub

    hub = CommandHub(["dog_a"])
    dog_web = hub.port("dog_a", source="web")
    dog_ros = hub.port("dog_a", source="ros2")

    dog_web.apply_control({"keys": {"w": True}})
    dog_ros.apply_twist((0.0, 0.25, -0.5))

    assert dog_web.poll() == pytest.approx((0.0, 0.25, -0.5))
    assert dog_web.status()["source"] == "ros2"

    dog_web.apply_control({"keys": {"s": True}})

    assert dog_ros.poll() == pytest.approx((-1.0, 0.0, 0.0))
    assert dog_ros.status()["source"] == "web"


def test_command_hub_stales_ros2_twist_to_zero():
    from urlab_bridge.control_server.commands import CommandHub

    clock = FakeClock()
    dog = CommandHub(["dog_a"], stale_timeout_s=0.5, now_fn=clock).port(
        "dog_a",
        source="ros2",
    )

    dog.apply_twist((0.2, 0.0, 0.0))
    clock.now += 0.6

    assert dog.stop_if_stale() is True
    assert dog.poll() == pytest.approx((0.0, 0.0, 0.0))
    assert dog.status()["stale"] is True
    assert dog.status()["source"] == "ros2"


def test_robot_command_port_syncs_dash_display_and_uses_runtime_limits():
    from urlab_bridge.control_server.commands import CommandHub

    runtime = FakeRuntime()
    hub = CommandHub(["dog_a"])
    dog_a = hub.port("dog_a")

    dog_a.apply_control({"keys": {"w": True, "shift": True}})

    assert dog_a.sync_runtime_ui(runtime, "dog_a") is True
    assert runtime.calls[-1] == (
        "dog_a",
        {
            "dash_active": True,
            "keys": {
                "w": True,
                "s": False,
                "q": False,
                "e": False,
                "a": False,
                "d": False,
                "space": False,
                "shift": True,
            },
        },
    )
    assert dog_a.poll() == pytest.approx((0.5, 0.0, 0.0))
