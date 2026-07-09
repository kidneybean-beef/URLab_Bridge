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


def test_robot_command_port_syncs_dash_display_and_uses_runtime_limits():
    from urlab_bridge.control_server.commands import CommandHub

    runtime = FakeRuntime()
    hub = CommandHub(["dog_a"])
    dog_a = hub.port("dog_a")

    dog_a.apply_control({"keys": {"w": True, "shift": True}})

    assert dog_a.sync_runtime_ui(runtime, "dog_a") is True
    assert runtime.calls[-1] == ("dog_a", {"dash_active": True})
    assert dog_a.poll() == pytest.approx((0.5, 0.0, 0.0))
