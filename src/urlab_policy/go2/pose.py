from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def capture_actuated_joint_pose(articulation: Any) -> dict[str, float]:
    """Return actuator targets equal to the articulation's current joint pose.

    URLab actuators expose the joint they drive. For position-target PD control,
    a safe takeover target is the current qpos of each driven hinge/slide joint.
    Free-base joints are not actuated and are ignored naturally because they have
    no actuator entry.
    """
    actuators = _as_mapping(getattr(articulation, "actuators", None), "actuators")
    joints = _as_mapping(getattr(articulation, "joints", None), "joints")
    qpos = getattr(articulation, "qpos_array", None)
    if qpos is None:
        raise ValueError(
            "articulation has no qpos_array; call client.step() before capture"
        )

    pose: dict[str, float] = {}
    for actuator_name, actuator in actuators.items():
        joint_name = getattr(actuator, "joint", None)
        if not joint_name:
            raise ValueError(f"actuator {actuator_name!r} is not bound to a joint")
        if joint_name not in joints:
            raise KeyError(
                f"actuator {actuator_name!r} references missing joint {joint_name!r}"
            )

        joint = joints[joint_name]
        qpos_dim = int(getattr(joint, "qpos_dim", 1))
        if qpos_dim != 1:
            raise ValueError(
                f"actuator {actuator_name!r} joint {joint_name!r} "
                f"has qpos_dim={qpos_dim}; "
                "only scalar joints can be captured as actuator targets"
            )

        offset = int(getattr(joint, "qpos_local_offset"))
        pose[str(actuator_name)] = float(qpos[offset])

    return pose


def pose_with_sine_offset(
    base_pose: Mapping[str, float],
    *,
    actuator_name: str,
    amplitude: float,
    frequency_hz: float,
    elapsed_s: float,
) -> dict[str, float]:
    """Return a copy of ``base_pose`` with one smooth sinusoidal actuator offset."""
    if actuator_name not in base_pose:
        raise KeyError(f"actuator {actuator_name!r} is not in the captured pose")

    pose = {str(name): float(value) for name, value in base_pose.items()}
    phase = 2.0 * math.pi * float(frequency_hz) * float(elapsed_s)
    pose[actuator_name] += float(amplitude) * math.sin(phase)
    return pose


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"articulation.{field_name} must be a mapping")
    return value
