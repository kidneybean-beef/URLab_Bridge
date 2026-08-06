# Go2 ROS2 And VLA Roadmap

## Goal

Make the current URLab Go2 walking example usable from ROS2 first, then extend the same interfaces toward VLA and world-action-model experiments without rewriting the working locomotion stack.

The immediate target is not native ROS2 inside the Unreal plugin. The immediate target is a ROS2 gateway in `URLab_Bridge` that feeds the existing central Go2 MoE policy runtime.

## Architecture Direction

Keep URLab's existing ZMQ/SHM control path as the fast simulation interface. Add ROS2 as a command, sensor, and integration layer around the central control server.

```text
ROS2 teleop / Nav2 / VLA node
        |
        | /<robot>/cmd_vel or task command
        v
URLab_Bridge Ros2Gateway
        |
        v
central command state
        |
        v
Go2 MoE locomotion policy
        |
        v
URLab client.step(...)
        |
        v
UE + MuJoCo
```

This keeps the current `moe_cts.pt` walking policy in charge of contact-rich leg motion while letting ROS2, web control, and future autonomy feed the same command path.

## Milestone 1: ROS2 Command Gateway

Add `Ros2Gateway` to `URLab_Bridge` and connect standard ROS2 velocity commands to the existing Go2 policy loop.

Required behavior:

- Subscribe to `/<robot>/cmd_vel` as `geometry_msgs/Twist`.
- Map `linear.x` to forward velocity, `linear.y` to lateral velocity, and `angular.z` to yaw rate.
- Write ROS2 commands into the same per-robot command state used by the web keyboard controller.
- Add stale timeout behavior so the robot brakes when ROS2 commands stop.
- Add command ownership or priority handling for web input vs ROS2 input.
- Keep one central `URLabClient`; do not create a second competing control loop.

Expected result:

```bash
ros2 topic pub /go2_go2_rl_gym_C_1/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
```

The Go2 walks forward using the same working MoE locomotion policy.

## Milestone 2: ROS2 State And Camera Export

Publish useful robot state and perception topics from the central server or a compatible broadcaster.

Target topics:

```text
/<robot>/joint_states
/<robot>/odom or /<robot>/base_state
/<robot>/imu
/<robot>/camera/<camera_name>/image_raw
/<robot>/camera/<camera_name>/camera_info
/tf
/tf_static
```

Missing work:

- Aggregate joint states into proper `sensor_msgs/JointState` messages.
- Publish base pose and velocity in a ROS-friendly frame convention.
- Publish camera metadata, frame ids, and image encodings.
- Decide depth, semantic, and instance image encodings.
- Define multi-robot naming and TF frame conventions.
- Choose QoS separately for state and camera topics.

## Milestone 3: ROS2 Teleop And Navigation Integration

Connect standard ROS2 tools to the command gateway.

Examples:

```text
teleop_twist_keyboard -> /<robot>/cmd_vel -> Go2 policy
Nav2 local planner -> /<robot>/cmd_vel -> Go2 policy
custom ROS2 node -> /<robot>/cmd_vel -> Go2 policy
```

Missing work:

- Validate whether URLab odometry is sufficient for Nav2.
- Add safety limits for incoming velocity commands.
- Add emergency stop and command source priority.
- Decide whether map and obstacle data come from depth, segmentation, imported scene geometry, or external ROS2 mapping nodes.

## Milestone 4: Dataset And Recording Path For VLA/WAM

Before running a direct-action model, collect synchronized data from the current stable walking stack.

Record:

```text
RGB/depth/semantic/instance camera frames
base state
joint states
cmd_vel
policy actions
task labels or language instructions
success/failure metadata
```

Missing work:

- Define an episode schema.
- Align timestamps across cameras, state, and actions.
- Decide ROS bag vs URLab recording vs both.
- Export data in a format usable by VLA/WAM training code.
- Add reset/task scripts for repeatable data collection.

## Milestone 5: High-Level VLA/WAM Control

Use a VLA or world-action model as a high-level decision maker first.

```text
VLA/WAM observes cameras and instruction
        |
        v
outputs waypoint, local goal, or cmd_vel
        |
        v
Go2 MoE locomotion policy handles leg motion
```

This is the safest first VLA path because the model does not need to solve balance, foot timing, and contact dynamics directly.

Missing work:

- VLA input adapter from ROS2 camera/state topics.
- Prompt and task interface.
- Output adapter to `/<robot>/cmd_vel` or a task/action topic.
- Runtime monitor for unsafe or stale commands.
- Evaluation tasks in URLab scenes.

## Milestone 6: Direct-Action VLA/WAM Mode

Add this only when there is a Go2-specific model trained for URLab's action interface.

Possible direct-action path:

```text
VLA/WAM
        |
        v
12 joint target action chunks
        |
        v
URLab PD / MuJoCo actuators
```

This can replace `moe_cts.pt`, but only if the model matches:

- Go2 morphology.
- Joint order.
- Action scale.
- Observation normalization.
- Control frequency.
- PD gains and actuator model.
- Contact-rich locomotion dynamics.

Missing work:

- Direct-action policy runner.
- Action chunk smoothing.
- Safety filter and fall detection.
- Recovery or standup fallback.
- Model/action schema registry.
- Sim-to-real validation plan.

## Milestone 7: Real Go2 Alignment

Keep the ROS2-facing interface stable so the same high-level nodes can target simulation or hardware.

```text
same VLA / teleop / Nav2 node
same /cmd_vel
same camera/state topic concepts
backend changes:
  URLab simulation OR real Unitree driver
```

Missing work:

- Unitree ROS2 driver compatibility check.
- Frame convention alignment.
- Camera calibration.
- Latency measurement.
- Hardware safety limits and emergency stop.

## Near-Term Recommendation

Implement Milestone 1 first:

```text
ROS2 /<robot>/cmd_vel -> URLab_Bridge -> existing Go2 MoE policy
```

This gives a working ROS2-controlled Go2 walking demo without touching the UE plugin, policy math, camera system, or direct-action VLA path.
