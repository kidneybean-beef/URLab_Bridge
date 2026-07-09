# Central Robot-Control Server Roadmap

## Purpose

This document defines the long-term direction for URLab web-based multi-robot control. It is intentionally a roadmap, not a concrete implementation plan. Each milestone should later be expanded into its own detailed implementation plan when it is ready to build.

The goal is to move from ad-hoc scripts toward a central robot-control server that is convenient for users, respects URLab's current single active UE RPC session, and scales toward multi-robot locomotion, navigation, perception, and task-level autonomy.

## Recommended Architecture

Use one central Python process that owns the active URLab session and coordinates all robot control.

```text
URLabControlServer
  SessionManager
    owns one URLabClient / UE session

  RobotRegistry
    tracks spawned robots, articulation names, control mode, health

  CommandHub
    stores per-robot desired commands, source ids, command age, brake state
    gives all gateways one source-agnostic write path

  ControlLoop
    runs low-level locomotion at 50 Hz
    reads desired commands from CommandHub
    batches policy inference by policy type when possible
    sends one shared client.step(... controlled articulations ...)

  WebGateway
    serves browser UI
    maps users/pages to robot controllers
    writes keyboard/buttons/joystick commands into CommandHub

  Ros2Gateway
    subscribes to ROS 2 command topics and writes commands into CommandHub
    publishes URLab robot state, sensors, and cameras as ROS 2 topics

  PolicyRuntime
    keeps one controller state per robot
    shares policy models by policy file/type when possible
    supports batched inference for same-policy robots

  HigherLevelRuntime
    runs navigation at lower rate
    runs perception asynchronously
    runs task planning/event logic outside the low-level control loop
```

## Design Rationale

URLab currently has one active UE RPC session. Running one independent policy script per robot causes the newest Python client to expire older sessions. A central server avoids `session_expired` by keeping one active `URLabClient` and routing all robot work through that session.

The central server is also better for performance. It avoids duplicated state reads, duplicated RPC calls, duplicated policy loads, and competing control loops. It also allows later batching: robots using the same policy can share one loaded model and eventually run inference as a batch instead of many independent Python calls.

The command path should be source-agnostic early. Web control, ROS 2,
joystick input, navigation, and task logic should all write desired robot
commands into the same per-robot command state instead of coupling the
low-level control loop to a specific UI or middleware. This keeps the current
web work useful when ROS 2 and navigation are added later.

From the current Go2 MoE policy benchmark, the locomotion model is small:

```text
policies/go2-rl-gym/moe_cts.pt
size: about 4.97 MB
CPU inference: about 1.24 ms per inference
```

For a few Go2 dogs, low-level locomotion policy cost is not expected to dominate. UE physics/rendering, state extraction, RPC stepping, cameras, and perception will likely become more important bottlenecks. Navigation and task-level logic must therefore be rate-separated from the low-level locomotion loop.

## Performance Principles

Use explicit rate separation:

```text
50 Hz: low-level locomotion policy and UE step
20-30 Hz: web input, keyboard, virtual buttons, joystick
5-30 Hz: camera updates and perception, depending on cost
1-10 Hz: navigation and waypoint planning
event-based or <1 Hz: task planning, semantic reasoning, LLM/VLM calls
```

Do not let web requests, camera frames, navigation, perception, or task planning block the 50 Hz locomotion control loop.

Prefer batching when many robots use the same model:

```text
obs: [num_robots, obs_dim]
action: [num_robots, action_dim]
```

GPU should be treated as optional at first. For small single-sample locomotion models, GPU overhead may outweigh benefits. GPU becomes more useful for many robots, larger policies, batched inference, perception models, and navigation/perception pipelines whose data already lives on GPU.

## Milestone Roadmap

### Milestone 1: Server Skeleton

Goal: replace the current multi-web script shape with reusable server modules while preserving current behavior.

Scope:
- One URLab session.
- Multiple existing Go2 articulations.
- One web controller per robot.
- Same keyboard/button behavior as the current web controller.
- No spawning, camera feedback, navigation, joystick, leases, or perception.

Expected later implementation plan:
- Extract a `SessionManager`.
- Extract a `RobotRegistry`.
- Extract a basic `ControlLoop`.
- Extract a `WebGateway`.
- Keep `run_go2_moe_multi_web.py` behavior working through the new modules.

### Milestone 2: Multi-Robot Control Loop

Goal: make shared multi-robot stepping robust and measurable.

Scope:
- Per-robot controller state.
- Per-robot source-agnostic command state.
- A `CommandHub` or equivalent abstraction for latest desired command,
  command source, command age, brake/release state, and stale timeout.
- `WebGateway` writes commands into the shared command state instead of being
  read directly by the low-level control loop.
- One shared `client.step()` per control tick.
- Per-robot stale-command braking.
- Clean shutdown that restores UI control source.
- Control-loop metrics.

Metrics to expose:
- Tick duration.
- Policy inference duration.
- UE step duration.
- Missed deadlines.
- Active robot count.
- Per-robot command age.

This milestone should answer whether N dogs can be controlled reliably at 50 Hz.
It should also ensure that future ROS 2, joystick, and navigation gateways can
feed the same command path without rewriting the policy loop.

### Milestone 3: Policy Runtime Optimization

Goal: improve policy runtime efficiency without changing the user-facing web control behavior.

Scope:
- Load one shared policy model per policy file/type.
- Keep separate history/state per robot.
- Support batched inference for robots using the same policy.
- Add `--device cpu/cuda` or equivalent config.
- Add a benchmark/report mode.

This milestone is where GPU support should be introduced if needed, because batching makes GPU usage much more meaningful.

### Milestone 4: ROS 2 Gateway

Goal: let ROS 2 stacks exchange commands, state, sensors, and cameras with the
central control server without creating another competing URLab client.

Scope:
- Add an optional `Ros2Gateway` using `rclpy`.
- Subscribe to standard command topics such as `/<robot>/cmd_vel`
  (`geometry_msgs/Twist`) and write them into the shared command state.
- Publish per-robot state topics such as `/<robot>/joint_states`, odometry or
  base state, sensor topics, and camera topics where available.
- Keep ROS 2 callbacks non-blocking relative to the 50 Hz locomotion loop.
- Reuse the same robot registry and command ownership/brake semantics as the
  web gateway.

This milestone should not replace URLab's native ZMQ/Python client path. ROS 2
is a gateway on top of the central server, so web control, ROS 2 teleop, and
future autonomy feed the same command hub.

### Milestone 5: Unified Web UI

Goal: improve control convenience.

Scope:
- One web server port.
- `/robots` page listing all controlled robots.
- `/robots/<id>` control page for each robot.
- Robot health, active control state, brake state, and connection state.
- Links/buttons to open a robot's control page.

This milestone should move away from requiring one port per robot, while preserving the ability for different users to control different dogs from different browser pages.

### Milestone 6: Leasing and Safety

Goal: prevent users from fighting over the same robot and add stronger safety semantics.

Scope:
- Robot lease ownership.
- Idle timeout.
- Manual release.
- Admin or force-release path.
- Global emergency stop.
- Per-robot brake and stale watchdog.

This milestone should be completed before serious LAN or multi-user use.

### Milestone 7: Spawning Integration

Goal: connect web control to robot creation and discovery.

Scope:
- Discover existing UE robots.
- Register spawned articulations automatically.
- Optional spawn API.
- Automatic control-page creation for new robots.
- Cleanup when robots are removed.

This turns the server from controlling a fixed set of existing dogs into managing a robot fleet.

### Milestone 8: Navigation Layer

Goal: add autonomy above locomotion.

Scope:
- Navigation command interface: target pose, waypoint, or path.
- Navigation loop at 1-10 Hz.
- Navigation output as velocity commands into the locomotion controller.
- Manual web override.
- Stop, resume, and cancel task controls.

Low-level locomotion should remain the 50 Hz layer. Navigation should not be embedded directly inside the locomotion policy loop.

### Milestone 9: Perception and Camera Feedback

Goal: add richer remote operation without blocking control.

Scope:
- Async camera streams.
- Per-robot camera selection.
- Web video display.
- Optional low-rate perception hooks.
- Backpressure so camera/perception cannot stall the 50 Hz control loop.

Camera and perception work must be separated from the low-level control tick.

### Milestone 10: Task-Level Runtime

Goal: support higher-level behaviors such as go-to-target, patrol, follow, search, and task queues.

Scope:
- Task state machine.
- Behavior queue.
- Event callbacks.
- Task planner API.
- Human override.

This layer should sit above navigation, not inside locomotion.

## Preferred Evolution Path

Start with Milestone 1 and Milestone 2 before adding new user-facing capabilities. These two milestones establish the server boundary and prove the core timing model.

After that, choose the next milestone based on the immediate research need:

- If more dogs are needed, do Milestone 3.
- If ROS 2 integration is needed, do Milestone 4.
- If better usability is needed, do Milestone 5.
- If multiple humans will control dogs, do Milestone 6.
- If robots must be created dynamically, do Milestone 7.
- If autonomy is needed, do Milestone 8 onward.

Each milestone should have its own detailed implementation plan and test plan before code changes begin.
