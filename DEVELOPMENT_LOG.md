# URLab_Bridge Development Log

This file records concise implementation notes for development work after the
`dev` branch commit stack documented in
`docs/development/2026-07-07-dev-branch-commits.md`.

The previous committed baseline is:

```text
79b1bd4 chore(policy): switch raw runner to ZMQ control
```

Entries below describe local development after that commit. They are written to
help future commit batching: one commit should usually correspond to a coherent
feature batch, not to every individual edit.

## Current Capability Summary

The bridge now has the pieces for browser-driven Go2 locomotion without running
a second competing URLab client from another terminal. A browser page can feed
velocity commands into the Unitree RL Gym Go2 MoE policy loop, while the Python
process owns the active URLab session and sends scoped controls only for the
selected articulations.

Single-robot web control is available through `run_go2_moe_web.py`, and
multi-robot web control is available through `run_go2_moe_multi_web.py` with one
web page per configured robot. The multi-web runner has been refactored into a
new `urlab_bridge.control_server` package so the current behavior can grow into
a central robot-control server without keeping all orchestration inside one
script.

## 2026-07-09 - Scoped Control Stepping and Twist UI State

- Added `URLabClient.step(..., control_articulations=...)` for direct/live step
  requests.
- When `control_articulations` is provided, the client sends control payloads
  only for those articulation prefixes instead of resending stale controls for
  every discovered robot.
- Added `runtime.set_twist_control_state(...)` so Python-side web command
  sources can push UI-facing twist limits and dash state to URLab when the
  plugin supports it.
- Added wire-test helpers and transport tests for selected-articulation step
  payloads and twist-control-state RPCs.

Main files:

- `src/urlab_client/client.py`
- `src/urlab_client/namespaces/runtime.py`
- `tests/test_transport.py`
- `tests/wire_replies.py`

Developer notes:

- Use `control_articulations` whenever a policy loop is meant to own only a
  subset of robots in a scene.
- Missing articulation names raise a `KeyError` with the available prefixes, so
  callers should resolve user-facing names before entering the control loop.
- `set_twist_control_state` is optional from the web-control side; older plugin
  builds should degrade by disabling UI sync rather than breaking locomotion.

## 2026-07-09 - Keyboard Command Convention Cleanup

- Changed Go2 command keys so `W/S` control forward velocity, `Q/E` control
  lateral velocity, and `A/D` control yaw.
- Updated help text, tests, README wording, and policy/demo comments from
  "WASD twist control" to the more accurate "keyboard twist control".
- Updated `run_go2_moe_keyboard.py` to:
  - use `control_articulations=(prefix,)` for every step call
  - sync command-source UI state before polling
  - release/zero command sources during teardown before restoring UI control

Main files:

- `src/urlab_policy/go2/twist_commands.py`
- `scripts/run_go2_moe_keyboard.py`
- `tests/test_go2_twist_commands.py`
- `tests/test_go2_moe_keyboard.py`
- `README.md`
- related policy/demo comments under `src/urlab_policy/` and `scripts/`

Developer notes:

- Keep key-to-command sign conventions in the command-source layer.
- Do not duplicate policy safety, handoff, action clipping, or target slew logic
  outside the existing Go2 MoE runners.
- Keep terminal keyboard, UE twist input, and browser input producing the same
  `[vx, vy, yaw]` command shape.

## 2026-07-09 - Browser Web Control for Go2 Policy Loops

- Added `urlab_bridge.web_control`, a dependency-light HTTP/browser control
  module.
- Added browser key state parsing, dash limits, stale timeout braking, release
  handling, JSON health/control endpoints, and an in-memory `WebCommandSource`
  for policy loops.
- Added `urlab-web-control` CLI entry point for direct browser-to-URLab twist
  control of one articulation.
- Added `run_go2_moe_web.py`, which runs the Go2 MoE policy and browser web page
  in the same Python process so the policy receives browser commands without a
  second URLab RPC session.
- Added focused unit tests for browser key mapping, stale timeout behavior, HTTP
  request handling, UI sync fallback, and web policy integration.

Main files:

- `src/urlab_bridge/web_control.py`
- `src/urlab_bridge/cli/web_control.py`
- `scripts/run_go2_moe_web.py`
- `tests/test_web_control.py`
- `tests/test_go2_moe_web.py`
- `pyproject.toml`

Developer notes:

- The browser control page is intentionally simple and local-process based.
- `WebCommandSource` is the bridge between browser input and policy runners; it
  stores the current command, handles stale braking, and can sync UE UI state.
- The direct `WebControlBroker` path still exists for one-articulation twist
  control without a policy loop.

## 2026-07-09 - Multi-Go2 Web Policy Runner

- Added `run_go2_moe_multi_web.py` for controlling multiple existing Go2
  articulations from one active URLab session.
- Preserved one web page per robot with repeated
  `--web-target ARTICULATION:PORT`.
- Added validation for duplicate articulation targets, duplicate ports, invalid
  ports, and fallback `--articulation X --web-port P` single-target mode.
- Kept one policy instance and one `WebCommandSource` per robot for now.
- The runner performs the same lifecycle as the single-Go2 policy runner:
  preflight, captured-pose staging, UI-to-ZMQ handoff, optional warmup, shared
  step loop, command release, and UI control-source restore.
- The shared step loop uses
  `client.step(..., control_articulations=(...))` so one RPC tick advances all
  configured dogs without sending unrelated controls.

Main files:

- `scripts/run_go2_moe_multi_web.py`
- `tests/test_go2_moe_multi_web.py`

Developer notes:

- This runner assumes all target robots already exist in the UE scene.
- It intentionally keeps the "one port per robot" model; the unified `/robots`
  UI belongs to a later control-server milestone.
- Policy model sharing and batched inference are not part of this batch.

## 2026-07-09 - Central Robot-Control Server Roadmap

- Added a roadmap for moving from ad-hoc web scripts toward a central
  robot-control server.
- Captured the preferred architecture:
  `URLabControlServer`, `SessionManager`, `RobotRegistry`, `ControlLoop`,
  `WebGateway`, `PolicyRuntime`, and higher-level runtime layers.
- Defined staged milestones for server skeleton, robust multi-robot control,
  policy runtime optimization, unified web UI, leasing/safety, spawning,
  navigation, perception, and task-level runtime.

Main file:

- `docs/superpowers/specs/2026-07-09-central-robot-control-server-roadmap.md`

Developer notes:

- Keep the low-level locomotion loop separate from web requests, camera work,
  navigation, perception, and task planning.
- The roadmap is not implementation by itself; each milestone should get a
  focused plan before code changes.

## 2026-07-09 - Control Server Skeleton Extraction

- Added `src/urlab_bridge/control_server/` as the first reusable server package.
- Added public server components:
  - `SessionManager`: owns a single `URLabClient`, connects once, exposes
    `client` and `runtime`, and closes the session during teardown.
  - `RobotRegistry`: parses/validates `ARTICULATION:PORT` targets and preserves
    ordered `control_articulations`.
  - `WebGateway`: starts one `ThreadingHTTPServer` and one `WebCommandSource`
    per configured robot, then closes all handles on teardown.
  - `Go2MoeControlLoop`: contains the existing multi-web Go2 MoE lifecycle while
    taking policy/helper functions as injected dependencies.
  - `URLabControlServer`: wires session, registry targets, web gateway, and the
    Go2 control loop together.
  - `WebPolicyTarget`: shared target tuple for CLI and package code.
- Refactored `run_go2_moe_multi_web.py` into a compatibility wrapper that keeps
  its current CLI and delegates orchestration to `URLabControlServer`.
- Expanded tests to cover registry validation, web gateway lifecycle, session
  ownership, one-client multi-robot stepping, control-source transitions, and
  CLI parsing.

Main files:

- `src/urlab_bridge/control_server/__init__.py`
- `src/urlab_bridge/control_server/models.py`
- `src/urlab_bridge/control_server/registry.py`
- `src/urlab_bridge/control_server/session.py`
- `src/urlab_bridge/control_server/web_gateway.py`
- `src/urlab_bridge/control_server/go2_moe.py`
- `src/urlab_bridge/control_server/server.py`
- `scripts/run_go2_moe_multi_web.py`
- `tests/test_go2_moe_multi_web.py`

Developer notes:

- This is a structural extraction only. It does not add spawning, leases,
  camera feedback, navigation, joystick support, metrics, batched inference, or
  a unified `/robots` UI.
- Keep policy math, action limiting, target slew, warmup, and restore semantics
  in sync with the existing Go2 MoE runners.
- `Go2MoeControlLoop` receives dependencies from the compatibility CLI so the
  package can be tested without importing every script helper directly.

## 2026-07-17 - Per-Robot LAN RGB Camera Streaming

- Added `CameraHub` and `MjpegCameraStream` to the central control server.
- A configured robot camera is read from the existing `URLabClient`; camera
  output never creates a second URLab session.
- Camera workers encode the newest URLab RGBA frame as JPEG on a dedicated
  thread. Old frames are dropped instead of queued, keeping camera latency and
  memory bounded independently of the 50 Hz locomotion loop.
- Extended each existing per-robot web server with:
  - `GET /api/camera/stream.mjpg` for the browser MJPEG feed
  - `GET /api/camera/status` for camera availability and frame age
- Added an optional camera panel alongside the existing keyboard/touch controls.
  Camera-enabled desktop pages place controls on the left and the stream on the
  right, then stack controls above the stream below 900 px. Pages without a
  configured camera retain their centered control layout and existing routes.
- Exposed the existing left-Shift dash state as a holdable web control. The Dash
  button uses the same command path and runtime limits as the physical key.
- Added repeatable `--web-camera ARTICULATION:CAMERA` mappings plus
  `--camera-fps` and `--camera-jpeg-quality` controls to
  `run_go2_moe_multi_web.py`.
- Declared Pillow in the optional `web-camera` dependency group. Importing the
  control server without camera output does not import or require Pillow.

Main files:

- `src/urlab_bridge/control_server/cameras.py`
- `src/urlab_bridge/control_server/models.py`
- `src/urlab_bridge/control_server/server.py`
- `src/urlab_bridge/control_server/web_gateway.py`
- `src/urlab_bridge/web_control.py`
- `scripts/run_go2_moe_multi_web.py`
- `tests/test_control_server_cameras.py`
- `tests/test_web_control.py`

Developer notes:

- Camera streaming must continue to use the session owned by `SessionManager`.
- HTTP request threads only write already-encoded JPEG bytes; image encoding
  belongs in `MjpegCameraStream`, never in the policy loop or request handler.
- One camera can serve multiple browser connections without duplicating JPEG
  encoding because all clients read the stream's latest encoded frame.
- `--web-camera` uses the exact camera key exposed in the articulation's
  `cameras` mapping. A missing camera fails before the control loop starts and
  reports the available names.
- Web JPEG encoding applies the standard linear-to-sRGB display transfer
  function. `UMjCamera` captures `SCS_FinalToneCurveHDR` into a linear
  `PF_B8G8R8A8` target, while UE applies the transfer function when displaying
  that target itself. Keep URLab's raw camera array linear for policy and
  vision consumers; display conversion belongs at the browser encoder.

## 2026-07-20 - Smooth Per-Robot Twist Commands

- Added a deterministic `TwistSlewLimiter` with independent acceleration and
  deceleration rates for forward, lateral, and yaw commands.
- Each Go2 policy state owns its limiter. Policy history starts at zero command,
  and each 50 Hz policy observation advances the applied twist by one bounded
  step toward the command source's desired twist.
- Space is an explicit immediate brake: it resets all applied axes to zero in
  the current policy tick instead of following the normal deceleration ramp.
- Preserved the existing normal and dash velocity defaults. Added
  `--cmd-accel-*` and `--cmd-decel-*` options for tuning only the transition
  rates.
- Added desired and applied twists to command diagnostics so command shaping
  can be distinguished from browser input and policy output.

Main files:

- `src/urlab_bridge/control_server/commands.py`
- `src/urlab_bridge/control_server/go2_moe.py`
- `src/urlab_bridge/web_control.py`
- `scripts/run_go2_moe_multi_web.py`
- `tests/test_control_server_commands.py`
- `tests/test_go2_moe_multi_web.py`

Developer notes:

- Keep command smoothing in the control server, after source polling and before
  policy observation construction. UE remains responsible for simulation and
  transport rather than controller-specific command shaping.
- New command sources, including ROS2, should expose their desired twist and an
  explicit brake state so they share this limiter without duplicating it.

## 2026-07-20 - Concurrent Multi-Robot Camera Feeds

- Confirmed the control server already keys camera streams by articulation, so
  two robots may both expose a camera named `front_rgb` without sharing a web
  stream or JPEG encoder.
- Added regressions for repeated `--web-camera ARTICULATION:CAMERA` mappings and
  for resolving same-named cameras to distinct articulation camera views.
- Fixed URLab's dashboard preview lifetime separately in the plugin: switching
  the selected articulation no longer disables a camera that is still required
  by the network broadcaster.

Main files:

- `tests/test_control_server_cameras.py`
- `tests/test_go2_moe_multi_web.py`

Developer notes:

- Camera identity in the control server is `(articulation, camera)`, not the
  camera name alone.
- Configure one repeated `--web-camera` mapping for each web-controlled robot.
- Dashboard selection and browser streaming are independent consumers; changing
  the selected or possessed robot must not determine which LAN streams remain
  active.

## 2026-07-28 - Measured Web Camera FPS

- Added an FPS badge over each configured web camera feed.
- The browser samples the existing camera status endpoint once per second and
  derives FPS from the change in `encoded_frames` over monotonic elapsed time.
- The display reports the measured JPEG stream production rate rather than the
  configured `--camera-fps` ceiling. A stalled stream falls to `0.0 FPS`, while
  unavailable or reset counters display `-- FPS`.

Main files:

- `src/urlab_bridge/web_control.py`
- `tests/test_web_control.py`

Developer notes:

- Keep this metric separate from UE renderer FPS (`stat fps`) and MuJoCo step
  frequency; each measures a different stage of the camera/control pipeline.
- `encoded_frames` remains the source of truth for the web badge. Do not infer
  delivered FPS from the configured camera rate.

## 2026-07-29 - Per-Camera Runtime Power Control

- Added `runtime.get_camera_enabled()` and `runtime.set_camera_enabled()` for
  explicit Python control of URLab camera master-switch RPCs.
- Camera handshake views retain their initial enabled state.
- URLab-disabled streams stop UE capture/readback. URLab
  suspends the camera's ZMQ publisher while retaining its endpoint, so existing
  subscribers reconnect when re-enabled without reconstructing the control
  server.

Main files:

- `src/urlab_client/articulation.py`
- `src/urlab_client/namespaces/runtime.py`
- `src/urlab_bridge/control_server/cameras.py`
- `src/urlab_bridge/web_control.py`
- `tests/test_control_server_cameras.py`
- `tests/test_web_control.py`
- `tests/test_entities.py`
- `tests/test_transport.py`

Developer notes:

- Camera power is explicit per robot; it is independent of possession and
  does not disable cameras on other web-controlled robots.
- The default comes from each URLab camera component's `Start Enabled`
  property. Browser controls must not call these runtime APIs; capture and
  transport remain owned by URLab, Blueprint defaults, or explicit Python
  clients.

## 2026-07-29 - Discovered Multi-Camera Web Controls

- Replaced the single configured web-camera switch with a per-robot camera
  inventory sourced from URLab's articulation handshake.
- `--web-camera ARTICULATION:CAMERA` now selects the initial preview; it does
  not define or hardcode the available camera list.
- Added a selector and independent browser-local show/hide control for every
  camera URLab advertises on that articulation. Camera names and modes remain
  model-defined.
- Added browser previews for depth, semantic, and instance modes. Depth is
  normalized with the camera's URLab-authored `depth_near_cm` and
  `depth_far_cm`; segmentation colors bypass the RGB linear-to-sRGB
  conversion and are reordered from ID-preserving BGRA wire order to RGB only
  at JPEG display encoding.
- JPEG work is viewer-driven, so cameras remain available for runtime toggling
  without encoding every feed when only one preview is open.
- Camera changes detach and blank the previous MJPEG image before opening a
  cache-busted URL for the new camera. A camera with no fresh frame can no
  longer leave the previous camera's image dimmed in place.

Main files:

- `src/urlab_bridge/control_server/cameras.py`
- `src/urlab_bridge/web_control.py`
- `scripts/run_go2_moe_multi_web.py`
- `tests/test_control_server_cameras.py`
- `tests/test_web_control.py`

Developer notes:

- Treat `(articulation, camera)` as the camera identity. Never infer camera
  names such as `front_rgb` from the robot type.
- Validate the initial camera against the connected articulation and populate
  web choices only from `articulation.cameras`.
- Keep depth visualization consistent with URLab: near is black, far is white,
  and the fixed camera range must come from the handshake rather than
  per-frame percentiles.
- Keep semantic and instance arrays in BGRA inside `URLabCameraView`; class-ID
  consumers depend on wire order. Channel reordering belongs only in display
  encoders.
- Browser show/hide controls close only that browser's MJPEG request. They must
  not call `runtime.set_camera_enabled()` or mutate UE capture/transmission.

## 2026-08-06 - ROS2 cmd_vel Input for Go2 MoE Control

- Added an optional ROS2 command gateway for the central control server.
- `--ros2-cmd-vel` starts a ROS2 node that subscribes to one
  `/<articulation>/cmd_vel` topic per configured `--web-target`.
- Incoming `geometry_msgs/Twist` commands are mapped to the existing Go2 policy
  command tuple `(vx, vy, yaw)` and written into the shared `CommandHub` with
  source `ros2`.
- Web commands and ROS2 commands now share the same latest-command state. The
  newest source controls the robot, and the existing stale timeout brakes stale
  ROS2 commands just like stale web commands.

Main files:

- `src/urlab_bridge/control_server/ros2_gateway.py`
- `src/urlab_bridge/control_server/commands.py`
- `src/urlab_bridge/control_server/server.py`
- `scripts/run_go2_moe_multi_web.py`
- `tests/test_control_server_ros2_gateway.py`

Developer notes:

- This gateway is command input only. ROS2 state/camera export remains separate
  from the older `urlab_tools.ros2_broadcaster` path and belongs to a later
  milestone.
- ROS2 imports stay lazy so non-ROS tests and non-ROS bridge usage continue to
  work without a sourced ROS workspace.
- Topic names intentionally follow the configured URLab articulation names,
  e.g. `/go2_go2_rl_gym_C_1/cmd_vel`.

## 2026-08-06 - ROS2 State And Camera Export

- Extended the central control server's ROS2 gateway from command input to
  bidirectional bridge behavior.
- `--ros2-publish-state` publishes each controlled robot's joint state and
  odometry from the same fresh articulation state used by the Go2 MoE loop.
- `--ros2-publish-sensors` publishes available MuJoCo sensor values as generic
  `Float64MultiArray` topics without reviving the older standalone broadcaster.
- `--ros2-publish-cameras` publishes raw ROS2 image topics from the existing
  `CameraHub` views. This is independent from browser MJPEG visibility and does
  not create a second URLab client.
- The control loop now accepts an optional post-step hook, used after the shared
  `client.step(...)` so state publishers see fresh per-articulation data.

Main files:

- `src/urlab_bridge/control_server/ros2_gateway.py`
- `src/urlab_bridge/control_server/server.py`
- `src/urlab_bridge/control_server/go2_moe.py`
- `scripts/run_go2_moe_multi_web.py`
- `tests/test_control_server_ros2_gateway.py`

Developer notes:

- ROS2 publish-only mode starts the same gateway node even when
  `--ros2-cmd-vel` is disabled.
- State topics are named from configured URLab articulation names. Camera topics
  are named from the actual cameras URLab advertises on that articulation.
- Depth images are published as `32FC1` meters; URLab depth frames arrive in UE
  scene centimeters and are converted at publish time.
- Real RGB images use the same linear-to-sRGB display transfer as the web
  stream. Semantic and instance images remain BGRA to preserve URLab's camera
  color payload.
- Camera `image_raw` and `camera_info` publishers use sensor-data QoS semantics:
  best effort, volatile, keep-last, depth 1. State, odometry, and sensor topics
  keep the normal reliable queue. Live camera streams should drop stale frames
  instead of back-pressuring the bridge when a ROS2 subscriber cannot keep up.
- Added periodic ROS2 raw-camera publish diagnostics. The log reports internal
  publish FPS, raw MB/s, duplicate skips, message-build time, `rclpy.publish()`
  call time, bytes per frame, and per-camera publish counts. Use this before
  changing camera payload formats so bottlenecks can be assigned to URLab frame
  source, Python message construction, ROS2 publish, or subscriber/DDS receive.
- Added `scripts/ros2_image_bench.py`, a lightweight ROS2 image subscriber
  benchmark that reports receive Hz and raw bandwidth from the same callback
  stream. It defaults to best-effort, keep-last, depth-1 QoS to match URLab's
  live camera publishers and avoids comparing separate `ros2 topic hz` and
  `ros2 topic bw` runs. It uses a persistent executor and separate interval
  counters so live `window` lines are true per-window measurements, while the
  final line reports the full run.
- Added an optional ROS2 compressed camera preview path. Raw `sensor_msgs/Image`
  topics remain available for correctness/debug, while
  `--ros2-publish-compressed-cameras` publishes JPEG
  `sensor_msgs/CompressedImage` topics at
  `/<articulation>/camera/<camera>/image_raw/compressed` using the same display
  encoder as the web stream. Use this when comparing ROS2 visual streaming
  against the browser MJPEG path.
- Added `ros2/urlab_image_transport_relay`, a small C++ ROS2 sidecar for the
  official `image_transport` plugin path. It subscribes to a URLab raw
  `sensor_msgs/Image` topic and republishes on a distinct output base topic
  through `image_transport`, allowing installed transports such as
  `compressed`, `compressedDepth`, `turbojpeg`, or `zstd` to provide standard
  derived topics. Keep the relay input and output topics different to avoid a
  raw-transport feedback loop.

## 2026-08-14 - Session-Safe Direct Camera Discovery

- Added URLab's sessionless `describe_runtime` contract to the direct C++
  camera bridge path. The bridge now discovers camera ZMQ endpoints without
  issuing `hello`, so it cannot replace the Python policy server's active
  controller session.
- Split runtime-description msgpack packing/parsing into a small reusable C++
  unit and added tests for all camera modes, malformed replies, disabled ZMQ
  broadcasting, and the one-field request shape.
- Removed sidecar camera-power control. `auto_enable_cameras=true` is retained
  only as a compatibility parameter and now fails before any URLab RPC.
  Camera capture remains owned by URLab UI, Blueprint, or the controller
  session that already owns the robot.
- The direct bridge logs the discovered camera transport metadata at startup so
  endpoint, capture state, and output resolution failures are actionable.

## 2026-08-14 - Real Camera Payload Contracts

- Added metadata-driven support for both `bgra8_linear` and `bgra8_srgb` Real
  camera payloads across the browser, Python ROS2 publisher, and direct C++
  ROS2 bridge.
- The attempted `bgra8_srgb` plugin default was rolled back after its Unreal
  render-target path produced black live captures. The active URLab contract is
  `bgra8_linear`; consumers apply the existing LUT before display or `rgb8`
  publication.

Developer notes:

- Treat `payload_encoding` as the source-of-truth camera color contract. Do
  not infer gamma from `mode == real`.
- `bgra8_srgb` remains a supported forward-compatible branch, but current
  plugin output is `bgra8_linear`.

Main files:

- `ros2/urlab_image_transport_relay/src/urlab_camera_bridge.cpp`
- `ros2/urlab_image_transport_relay/src/runtime_description.cpp`
- `ros2/urlab_image_transport_relay/include/urlab_image_transport_relay/runtime_description.hpp`
- `ros2/urlab_image_transport_relay/test/test_runtime_description.cpp`

## 2026-08-17 - Front RGB Rendering and Camera Calibration

### Spatial post processing

- Removed the camera `BeginPlay` copy of the first enabled
  `PostProcessVolume`. Unreal now blends volumes at each sensor location using
  normal bounds, priority, and blend-weight rules.
- Kept the independent capture path. URLab still fills missing project GI,
  reflection, and Lumen cache overrides without replacing post-process settings
  authored on the camera.

### Temporal anti-aliasing

- Real cameras retain the underlying Scene Capture component's native
  **Advanced Show Flags** for anti-aliasing. URLab does not expose a second
  Temporal-AA property or force an AA method, while motion blur stays disabled
  for every camera mode.
- The temporary wrapper property and scoped view-extension experiment were
  removed: the SceneCapture `TemporalAA` show flag is not an AA-method
  selector, and neither mechanism addressed the black-frame diagnosis.

### Projection and ROS2 calibration

- Kept public `fovy` as MuJoCo vertical FOV and converted it to Unreal's
  horizontal `FOVAngle` using the active aspect ratio for every camera mode.
- Replaced width/height-only ROS2 `CameraInfo` messages in both the Python
  publisher and direct C++ sidecar with a centered square-pixel pinhole model.
  `D`, `K`, `R`, and `P` are now populated consistently from resolution and
  vertical FOV; invalid metadata fails before camera publishing starts.

Developer notes:

- Do not pass MuJoCo `fovy` directly to Unreal `FOVAngle`; the two APIs use
  vertical and horizontal conventions respectively.
- Keep Python and C++ calibration formulas synchronized. At 640x480 and 90
  degrees vertical FOV, both must produce `fx=fy=240`, `cx=319.5`, and
  `cy=239.5`.
- The original calibration work did not alter the Real-camera payload contract;
  the final-LDR sRGB update below supersedes its former linear contract.

### Final LDR sRGB Real Camera Contract

- Changed URLab `Real` cameras from the intermediate
  `SCS_FinalToneCurveHDR` / linear target path to
  `SCS_FinalColorLDR` / `RTF_RGBA8_SRGB`.
- The camera render target leaves `TargetGamma` unset so its sRGB contract is
  not overridden by the editor display gamma. URLab now advertises the native
  BGRA bytes as `bgra8_srgb`.
- Browser, Python ROS2, and direct C++ image-transport consumers already branch
  on `payload_encoding`; new frames take the channel-reorder-only path while
  old `bgra8_linear` plugins retain exactly one legacy display transfer.
- Depth, semantic, and instance capture paths are unchanged.

Main files:

- `src/urlab_bridge/control_server/ros2_gateway.py`
- `ros2/urlab_image_transport_relay/include/urlab_image_transport_relay/runtime_description.hpp`
- `ros2/urlab_image_transport_relay/src/runtime_description.cpp`
- `ros2/urlab_image_transport_relay/src/urlab_camera_bridge.cpp`
- `tests/test_control_server_ros2_gateway.py`
- `ros2/urlab_image_transport_relay/test/test_runtime_description.cpp`

## Verification Notes

Recent focused checks used during this development pass:

- `tests/test_go2_moe_multi_web.py`
- `tests/test_web_control.py`
- `tests/test_go2_moe_keyboard.py`
- `tests/test_go2_moe_apply.py`
- Ruff over `src/urlab_bridge`, `scripts/run_go2_moe_multi_web.py`, and
  `tests/test_go2_moe_multi_web.py`

Keep future verification notes brief and tied to the relevant entry. Avoid
turning this file into a raw command transcript.

## Local Artifact Policy

Do not include these local or generated artifacts in feature commits unless a
future plan explicitly changes that policy:

- `.vscode/`
- `generated_scenes/`
- `policies/`
- Python `__pycache__/` directories
- abandoned experimental scripts such as
  `scripts/ue_import_mjcf_scene_with_physics.py`
