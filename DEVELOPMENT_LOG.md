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
