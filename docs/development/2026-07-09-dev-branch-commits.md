# Development Log: URLab_Bridge July 9 Dev Commits

Date: 2026-07-09

This note documents the commits added on top of:

```text
79b1bd4 chore(policy): switch raw runner to ZMQ control
```

It is meant as a quick map for future development and review. It does not list
test output; use the commit diffs and focused tests for verification details.

## Commit Stack

```text
1ea0481 feat(control): add Go2 web server skeleton
4f721e5 feat(go2): add browser policy control
4914ba0 feat(client): scope live step controls
```

## 4914ba0: Scoped Live Step Controls

Purpose: let one active URLab client update only the articulations owned by the
current policy loop.

Main files:

- `src/urlab_client/client.py`
- `src/urlab_client/namespaces/runtime.py`
- `tests/test_transport.py`
- `tests/wire_replies.py`

Developer notes:

- `URLabClient.step(..., control_articulations=...)` narrows the per-articulation
  control payload for direct/live stepping.
- Use this in multi-robot control loops so one robot's runner does not resend
  stale controls for every discovered articulation.
- `runtime.set_twist_control_state(...)` is available for web or UE UI state
  sync, but callers should keep graceful fallback behavior for older plugin
  builds.

## 4f721e5: Browser Policy Control

Purpose: add browser keyboard control for Go2 policy loops without requiring a
separate terminal process to own keyboard focus.

Main files:

- `src/urlab_bridge/web_control.py`
- `src/urlab_bridge/cli/web_control.py`
- `scripts/run_go2_moe_web.py`
- `scripts/run_go2_moe_keyboard.py`
- `src/urlab_policy/go2/twist_commands.py`
- `tests/test_web_control.py`
- `tests/test_go2_moe_web.py`
- `tests/test_go2_twist_commands.py`
- `pyproject.toml`

Developer notes:

- Browser input and terminal input now follow the same command convention:
  `W/S` forward, `Q/E` lateral, `A/D` yaw.
- `WebCommandSource` is the policy-loop adapter; it stores the latest command,
  applies stale braking, and can sync dash/UI state through the runtime
  namespace.
- `run_go2_moe_web.py` keeps the policy loop and web server in one process, so
  browser control does not create a second competing URLab RPC session.

## 1ea0481: Go2 Web Server Skeleton

Purpose: extract the multi-Go2 web runner into reusable central-control-server
modules while preserving current behavior.

Main files:

- `docs/superpowers/specs/2026-07-09-central-robot-control-server-roadmap.md`
- `scripts/run_go2_moe_multi_web.py`
- `src/urlab_bridge/control_server/`
- `tests/test_go2_moe_multi_web.py`

Developer notes:

- `URLabControlServer` wires together one `SessionManager`, one `WebGateway`,
  the configured targets, and the `Go2MoeControlLoop`.
- `RobotRegistry` owns target validation and preserves ordered
  `control_articulations`.
- `WebGateway` still follows the Milestone 1 model: one web page and one port
  per robot.
- `Go2MoeControlLoop` keeps the existing lifecycle: preflight, pose staging,
  ZMQ handoff, warmup, shared `client.step(...)`, command release, and UI
  restore.
- This commit intentionally does not add spawning, leases, camera feedback,
  navigation, joystick support, metrics, batched inference, or a unified
  `/robots` UI.

## Left Uncommitted

These local artifacts remain outside the commit stack:

```text
.vscode/
generated_scenes/
policies/
scripts/ue_import_mjcf_scene_with_physics.py
```

Reason:

- `.vscode/` is local editor state.
- `generated_scenes/` is generated output.
- `policies/` contains local model artifacts.
- `ue_import_mjcf_scene_with_physics.py` is an abandoned experiment and is not
  part of the QuickConvert/web-control path.

## feat(control): Add Multi-Robot Command Hub Metrics

Purpose: make the shared multi-Go2 web policy loop robust and measurable without
adding batching, spawning, leasing, camera feedback, navigation, or UE plugin
changes.

Main files:

- `src/urlab_bridge/control_server/commands.py`
- `src/urlab_bridge/control_server/metrics.py`
- `src/urlab_bridge/control_server/go2_moe.py`
- `src/urlab_bridge/control_server/web_gateway.py`
- `src/urlab_bridge/control_server/server.py`
- `src/urlab_bridge/web_control.py`
- `scripts/run_go2_moe_multi_web.py`
- `tests/test_control_server_commands.py`
- `tests/test_control_server_metrics.py`
- `tests/test_go2_moe_multi_web.py`
- `tests/test_web_control.py`

Developer notes:

- `CommandHub` owns per-robot command state and keeps stale braking isolated to
  the robot whose browser stopped sending commands.
- `RobotCommandPort` adapts the command hub to the existing web command-source
  interface used by `WebGateway` and `Go2MoeControlLoop`.
- `ControlLoopMetrics` records tick count, missed deadlines, policy duration,
  UE step duration, max tick duration, active robot count, and per-robot command
  summaries.
- Each per-robot web server can expose the same global diagnostics through
  `GET /metrics`.
- `run_go2_moe_multi_web.py` now accepts `--metrics-log-interval-s`; use `0` to
  disable periodic metric logs.

Verification:

```text
/home/xin/app/miniconda3/envs/urlab/bin/python -m py_compile ...
/home/xin/app/miniconda3/envs/urlab/bin/python -m pytest ...
```

The focused pytest set covers command hub behavior, metrics serialization,
multi-web integration, web handler metrics, and existing Go2 web/keyboard
regressions.
