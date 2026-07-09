# Development Pickup Log: 2026-07-09

This file is a working-memory log for picking URLab_Bridge development back up
quickly. It is intentionally separate from commit documentation.

## Milestone 2: Multi-Robot Control Loop

Context:

- Milestone 1 had already split the multi-web Go2 runner into central-control
  modules: `SessionManager`, `RobotRegistry`, `WebGateway`, and
  `Go2MoeControlLoop`.
- The next goal was to keep one shared policy loop for multiple dogs, but move
  browser commands into a source-agnostic per-robot command layer and make loop
  timing visible.
- No UE plugin changes were part of this slice.

What changed:

- Added `CommandHub` and `RobotCommandPort` in
  `src/urlab_bridge/control_server/commands.py`.
- Added `ControlLoopMetrics` in
  `src/urlab_bridge/control_server/metrics.py`.
- Updated `WebGateway` so each per-robot web page writes into one shared
  `CommandHub` through a `RobotCommandPort`.
- Updated `Go2MoeControlLoop` so every tick stale-checks each robot
  independently, polls per-robot commands, runs the shared Go2 policy step once,
  and records timing metrics.
- Added optional `GET /metrics` support to the existing web-control handler.
- Added `--metrics-log-interval-s` to `scripts/run_go2_moe_multi_web.py`.
- Updated the central-control roadmap to reflect that command hub and in-memory
  metrics are now Milestone 2 work.

Behavior to remember:

- Stale braking is per robot. A stale browser zeros only that robot command.
- `client.step(..., control_articulations=prefixes)` is still shared once per
  tick across all selected dogs.
- Metrics are in-memory diagnostics exposed from each per-robot web server for
  now. They are not Prometheus/OpenTelemetry yet.
- A missed deadline means one control-loop tick exceeded `1 / freq` seconds.
  At 50 Hz, that budget is 20 ms.
- Command age means seconds since the selected robot last received a fresh web
  control payload or release.

How to run the current two-dog path:

```bash
/home/xin/app/miniconda3/envs/urlab/bin/python scripts/run_go2_moe_multi_web.py \
  --web-target go2_go2_rl_gym_C_1:8099 \
  --web-target go2_go2_rl_gym_C2:8100
```

Useful browser endpoints:

```text
http://127.0.0.1:8099
http://127.0.0.1:8100
http://127.0.0.1:8099/metrics
http://127.0.0.1:8100/metrics
```

Verification run for this slice:

```bash
/home/xin/app/miniconda3/envs/urlab/bin/python -m py_compile \
  src/urlab_bridge/control_server/commands.py \
  src/urlab_bridge/control_server/metrics.py \
  src/urlab_bridge/control_server/go2_moe.py \
  src/urlab_bridge/control_server/web_gateway.py \
  src/urlab_bridge/control_server/server.py \
  src/urlab_bridge/web_control.py \
  scripts/run_go2_moe_multi_web.py

/home/xin/app/miniconda3/envs/urlab/bin/python -m pytest \
  tests/test_control_server_commands.py tests/test_control_server_metrics.py \
  tests/test_go2_moe_multi_web.py tests/test_web_control.py \
  tests/test_go2_moe_web.py tests/test_go2_twist_commands.py \
  tests/test_go2_moe_keyboard.py
```

Local artifacts intentionally left out of this work:

```text
.vscode/
generated_scenes/
policies/
scripts/ue_import_mjcf_scene_with_physics.py
```
