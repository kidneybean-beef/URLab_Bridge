# Development Log: URLab_Bridge Dev Branch

Date: 2026-07-07

This note explains the `dev` branch commits added after merging
`feat/remote_stepping_clean` into `main`. It is meant as a quick map for future
development, not a full run log.

## Branch Context

`main` was fast-forwarded to:

```text
bd3a63b feat(camera): cv2 popout for live feeds + v2 capture-time content age
```

`dev` was then created from that updated `main`.

## Dev Commit Stack

```text
79b1bd4 chore(policy): switch raw runner to ZMQ control
31cc236 feat(go2): add keyboard twist control
4d7a1d6 feat(go2): add policy deployment runners
f190d46 feat(scene): add MJCF QuickConvert importer
73ec0d0 feat(client): add batched QuickConvert outliner API
```

## 73ec0d0: Batched QuickConvert Client API

Purpose: expose the plugin-side `outliner.add_quick_convert_many` RPC in the
Python client so large imported scenes can attach QuickConvert physics with one
request.

Main files:

- `src/urlab_client/namespaces/outliner.py`
- `src/urlab_client/results.py`
- `src/urlab_client/__init__.py`
- `tests/test_editor_ops.py`
- `tests/wire_replies.py`

Developer notes:

- Keep `add_quick_convert` and `add_quick_convert_many` compatible but separate.
- Batch replies are partial-result based: one failed item should not hide other
  item results.
- If the plugin protocol adds QuickConvert fields, update payload building,
  result decoding, and tests together.

## f190d46: MJCF QuickConvert Scene Importer

Purpose: import MJCF scene geoms such as stairs, race tracks, and obstacle boxes
as UE static mesh actors, then attach URLab QuickConvert physics through the
client batch API.

Main files:

- `scripts/ue_import_mjcf_scene_quickconvert.py`
- `src/urlab_tools/mjcf_scene_geoms.py`
- `src/urlab_tools/mjcf_quickconvert_scene.py`
- `src/urlab_tools/mjcf_quickconvert_workflow.py`
- `src/urlab_tools/ue_mjcf_scene_spawn.py`
- `tests/test_mjcf_*`
- `tests/test_ue_mjcf_scene_spawn.py`

Developer notes:

- Scene obstacles are intentionally **not** imported as `MjArticulation`.
- XML parsing, UE actor spawning, and QuickConvert attachment are separate
  layers. Keep those boundaries clean.
- `ue_import_mjcf_scene_with_physics.py` is not committed because it was the
  abandoned articulation-based experiment.
- Start simulation only after the importer finishes; otherwise spawned meshes
  may exist before QuickConvert collision is attached.

## 4d7a1d6: Go2 Policy Deployment Runners

Purpose: add Go2 policy helpers and runner scripts for pose capture, Genesis
policy testing, Unitree RL Gym MoE compatibility checks, shadow runs, and live
policy application.

Main files:

- `src/urlab_policy/go2/pose.py`
- `src/urlab_policy/go2/unitree_policy.py`
- `src/urlab_policy/go2/unitree_rl_gym_moe.py`
- `src/urlab_policy/go2/unitree_rl_gym_moe_compat.py`
- `scripts/run_go2_capture_hold.py`
- `scripts/run_go2_policy_*.py`
- `scripts/run_go2_moe_{preflight,shadow,apply}.py`
- `tests/test_go2_*`

Developer notes:

- Policy joint order, MJCF joint order, and actuator order must be checked
  explicitly.
- Default stand pose belongs to the policy adapter, not the generic client.
- Safe handoff stages the current pose before switching the articulation to ZMQ.
- When a policy fails in URLab, check model compatibility, observation order,
  default pose error, command convention, timestep/decimation, and contact
  settings before changing safety limits.

## 31cc236: Keyboard and UE Twist Control

Purpose: let the MoE Go2 runner receive motion commands from terminal keyboard
input or UE-side twist state.

Main files:

- `src/urlab_policy/go2/twist_commands.py`
- `scripts/run_go2_moe_keyboard.py`
- `scripts/run_go2_moe_ue_twist.py`
- `tests/test_go2_twist_commands.py`
- `tests/test_go2_moe_keyboard.py`
- `tests/test_go2_moe_ue_twist.py`

Developer notes:

- Keep command sign conventions in the command-source layer.
- Avoid duplicating policy handoff, rate-limit, and safety logic from
  `run_go2_moe_apply.py`.
- Terminal keyboard control and UE twist control should feed the same policy
  command vector shape.

## 79b1bd4: Raw Runner ZMQ Control

Purpose: make `run_raw_policy.py` switch the selected articulation to ZMQ before
sending controls.

Main file:

- `scripts/run_raw_policy.py`

Developer notes:

- Do not hardcode generated articulation names such as `go2_C_1` or `go2_C_3`.
- Resolve the prefix from `--articulation` or the single available articulation,
  then switch that prefix to ZMQ.

## Left Uncommitted

These local files were intentionally not included in the `dev` commits:

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
- `ue_import_mjcf_scene_with_physics.py` is the abandoned scene-as-articulation
  importer.

## Verification

Before this log was written, the committed code was checked with:

- focused pytest suite for changed client, scene, and Go2 modules
- Ruff over changed source, scripts, and tests

Current known result:

```text
127 passed
ruff: all checks passed
```
