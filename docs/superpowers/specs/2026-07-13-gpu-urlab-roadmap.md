# GPU URLab Roadmap

## Purpose

This document defines the long-term development path for a GPU-capable version
of URLab. It is a roadmap, not a concrete implementation plan. Each milestone
should later be expanded into its own detailed implementation plan when it is
ready to build.

The research goal is to keep Unreal Engine as the high-fidelity rendering and
scene-authoring environment while enabling high-throughput GPU physics and RL
training for many robots. UI and operator convenience are intentionally moved to
the final milestones; early work should focus on correctness, throughput,
repeatability, and clean backend boundaries.

This roadmap complements:

- `docs/superpowers/specs/2026-07-09-central-robot-control-server-roadmap.md`

The central robot-control server roadmap focuses on multi-robot control,
web/ROS gateways, batching policy inference, and runtime coordination. This GPU
URLab roadmap focuses on replacing or augmenting the physics backend.

## Current Baseline

URLab currently integrates classic MuJoCo into Unreal Engine by embedding
upstream MuJoCo as a third-party C/C++ dependency. The UE plugin owns normal
MuJoCo objects such as `mjSpec`, `mjModel`, and `mjData`, and advances physics
with classic calls such as `mj_step`, `mj_forward`, and `mj_resetData`.

The current architecture is roughly:

```text
Unreal Engine actors/components/rendering
  URLab C++ wrappers and bridge code
    upstream classic MuJoCo C/C++ API
      CPU MuJoCo physics
```

This is stable and should remain the fallback path. The GPU roadmap should not
break the existing CPU MuJoCo workflow.

Current performance measurements distinguish two separate limits:

- Transport/control throughput: single-step RPC throughput through ZMQ/SHM.
- Physics-dominated throughput: many MuJoCo steps per RPC, measuring whole-world
  physics steps per second.

For GPU comparisons, the primary metric should be whole-world physics steps per
second, plus aggregate robot-steps per second when the world contains many
independent robot instances.

## GPU Backend Landscape

### Classic MuJoCo

Classic MuJoCo is the CPU C/C++ engine currently used by URLab. It exposes
`mjModel`, `mjData`, `mj_step`, `mj_forward`, `mj_resetData`, `mjs_addBody`,
`mjs_addGeom`, and related APIs.

It is the best baseline for:

- correctness comparison;
- single-scene interactive use;
- UE plugin stability;
- model compilation and scene import/export;
- regression tests.

### MuJoCo Warp / MJWarp

MJWarp is the NVIDIA GPU-oriented MuJoCo implementation built around NVIDIA
Warp. It does not expose the same runtime API as classic MuJoCo. A typical flow
is:

```text
classic MuJoCo MjModel on host
  -> mjw.put_model(...)
  -> mjw.make_data(..., nworld=N)
  -> mjw.step(...)
```

MJWarp is most useful for batched training with many worlds. It should be used
first as an external research backend, not immediately embedded into the UE
plugin.

### mujoco_musa

`mujoco_musa` is a Moore Threads MUSA kernel library for MuJoCo-style GPU
physics. It should be viewed as the Moore Threads GPU kernel layer, not as a
drop-in replacement for classic MuJoCo's `mj_step` API.

For NVIDIA GPUs, `mujoco_musa` is not the natural target. For Moore Threads
GPUs, it may become the native kernel layer under a MUSA-based GPU backend.

### mujoco_warp_musa

`mujoco_warp_musa` is the higher-level MJWarp-style path for Moore Threads
hardware. Conceptually:

```text
mujoco_warp
  -> NVIDIA Warp
  -> CUDA kernels
  -> NVIDIA GPU

mujoco_warp_musa
  -> MUSA-adapted MJWarp-style layer
  -> mujoco_musa kernels
  -> Moore Threads GPU
```

This roadmap should keep the NVIDIA and Moore Threads paths separate until each
has an independently measured feasibility result.

## Recommended Architecture

The long-term architecture should make URLab depend on an explicit simulation
backend contract instead of directly assuming that all physics state is stored
in `mjModel*` and `mjData*`.

```text
Unreal Engine
  URLab scene, rendering, sensors, cameras, bridge
    SimulationBackend interface
      ClassicMuJoCoBackend
        upstream CPU MuJoCo

      ExternalMJWarpBackend
        Python MJWarp process
        GPU batched physics prototype

      NativeGpuMuJoCoBackend
        future C++/CUDA or C++/MUSA backend
        only after feasibility is proven
```

The backend contract should make the following data explicit:

```text
Inputs:
  controls / actuator targets
  qpos / qvel reset state
  external forces
  mocap body poses
  timestep and solver options
  random seed
  per-environment reset requests

Outputs:
  body transforms for UE rendering
  joint positions and velocities
  actuator/sensor values
  contacts or contact summaries where needed
  observations for policies
  simulation time and frame id
  error/health/status metrics
```

The first GPU backend should not try to serve every UI and editor feature. It
should serve research loops first: reset, step, observe, render selected frames,
and compare rollouts.

## Design Principles

- Keep the current CPU MuJoCo backend as the correctness oracle and fallback.
- Do not start by rewriting MuJoCo.
- Use MJWarp externally first to prove throughput and model compatibility.
- Treat native GPU integration as a later milestone, not the first milestone.
- Optimize for many environments and batched stepping, not one interactive dog.
- Keep UI, web pages, and editor polish until the final milestones.
- Separate rendering rate from physics rate.
- Measure all claims with reproducible benchmark scripts.
- Prefer one clear backend contract over many ad-hoc bridges.
- Support NVIDIA and Moore Threads as separate backend families.

## Metrics

Every milestone should report the metrics it can measure.

Core physics metrics:

```text
world_steps_per_second
robot_steps_per_second = world_steps_per_second * robot_count
real_time_factor
mean_step_ms
p50_step_ms
p95_step_ms
p99_step_ms
```

Integration metrics:

```text
host_to_device_copy_ms
device_to_host_copy_ms
ue_render_sync_ms
policy_inference_ms
reset_ms
missed_deadlines
gpu_memory_usage
cpu_memory_usage
```

Correctness metrics:

```text
qpos_error_vs_cpu
qvel_error_vs_cpu
base_pose_error_vs_cpu
contact_event_similarity
reward_curve_similarity
policy_survival_time
```

Research benchmarks should always record:

```text
scene name
robot model
robot count
timestep
solver settings
control rate
render rate
device
driver/runtime versions
backend type
```

## Milestone Roadmap

### Milestone 0: CPU Baseline And Benchmark Discipline

Goal: establish repeatable CPU MuJoCo measurements before introducing a GPU
backend.

Scope:

- Keep using the current URLab CPU MuJoCo backend.
- Add or formalize benchmark scripts for physics-dominated throughput.
- Measure one dog, many dogs, simple ground, rough terrain, and representative
  contact-heavy scenes.
- Measure startup transients separately from settled-state throughput.
- Record both whole-world steps/s and aggregate robot-steps/s.
- Record render FPS separately from physics throughput.

Success criteria:

- One command can reproduce CPU physics throughput for a named scene.
- Results clearly distinguish communication throughput from physics throughput.
- CPU baseline numbers are saved with enough scene/config metadata to compare
  against MJWarp later.

### Milestone 1: Backend Data Contract

Goal: define the minimal simulation backend interface URLab needs before adding
any GPU physics.

Scope:

- Document required state inputs and outputs.
- Identify every current direct `mjModel*` / `mjData*` assumption in URLab that
  would block a non-classic backend.
- Define a backend-neutral representation for:
  - actuator controls;
  - qpos/qvel state;
  - body transforms;
  - joint state;
  - selected sensors;
  - per-robot observations;
  - frame id and sim time.
- Keep this as an internal C++/Python contract first, not UI.

Success criteria:

- The contract can describe current CPU MuJoCo behavior without losing needed
  state.
- The contract can also describe MJWarp batched state without pretending it is
  `mjData`.
- Later backend implementations have a stable target.

### Milestone 2: Standalone MJWarp Feasibility

Goal: prove whether the current Go2 models and representative scenes can run in
MJWarp outside UE.

Scope:

- Export or load the same MJCF/MJB used by URLab.
- Load it into classic MuJoCo Python.
- Convert it to MJWarp with `mjw.put_model`.
- Allocate batched data with `mjw.make_data`.
- Run `mjw.step` for batch sizes such as 1, 2, 4, 8, 16, 32, 64, and 128.
- Compare short rollouts against classic MuJoCo CPU.
- Identify unsupported fields, sensors, contacts, solver options, or model
  features.

Success criteria:

- Go2 can step in MJWarp for at least one simple scene.
- Batch scaling is measured on the target NVIDIA GPU.
- Unsupported model features are listed explicitly.
- A CPU-vs-MJWarp rollout comparison exists for a fixed seed and command
  sequence.

### Milestone 3: Standalone mujoco_warp_musa / mujoco_musa Feasibility

Goal: independently evaluate the Moore Threads path without mixing it with the
NVIDIA path.

Scope:

- Build `mujoco_musa` on a Moore Threads machine.
- Run upstream or repository-provided tests.
- Run a Go2-like model if the higher-level `mujoco_warp_musa` stack is
  available.
- Measure batch throughput and compare against CPU MuJoCo.
- Identify whether the library exposes a usable native C++ API or only serves
  the MUSA-adapted higher-level runtime.

Success criteria:

- The Moore Threads stack builds and runs on target hardware.
- Its API layer is understood well enough to decide whether URLab can call it.
- Results are recorded separately from NVIDIA MJWarp results.

### Milestone 4: External GPU Physics Prototype

Goal: connect UE/URLab to an external GPU physics process through the backend
contract, without claiming final integration.

Scope:

- Keep UE as renderer and scene viewer.
- Run MJWarp in a separate Python process.
- Send reset/control commands to the MJWarp process.
- Step MJWarp on GPU.
- Return selected body transforms and observations.
- Push transforms into UE for visualization.
- Use a low render rate at first, such as 10-30 Hz, while physics runs faster.
- Do not build web UI or editor UI.

Success criteria:

- One dog can be simulated by external MJWarp and visualized in UE.
- UE rendering reads the GPU physics result instead of CPU MuJoCo state for
  that prototype path.
- Data copy cost is measured.
- The prototype can be disabled to return to the CPU MuJoCo path.

### Milestone 5: Research Training Runner

Goal: use GPU physics for actual RL-style rollout throughput before trying to
make the UE plugin native.

Scope:

- Build a training runner around the external MJWarp backend.
- Keep physics and policy on GPU when possible.
- Use UE rendering only for sampled visualization or visual observations, not
  necessarily every physics step.
- Support batch resets.
- Support domain-randomized initial states where MJWarp allows it.
- Report reward/sample throughput and not only raw physics steps/s.

Success criteria:

- A locomotion policy can collect batched rollout data through MJWarp.
- Sample throughput exceeds the CPU baseline for a meaningful batch size.
- UE can render selected environments/episodes for debugging.
- The runner does not depend on web UI.

### Milestone 6: URLab Backend Abstraction In UE

Goal: refactor URLab so classic CPU MuJoCo is one backend implementation rather
than an assumption spread everywhere.

Scope:

- Introduce an internal simulation backend interface in the UE plugin.
- Implement `ClassicMuJoCoBackend` by wrapping the existing `mj_step` path.
- Keep public behavior unchanged for CPU MuJoCo.
- Move direct state reads/writes behind backend methods where needed.
- Keep current `mjModel*` / `mjData*` available for the classic backend and
  legacy code until migration is complete.
- Add tests around backend-neutral state extraction.

Success criteria:

- Existing URLab CPU scenes still work.
- Existing bridge and policy scripts still work.
- The new interface can host an external GPU backend without invasive changes
  to higher-level URLab code.

### Milestone 7: Native GPU Backend Decision Gate

Goal: decide whether to build a native UE-callable GPU backend, and which GPU
family to target first.

Decision inputs:

- MJWarp standalone throughput and correctness.
- External backend copy/sync overhead.
- CPU MuJoCo baseline on target scenes.
- Research need for high-throughput batched training.
- Hardware target:
  - NVIDIA CUDA/Warp;
  - Moore Threads MUSA;
  - both, but not in the same first implementation.

Possible outcomes:

- Continue using external MJWarp for training and CPU MuJoCo for UE rendering.
- Build a native NVIDIA backend inspired by MJWarp.
- Build a native Moore Threads backend around `mujoco_warp_musa` /
  `mujoco_musa`.
- Defer native GPU physics because CPU MuJoCo is good enough for the current
  research stage.

Success criteria:

- A written decision records target hardware, expected benefit, risk, and the
  first supported scene/model subset.

### Milestone 8: Native GPU Backend Prototype

Goal: create the smallest native GPU backend that can run one supported robot
and synchronize state to UE.

Scope:

- Support one robot model first, likely Go2.
- Support a minimal subset:
  - reset;
  - control write;
  - step;
  - body transform readback;
  - joint state readback;
  - selected observations.
- Do not support all sensors, UI controls, replay, editor tools, or arbitrary
  scene authoring yet.
- Keep CPU MuJoCo fallback.
- Measure GPU step time, copy time, and render sync time separately.

Success criteria:

- One robot can run through the native GPU backend.
- It can be visualized in UE.
- Its rollout is compared against CPU MuJoCo for a simple command sequence.
- The prototype is isolated behind the backend interface.

### Milestone 9: Batched Native GPU Environments

Goal: scale the native GPU backend from one robot to many environments.

Scope:

- Represent many worlds/environments explicitly.
- Support batched controls, batched reset, and batched observations.
- Keep per-environment state independent.
- Support rendering one or a small subset of environments in UE.
- Avoid copying all GPU state back to CPU every step.
- Add throughput reports for 1, 2, 4, 8, 16, 32, 64+ environments.

Success criteria:

- Aggregate robot-steps/s improves over CPU MuJoCo for the target scene.
- UE can visualize selected environments without stalling the training loop.
- Reset and observation semantics are stable enough for RL experiments.

### Milestone 10: Visual RL Synchronization

Goal: make UE rendering usable for visual observations without forcing every
physics step to wait on a full render frame.

Scope:

- Define when visual frames are sampled relative to physics time.
- Support lower render rate than physics rate.
- Attach frame ids and sim times to rendered observations.
- Measure render readback cost.
- Support selected cameras/environments first.
- Keep non-visual observations available directly from physics backend.

Success criteria:

- A policy/training loop can request visual observations with known freshness.
- Physics can run faster than rendering when visual observations are not needed.
- Render synchronization cost is measured and visible in metrics.

### Milestone 11: Multi-Robot Control Server Integration

Goal: connect GPU backend work with the central robot-control server roadmap.

Scope:

- Let the central control server select backend mode.
- Keep command hub, policy runtime, and metrics backend-agnostic where possible.
- Support CPU MuJoCo and GPU backend metrics in the same reporting structure.
- Preserve current web/keyboard control for CPU mode.
- Add GPU-backend control only after physics/backend semantics are stable.

Success criteria:

- The same high-level command path can drive CPU or GPU backend experiments.
- Metrics show physics backend timing separately from policy and transport
  timing.

### Milestone 12: Research UX And UI

Goal: add UI only after the research backend is useful.

Scope:

- Backend selector.
- Benchmark dashboard.
- Robot/environment status.
- Visual environment picker.
- Basic training-run monitor.
- Links to saved benchmark reports.

Success criteria:

- UI helps inspect an already-working system.
- UI is not required to run benchmarks, training, or backend tests.

## Recommended First Path

The recommended path is:

```text
Milestone 0
  -> Milestone 1
  -> Milestone 2
  -> Milestone 4
  -> Milestone 5
  -> Milestone 6
  -> Milestone 7 decision
```

Do not start by trying to embed MJWarp Python inside UE. That path is attractive
but brittle: Python packaging, CUDA context lifetime, GIL behavior, UE thread
safety, native extension crashes, and state-copy costs all become mixed at
once.

Do not start by rewriting MuJoCo. Use CPU MuJoCo as the correctness baseline,
use MJWarp as the GPU reference/prototype, and only build native GPU backend
pieces after the external prototype proves the target model and throughput.

## Out Of Scope Until Late Milestones

- Full web UI for backend control.
- Editor-facing polished controls.
- Multi-user leasing for GPU training.
- Arbitrary robot families beyond the initial Go2 proof of concept.
- Full feature parity with classic MuJoCo.
- Every MuJoCo sensor type.
- Camera rendering for every environment on every physics step.
- Native GPU backend for NVIDIA and Moore Threads at the same time.

## Open Questions

- Which hardware is the first serious target: NVIDIA only, Moore Threads only,
  or both in separate tracks?
- What is the minimum model subset required for the first successful Go2 GPU
  training experiment?
- Are visual observations required during training from the beginning, or can
  visual rendering be sampled only for debugging at first?
- What rollout divergence from CPU MuJoCo is acceptable for the target research?
- Should the first GPU backend support one large world with many dogs, or many
  independent batched worlds?

