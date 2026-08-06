# City-Scale Hybrid Simulation Platform Roadmap

## Purpose

This document summarizes the long-term simulation-platform direction discussed
for URLab, CARLA/Unreal Engine, MuJoCo CPU, MuJoCo Warp, MJX-JAX, and
MUSA-based MuJoCo variants. The target is a city-scale platform that supports
both validation and training across embodied robots, cars, pedestrians,
rule-based agents, learned agents, world models, and city-operation policies.

This is a roadmap-level document, not a concrete implementation plan. Each
major track should later be expanded into its own milestone plan.

Related documents:

- `docs/superpowers/specs/2026-07-09-central-robot-control-server-roadmap.md`
- `docs/superpowers/specs/2026-07-13-gpu-urlab-roadmap.md`
- `docs/superpowers/specs/2026-07-16-city-scale-hybrid-simulation-platform-roadmap.drawio`

## Core Direction

The platform should be CARLA/Unreal-native at the city level, with URLab and
MuJoCo acting as robot-physics extensions rather than as the only simulation
runtime.

Recommended high-level architecture:

```text
CARLA/Unreal City Runtime
  city map, roads, traffic lights, weather, cameras, sensors, replay

URLab Robot Extension
  articulated robot actors, robot state sync, CPU MuJoCo validation backend

External GPU Physics/Training Backends
  MuJoCo Warp for NVIDIA GPU batched robot training
  MJX-JAX for selected differentiable research
  mujoco_warp_musa / mujoco_musa for Moore Threads hardware experiments

Scenario Orchestrator
  resets, seeds, incidents, city events, route tasks, rare-event search

Canonical CityState / Data Contract
  backend-neutral runtime state and logged training/validation data

Experiment/Data Layer
  datasets, metrics, replay, rare-case mining, world-model training data
```

The platform should not force every task into one real-time UE loop. Instead,
each task should use the cheapest loop that preserves the phenomenon being
studied, then validate important cases in the richer CARLA/UE city runtime.

## Key Architecture Choice

Keep two physics paths:

```text
In-UE CPU MuJoCo
  validation, debugging, small shared scenes, interactive PIE inspection

External MuJoCo Warp / MJX / MUSA backend
  high-throughput training, batched isolated environments, research loops
```

This hybrid split avoids embedding all Python/GPU training complexity directly
inside UE while preserving URLab's useful in-editor validation path.

## CARLA/UE Role

CARLA/UE should be the city operating system:

- city-scale maps and assets;
- vehicles, traffic lights, weather, sensors, replay;
- pedestrians and custom rule-based city agents;
- visual rendering for validation, perception, VLA, and dataset generation;
- scenario playback for discovered rare events;
- final inspection environment for trained policies.

CARLA should remain the main car-simulation path. MuJoCo should not replace
CARLA vehicle simulation. URLab/MuJoCo should extend CARLA/UE with articulated
robots.

## MuJoCo Backend Roles

### CPU MuJoCo

Use CPU MuJoCo as the correctness baseline and validation backend:

- interactive debugging;
- small shared physical scenes;
- rollout comparison against GPU backends;
- contact and robot-control diagnosis;
- fallback when GPU backend does not support a feature.

### MuJoCo Warp

Use MuJoCo Warp as the default NVIDIA GPU robot-training backend:

- many isolated robot worlds;
- locomotion and terrain robustness;
- low-level control pretraining;
- batched robot rollouts;
- selected visual-observation requests through CARLA/UE when needed.

MuJoCo Warp is strongest when the batch dimension is many independent worlds,
not one giant world containing many interacting robots.

### MJX-JAX

Use MJX-JAX selectively for differentiable research:

- small differentiable control experiments;
- system-identification studies;
- model-based optimization research;
- not the default locomotion or city-scale training backend.

### mujoco_warp_musa / mujoco_musa

Keep the Moore Threads path separate from the NVIDIA path:

- `mujoco_musa` is the lower-level MUSA kernel layer;
- `mujoco_warp_musa` is the higher-level MJWarp-style package for Moore
  Threads hardware;
- use this only when targeting Moore Threads GPUs.

## Training And Validation Paths By Task

### Robot Locomotion

```text
MuJoCo Warp batched training
  -> CPU MuJoCo rollout comparison
  -> CARLA/UE city validation
  -> real robot / log comparison when available
```

UE rendering is not needed in the inner loop unless the task uses visual
observations.

### Terrain Robustness

```text
MuJoCo Warp randomized terrain and dynamics
  -> selected terrain patches rendered in CARLA/UE
  -> failure replay and policy diagnosis
```

### Robot Navigation

Use layered control:

```text
low-level locomotion policy from MuJoCo Warp
high-level navigation policy/planner from city scenarios
CARLA/UE validation with maps, traffic, pedestrians, and sensors
```

Vision-based navigation should use CARLA/UE as a render/sensor server at a
lower rate than physics.

### VLA And Vision-Based Robot Policies

VLA requires UE/CARLA visual observations, but physics still does not have to
live inside UE.

```text
MuJoCo physics state
  -> selected CARLA/UE render request
  -> RGB/depth/segmentation/language observation
  -> VLA policy action
  -> low-level controller
```

Render fewer environments than the physics batch, and render at the VLA decision
rate rather than at every physics substep.

### Robot World Models

World models should consume data from all loops:

```text
MuJoCo Warp rollouts
CARLA/UE visual rollouts
CPU MuJoCo validation rollouts
real logs when available
```

Standard training tuples should include:

```text
env_id, timestamp, robot_state, action, proprioception, images, depth,
segmentation, map patch, language instruction, reward/event labels,
next_state, done/reset reason
```

### Autonomous Driving And Car Simulation

Use CARLA as the primary training/validation backend:

- ego vehicle behavior;
- camera/LiDAR/radar sensor suites;
- traffic-light interaction;
- replay and scenario validation;
- learned driving policies and rule-based traffic participants.

MuJoCo is not the natural backend for city vehicle simulation.

### Pedestrians And Rule-Based City Agents

Start inside CARLA/UE:

- CARLA walkers where useful;
- custom UE AI/rule agents for city operations;
- learned crowd or pedestrian models later;
- incidents and abnormal behavior injected by the scenario orchestrator.

### Perception-Only Training

Use CARLA/UE heavily:

- object detection;
- segmentation;
- depth;
- occupancy;
- semantic maps;
- synthetic data and domain randomization.

Physics can be simplified or replay-based for this track.

### Multi-Agent And Emergence Studies

Use a two-stage loop:

```text
large batches of cheap/headless simulation
  -> rare-event mining
  -> high-fidelity CARLA/UE replay
  -> policy or city-operation adjustment
```

Do not rely only on high-fidelity real-time simulation to discover rare cases.
The platform should run many seeds, demand profiles, weather conditions, agent
mixes, and incident schedules, then replay the interesting cases.

## Canonical Data Contract

The platform needs one shared runtime/logging contract so each backend can
participate without becoming tightly coupled to another engine.

Core fields:

```text
world_id
env_id
actor_id
actor_type
timestamp
pose
velocity
joint_state
control_command
sensor_request
sensor_result
map_reference
route_reference
scenario_event
reward
done
reset_reason
backend_name
frame_id
```

The contract should support both live synchronization and offline replay.

## Roadmap Tracks

### Track A: CARLA/UE City Host

Goal: make CARLA/UE the shared city runtime for validation, rendering, sensors,
scenario playback, and city actors.

### Track B: URLab Robot Extension

Goal: run URLab robots inside the CARLA/UE world, preserve CPU MuJoCo validation,
and expose robot state/actions through the canonical contract.

### Track C: GPU Robot Training

Goal: train articulated robot policies with MuJoCo Warp using many isolated
parallel worlds, then validate selected policies in CARLA/UE.

### Track D: Scenario Orchestrator

Goal: define scenarios, resets, random seeds, routes, incidents, weather,
traffic mixes, pedestrians, robot tasks, and rare-event search.

### Track E: Data And World Models

Goal: record all simulation loops into reusable datasets for world models,
perception, VLA, navigation, city-operation analysis, and replay.

### Track F: City Operation And Emergence

Goal: run large experiment sweeps, discover extreme cases, test policies, and
use high-fidelity replay to understand why failures or emergent behaviors
occurred.

## Near-Term Recommended Sequence

1. Define the canonical `CityState` and experiment log schema.
2. Prototype CARLA/UE plus URLab robot coexistence.
3. Keep CPU MuJoCo as the in-UE validation backend.
4. Build an external MuJoCo Warp robot-training backend.
5. Add a state bridge from external robot physics to CARLA/UE visual actors.
6. Add CARLA/UE render requests for selected visual-observation tasks.
7. Add scenario orchestration and replay.
8. Add dataset generation and rare-event mining.
9. Train task-specific policies and world models from the shared data layer.
10. Validate discovered policies and rare cases inside CARLA/UE.

## Diagram

The accompanying draw.io roadmap is:

```text
docs/superpowers/specs/2026-07-16-city-scale-hybrid-simulation-platform-roadmap.drawio
```

The diagram places the central platform components in the middle and branches
outward to the major task tracks: locomotion, terrain robustness, robot
navigation, VLA, world models, autonomous driving, pedestrians/rule agents,
perception, multi-agent emergence, and city-operation policy.
