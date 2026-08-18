import time

from urlab_client import ControlMode, URLabClient


WARMUP_STEPS = 1_000
TIMED_STEPS = 20_000

client = URLabClient(
    "tcp://127.0.0.1",
    step_mode="direct",
    transport="zmq",
    auto_promote_step_mode=True,
    recv_timeout_ms=5_000,
)
client.connect(observations="minimal")

if len(client.articulations) != 1:
    raise RuntimeError(
        f"expected exactly one articulation, got {list(client.articulations)}"
    )

prefix, go2 = next(iter(client.articulations.items()))

model = client.model
actual = (model.nq, model.nv, model.nu, model.ngeom)
expected = (19, 18, 12, 61)
if actual != expected:
    raise RuntimeError(
        f"model does not match standalone Go2: actual={actual}, expected={expected}"
    )

# URLab may place world geometry under an additional static wrapper body.
# It has no joint or mass, so it does not add simulation degrees of freedom.
static_wrapper_bodies = [
    body_id
    for body_id in range(1, model.nbody)
    if model.body_jntnum[body_id] == 0 and model.body_mass[body_id] == 0.0
]
print(
    f"model: nq={model.nq} nv={model.nv} nu={model.nu} "
    f"nbody={model.nbody} ngeom={model.ngeom}"
)
print(f"static_wrapper_bodies={static_wrapper_bodies}")

client.runtime.set_sim_options(
    timestep=0.002,
    iterations=100,
    ls_iterations=50,
    integrator="euler",
    solver="newton",
    cone="pyramidal",
    num_worker_threads=0,
)

# Bypass MjPDController and send zero motor torque.
go2.control_mode = ControlMode.RAW
go2.ctrl_array[:] = 0.0

# mj_resetData: model qpos0, zero qvel, zero ctrl.
client.reset()

# Same untimed zero-torque warmup as standalone.
warmup = client.step(
    n_steps=WARMUP_STEPS,
    observations="minimal",
    control_articulations=(prefix,),
)

start_step = int(warmup["step"])
start_sim_time = float(warmup["time"])

t0 = time.perf_counter()
result = client.step(
    n_steps=TIMED_STEPS,
    observations="minimal",
    control_articulations=(prefix,),
)
wall_s = time.perf_counter() - t0

completed_steps = int(result["step"]) - start_step
sim_s = float(result["time"]) - start_sim_time

print(f"articulation={prefix}")
print(f"wall_s={wall_s:.6f}")
print(f"physics_steps={completed_steps}")
print(f"steps_per_s={completed_steps / wall_s:.1f}")
print(f"sim_s={sim_s:.6f}")
print(f"real_time_factor={sim_s / wall_s:.2f}x")

client.close()
