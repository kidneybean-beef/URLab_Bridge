import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_client import URLabClient

ADDRESS = "tcp://127.0.0.1"
BATCH_STEPS = 200
BATCHES = 100

client = URLabClient(
    ADDRESS,
    step_mode="direct",
    transport="zmq",
    auto_promote_step_mode=True,
)
client.connect()

# Try different values later: 0, 4, 8, 16, 32.
client.runtime.set_sim_options(num_worker_threads=0, required=False)

for _ in range(5):
    client.step(n_steps=BATCH_STEPS, observations="minimal")

t0 = time.perf_counter()
start = client.step(n_steps=1, observations="minimal").get("time", 0.0)

for _ in range(BATCHES):
    client.step(n_steps=BATCH_STEPS, observations="minimal")

end = client.step(n_steps=1, observations="minimal").get("time", 0.0)
wall = time.perf_counter() - t0

steps = BATCH_STEPS * BATCHES
print(f"wall_s={wall:.3f}")
print(f"physics_steps={steps}")
print(f"steps_per_s={steps / wall:.1f}")
if end > start:
    print(f"sim_s={end - start:.3f}")
    print(f"real_time_factor={(end - start) / wall:.2f}x")

client.close()
