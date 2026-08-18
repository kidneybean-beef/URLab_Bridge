#!/usr/bin/env python3
"""Spawn a grid of Go2 robot Blueprints into the currently loaded URLab level.

Uses ``client.scene.spawn_grid`` (single RPC, server-side batched) to place
``count_x`` * ``count_y`` copies of a Blueprint at ``spacing`` metres apart.
Defaults to 20 robots (5 x 4) at 2 m spacing, referencing the Go2 Blueprint at
``/Game/go2_go2_rl_gym.go2_go2_rl_gym_C``.

PIE must be OFF when scene-authoring ops run (UE holds a lock on the PIE world),
so this script stops PIE first if it is active. It does *not* start PIE back up
afterwards — start it from the dashboard or a follow-up script once you are
happy with the grid.

Example:
    python scripts/spawn_go2_grid.py
    python scripts/spawn_go2_grid.py --count-x 4 --count-y 4 --spacing 1.5
    python scripts/spawn_go2_grid.py --origin 0 0 0.3 --base-id go2_dog
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from urlab_client import URLabClient  # noqa: E402

logger = logging.getLogger("spawn_go2_grid")

DEFAULT_BLUEPRINT = "/Game/go2_go2_rl_gym.go2_go2_rl_gym_C"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--address", default="tcp://127.0.0.1", help="URLab server address (default: tcp://127.0.0.1)")
    p.add_argument("--step-port", type=int, default=5559, help="Step/RPC port (default: 5559)")
    p.add_argument("--blueprint", default=DEFAULT_BLUEPRINT,
                   help=f"Blueprint class path (default: {DEFAULT_BLUEPRINT})")
    p.add_argument("--base-id", default="go2", help="Base actor id; per-cell ids are '<base>_<i>_<j>' (default: go2)")
    p.add_argument("--count-x", type=int, default=5, help="Grid cells along X (default: 5)")
    p.add_argument("--count-y", type=int, default=4, help="Grid cells along Y (default: 4)")
    p.add_argument("--spacing", type=float, nargs=2, default=(2.0, 2.0), metavar=("SX", "SY"),
                   help="Spacing in metres (X Y) (default: 2.0 2.0)")
    p.add_argument("--origin", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                   help="Grid origin in metres (default: 0 0 0)")
    p.add_argument("--yaw-deg", type=float, default=0.0, help="Yaw applied to every robot (deg, default: 0)")
    p.add_argument("--save-level", action="store_true", help="Save the level after spawning")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return p.parse_args()


def _ensure_pie_stopped(client: URLabClient) -> None:
    status = client.sim.status()
    if getattr(status, "is_running", False):
        logger.info("PIE is running; stopping so scene-authoring can proceed")
        client.sim.stop()


def _log_summary(handles: dict, origin: Sequence[float], spacing: Tuple[float, float]) -> None:
    if not handles:
        logger.warning("spawn_grid returned no handles")
        return
    reused = sum(1 for h in handles.values() if getattr(h, "was_existing", False))
    fresh = len(handles) - reused
    bp = next(iter(handles.values())).blueprint_class_path
    logger.info("Spawned %d actors (%d new, %d updated) of %s", len(handles), fresh, reused, bp)
    logger.info(
        "Grid: %dx%d, spacing=(%.3f, %.3f) m, origin=(%.3f, %.3f, %.3f) m",
        max((int(k.split("_")[-2]) for k in handles), default=0) + 1,
        max((int(k.split("_")[-1]) for k in handles), default=0) + 1,
        spacing[0], spacing[1], origin[0], origin[1], origin[2],
    )
    for aid in sorted(handles):
        h = handles[aid]
        loc = h.location
        logger.debug("  %-20s loc=(%.3f, %.3f, %.3f)  actor=%s", aid, loc[0], loc[1], loc[2], h.actor_name)
    if any(getattr(h, "requires_pie_restart", False) for h in handles.values()):
        logger.warning("At least one actor reports requires_pie_restart=True; restart PIE before stepping")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.count_x <= 0 or args.count_y <= 0:
        raise SystemExit("count_x and count_y must be positive")
    total = args.count_x * args.count_y
    if total > 1024:
        raise SystemExit(f"spawn_grid caps at 1024 cells; requested {total}")

    rotation_euler = (0.0, 0.0, args.yaw_deg) if args.yaw_deg else None

    client = URLabClient(
        args.address,
        step_port=args.step_port,
        mujoco_version_check=False,
        recv_timeout_ms=120_000,
    )
    try:
        logger.info("Connecting to %s:%d ...", args.address, args.step_port)
        client.connect()
        logger.info("Current level: %s", client.scene.current_level())

        _ensure_pie_stopped(client)

        logger.info(
            "Spawning %dx%d = %d '%s' actors at spacing=(%.2f, %.2f) m from origin=(%.2f, %.2f, %.2f) m",
            args.count_x, args.count_y, total, args.blueprint,
            args.spacing[0], args.spacing[1],
            args.origin[0], args.origin[1], args.origin[2],
        )
        handles = client.scene.spawn_grid(
            blueprint=args.blueprint,
            base_actor_id=args.base_id,
            count_x=args.count_x,
            count_y=args.count_y,
            spacing=(args.spacing[0], args.spacing[1], 0.0),
            origin=tuple(args.origin),
            rotation_euler=rotation_euler,
        )
        _log_summary(handles, args.origin, tuple(args.spacing))

        if args.save_level:
            logger.info("Saving level ...")
            client.scene.save_level()
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
