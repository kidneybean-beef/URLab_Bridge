# Copyright (c) 2026 Jonathan Embley-Riches. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`urlab-web-control` — browser keyboard control for one twist robot."""

from __future__ import annotations

import argparse
import logging
import sys

from .common import setup_logging
from ..web_control import WebTwistConfig, run_server

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urlab-web-control",
        description="Serve a local browser keyboard controller for one URLab twist articulation.",
    )
    parser.add_argument("--address", default="tcp://127.0.0.1")
    parser.add_argument("--step-port", type=int, default=5559)
    parser.add_argument("--articulation", default="go2")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--stale-timeout-s", type=float, default=0.5)
    parser.add_argument("--max-vx", type=float, default=1.0)
    parser.add_argument("--max-vy", type=float, default=0.5)
    parser.add_argument("--max-yaw", type=float, default=1.57)
    parser.add_argument("--dash-max-vx", type=float, default=2.0)
    parser.add_argument("--dash-max-vy", type=float, default=1.0)
    parser.add_argument("--dash-max-yaw", type=float, default=3.14)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    from urlab_client import URLabClient

    client = URLabClient(
        args.address,
        step_mode="live",
        step_port=args.step_port,
        mujoco_version_check=False,
        local_model=False,
    )
    try:
        logger.info("connecting to URLab step server at %s:%d", args.address, args.step_port)
        client.connect(observations="standard")
        if not client.manager_present:
            raise SystemExit(
                "URLab manager is not present. Start PIE before launching web control."
            )
        if args.articulation not in client.articulations:
            available = ", ".join(sorted(client.articulations)) or "<none>"
            raise SystemExit(
                f"articulation {args.articulation!r} not found; available: {available}"
            )

        config = WebTwistConfig(
            max_vx=args.max_vx,
            max_vy=args.max_vy,
            max_yaw=args.max_yaw,
            dash_max_vx=args.dash_max_vx,
            dash_max_vy=args.dash_max_vy,
            dash_max_yaw=args.dash_max_yaw,
        )
        logger.info(
            "open http://%s:%d and focus the page; articulation=%s",
            args.bind,
            args.port,
            args.articulation,
        )
        run_server(
            client,
            articulation=args.articulation,
            bind=args.bind,
            port=args.port,
            config=config,
            stale_timeout_s=args.stale_timeout_s,
        )
    except KeyboardInterrupt:
        logger.info("stopping web control")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
