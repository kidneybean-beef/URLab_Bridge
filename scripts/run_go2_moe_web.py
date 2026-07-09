#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import sys
import threading
from http.server import ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
for _path in (_HERE, _SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from run_go2_moe_apply import resolve_limit_mode  # noqa: E402
from run_go2_moe_keyboard import (  # noqa: E402
    build_arg_parser as build_keyboard_arg_parser,
    run_keyboard_policy,
)
from urlab_bridge.web_control import (  # noqa: E402
    WebCommandSource,
    WebTwistConfig,
    make_handler,
)
from urlab_policy.go2.twist_commands import KeyboardCommandConfig  # noqa: E402

logger = logging.getLogger("run_go2_moe_web")


def build_arg_parser():
    parser = build_keyboard_arg_parser()
    parser.description = (
        "Drive the Unitree RL Gym Go2 CTS/MoE policy from a local browser. "
        "This owns the URLabClient and serves the webpage in the same process, "
        "so browser commands feed the policy without a second URLab RPC session."
    )
    parser.set_defaults(max_vx=1.0, max_vy=0.5, max_yaw=1.57)
    parser.add_argument("--web-bind", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8088)
    parser.add_argument("--web-stale-timeout-s", type=float, default=0.5)
    parser.add_argument("--dash-max-vx", type=float, default=2.0)
    parser.add_argument("--dash-max-vy", type=float, default=1.0)
    parser.add_argument("--dash-max-yaw", type=float, default=3.14)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    limit_mode = resolve_limit_mode(args)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    web_config = WebTwistConfig(
        max_vx=args.max_vx,
        max_vy=args.max_vy,
        max_yaw=args.max_yaw,
        dash_max_vx=args.dash_max_vx,
        dash_max_vy=args.dash_max_vy,
        dash_max_yaw=args.dash_max_yaw,
    )
    command_source = WebCommandSource(
        config=web_config,
        stale_timeout_s=args.web_stale_timeout_s,
    )
    server = ThreadingHTTPServer(
        (args.web_bind, int(args.web_port)),
        make_handler(command_source),
    )
    stop_event = threading.Event()
    watchdog_interval_s = max(
        0.05,
        min(0.25, args.web_stale_timeout_s / 2.0 if args.web_stale_timeout_s else 0.05),
    )

    def watchdog() -> None:
        while not stop_event.wait(watchdog_interval_s):
            command_source.stop_if_stale()

    web_thread = threading.Thread(target=server.serve_forever, daemon=True)
    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    web_thread.start()
    watchdog_thread.start()
    logger.info("open http://%s:%d and focus the page", args.web_bind, args.web_port)

    key_config = KeyboardCommandConfig(
        step_vx=args.step_vx,
        step_vy=args.step_vy,
        step_yaw=args.step_yaw,
        max_vx=args.max_vx,
        max_vy=args.max_vy,
        max_yaw=args.max_yaw,
    )
    try:
        return run_keyboard_policy(args, command_source, limit_mode, key_config)
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()
        web_thread.join(timeout=2.0)
        watchdog_thread.join(timeout=1.0)
        command_source.close()


if __name__ == "__main__":
    sys.exit(main())
