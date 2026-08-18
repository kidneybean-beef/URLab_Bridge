#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ImageTrafficReport:
    elapsed_s: float
    messages: int
    bytes_total: int
    hz: float
    mb_s: float
    mean_message_mb: float
    min_period_s: float | None
    max_period_s: float | None
    std_period_s: float | None


@dataclass
class ImageTrafficStats:
    started_at: float
    messages: int = 0
    bytes_total: int = 0
    last_message_at: float | None = None
    periods: list[float] = field(default_factory=list)

    def record(self, *, now: float, byte_count: int) -> None:
        now = float(now)
        if self.last_message_at is not None:
            period = max(0.0, now - self.last_message_at)
            self.periods.append(period)
        self.last_message_at = now
        self.messages += 1
        self.bytes_total += max(0, int(byte_count))

    def reset(self, *, now: float) -> None:
        self.started_at = float(now)
        self.messages = 0
        self.bytes_total = 0
        self.last_message_at = None
        self.periods.clear()

    def report(self, *, now: float) -> ImageTrafficReport:
        elapsed_s = max(0.0, float(now) - float(self.started_at))
        hz = self.messages / elapsed_s if elapsed_s > 0.0 else 0.0
        mb_s = (self.bytes_total / (1024.0 * 1024.0)) / elapsed_s if elapsed_s > 0.0 else 0.0
        mean_message_mb = (
            (self.bytes_total / self.messages) / (1024.0 * 1024.0)
            if self.messages > 0
            else 0.0
        )
        if self.periods:
            mean_period = sum(self.periods) / len(self.periods)
            variance = sum((value - mean_period) ** 2 for value in self.periods) / len(self.periods)
            min_period_s = min(self.periods)
            max_period_s = max(self.periods)
            std_period_s = math.sqrt(variance)
        else:
            min_period_s = None
            max_period_s = None
            std_period_s = None
        return ImageTrafficReport(
            elapsed_s=elapsed_s,
            messages=self.messages,
            bytes_total=self.bytes_total,
            hz=hz,
            mb_s=mb_s,
            mean_message_mb=mean_message_mb,
            min_period_s=min_period_s,
            max_period_s=max_period_s,
            std_period_s=std_period_s,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure ROS2 sensor_msgs/Image receive Hz and bandwidth in one subscriber.",
    )
    parser.add_argument("topic", help="sensor_msgs/Image topic to subscribe to")
    parser.add_argument(
        "--message-type",
        choices=("auto", "image", "compressed_image"),
        default="auto",
        help="ROS2 message type; auto treats topics ending in /compressed as sensor_msgs/CompressedImage",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to measure; 0 runs until Ctrl-C")
    parser.add_argument("--report-interval-s", type=float, default=1.0, help="seconds between live reports")
    parser.add_argument(
        "--qos-reliability",
        choices=("best_effort", "reliable"),
        default="best_effort",
        help="subscriber reliability QoS",
    )
    parser.add_argument(
        "--qos-history",
        choices=("keep_last",),
        default="keep_last",
        help="subscriber history QoS",
    )
    parser.add_argument("--qos-depth", type=int, default=1, help="subscriber queue depth")
    parser.add_argument("--node-name", default="urlab_image_bench", help="ROS2 node name")
    return parser


def _load_ros2() -> tuple[Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import CompressedImage, Image
    except ImportError as exc:
        raise SystemExit(
            "ROS2 image benchmark requires rclpy and sensor_msgs. "
            "Source your ROS2 setup first, for example `source /opt/ros/jazzy/setup.bash`."
        ) from exc
    return (
        rclpy,
        {"image": Image, "compressed_image": CompressedImage},
        SingleThreadedExecutor,
        (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy),
    )


def _resolve_message_type(raw: str, topic: str) -> str:
    if raw != "auto":
        return raw
    return "compressed_image" if str(topic).endswith("/compressed") else "image"


def _build_qos(args: argparse.Namespace, qos_types: Any) -> Any:
    qos_profile_type, history_policy, reliability_policy = qos_types
    reliability = (
        reliability_policy.BEST_EFFORT
        if args.qos_reliability == "best_effort"
        else reliability_policy.RELIABLE
    )
    return qos_profile_type(
        history=history_policy.KEEP_LAST,
        depth=max(1, int(args.qos_depth)),
        reliability=reliability,
    )


def _format_report(prefix: str, report: ImageTrafficReport) -> str:
    min_period = _format_optional_seconds(report.min_period_s)
    max_period = _format_optional_seconds(report.max_period_s)
    std_period = _format_optional_seconds(report.std_period_s)
    return (
        f"{prefix}: hz={report.hz:.3f} mib_s={report.mb_s:.3f} "
        f"messages={report.messages} mean_msg_mib={report.mean_message_mb:.3f} "
        f"min_period={min_period} max_period={max_period} std_period={std_period}"
    )


def _format_optional_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}s"


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.duration < 0.0:
        raise SystemExit("--duration must not be negative")
    if args.report_interval_s <= 0.0:
        raise SystemExit("--report-interval-s must be greater than zero")

    message_type_name = _resolve_message_type(args.message_type, args.topic)
    rclpy, message_types, executor_type, qos_types = _load_ros2()
    image_type = message_types[message_type_name]
    rclpy.init(args=None)
    node = rclpy.create_node(args.node_name)
    started_at = time.monotonic()
    total_stats = ImageTrafficStats(started_at=started_at)
    window_stats = ImageTrafficStats(started_at=started_at)

    def callback(msg: Any) -> None:
        now = time.monotonic()
        byte_count = len(getattr(msg, "data", b"") or b"")
        total_stats.record(now=now, byte_count=byte_count)
        window_stats.record(now=now, byte_count=byte_count)

    def report_window() -> None:
        now = time.monotonic()
        print(_format_report("window", window_stats.report(now=now)), flush=True)
        window_stats.reset(now=now)

    node.create_subscription(image_type, args.topic, callback, _build_qos(args, qos_types))
    node.create_timer(args.report_interval_s, report_window)
    print(
        "subscribed: "
        f"topic={args.topic} reliability={args.qos_reliability} "
        f"history={args.qos_history} depth={args.qos_depth} "
        f"message_type={message_type_name}"
    )
    executor = executor_type()
    executor.add_node(node)
    try:
        while True:
            executor.spin_once(timeout_sec=0.1)
            now = time.monotonic()
            if args.duration > 0.0 and now - started_at >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        final_report = total_stats.report(now=time.monotonic())
        print(_format_report("final", final_report), flush=True)
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
