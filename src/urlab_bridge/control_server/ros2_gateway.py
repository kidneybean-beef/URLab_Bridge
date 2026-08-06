from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from .cameras import linear_rgb_to_srgb_u8
from .commands import CommandHub
from .models import WebPolicyTarget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ros2MessageTypes:
    Twist: Any
    JointState: Any | None = None
    Odometry: Any | None = None
    Float64MultiArray: Any | None = None
    Image: Any | None = None
    CameraInfo: Any | None = None
    CameraQoS: Any | None = None


def _load_ros2() -> tuple[Any, Ros2MessageTypes]:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from std_msgs.msg import Float64MultiArray

    camera_qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )

    return rclpy, Ros2MessageTypes(
        Twist=Twist,
        JointState=JointState,
        Odometry=Odometry,
        Float64MultiArray=Float64MultiArray,
        Image=Image,
        CameraInfo=CameraInfo,
        CameraQoS=camera_qos,
    )


class Ros2Gateway:
    def __init__(
        self,
        *,
        targets: Sequence[WebPolicyTarget],
        command_hub: CommandHub,
        node_name: str = "urlab_go2_control_server",
        subscribe_cmd_vel: bool = True,
        publish_state: bool = False,
        publish_sensors: bool = False,
        publish_cameras: bool = False,
        state_hz: float | None = None,
        camera_fps: float | None = None,
        camera_streams: Mapping[str, Any] | None = None,
        ros2_loader: Callable[[], tuple[Any, Any]] = _load_ros2,
        thread_factory: Callable[..., Any] = threading.Thread,
        camera_thread_factory: Callable[..., Any] = threading.Thread,
        monotonic: Callable[[], float] = time.monotonic,
        log: logging.Logger = logger,
    ) -> None:
        self.targets = tuple(targets)
        self.command_hub = command_hub
        self.node_name = str(node_name)
        self.subscribe_cmd_vel = bool(subscribe_cmd_vel)
        self.publish_state = bool(publish_state)
        self.publish_sensors = bool(publish_sensors)
        self.publish_cameras = bool(publish_cameras)
        self.state_hz = float(state_hz) if state_hz is not None else None
        if self.state_hz is not None and self.state_hz <= 0.0:
            raise ValueError("ROS2 state rate must be greater than zero")
        self.camera_fps = float(camera_fps) if camera_fps is not None else 20.0
        if self.camera_fps <= 0.0:
            raise ValueError("ROS2 camera FPS must be greater than zero")
        self.camera_streams = dict(camera_streams or {})
        self._ros2_loader = ros2_loader
        self._thread_factory = thread_factory
        self._camera_thread_factory = camera_thread_factory
        self._monotonic = monotonic
        self._logger = log
        self._rclpy: Any | None = None
        self._node: Any | None = None
        self._types: Ros2MessageTypes | SimpleNamespace | None = None
        self._executor: Any | None = None
        self._thread: Any | None = None
        self._camera_thread: Any | None = None
        self._camera_stop_event = threading.Event()
        self._initialized_rclpy = False
        self._subscriptions: list[Any] = []
        self._publishers: dict[tuple[Any, str], Any] = {}
        self._last_state_publish_at: float | None = None
        self._camera_source_keys: dict[tuple[str, str], tuple[int, int]] = {}
        self._seen_commands: set[str] = set()

    def start(self) -> None:
        if self._node is not None:
            return
        try:
            rclpy, message_types = self._ros2_loader()
        except ImportError as exc:
            raise SystemExit(
                "ROS2 gateway requires rclpy and standard ROS2 message packages. "
                "Run `source /opt/ros/humble/setup.bash` or source a compatible "
                "ROS2 workspace before starting ROS2 bridge features."
            ) from exc

        self._rclpy = rclpy
        self._types = self._coerce_message_types(message_types)
        ok = getattr(rclpy, "ok", None)
        if not callable(ok) or not ok():
            rclpy.init(args=None)
            self._initialized_rclpy = True

        self._node = rclpy.create_node(self.node_name)
        if self.subscribe_cmd_vel:
            for target in self.targets:
                port = self.command_hub.port(target.articulation, source="ros2")
                topic = f"/{target.articulation}/cmd_vel"
                self._subscriptions.append(
                    self._node.create_subscription(
                        self._types.Twist,
                        topic,
                        self._make_twist_callback(target.articulation, port),
                        10,
                    )
                )
                self._logger.info("ROS2 cmd_vel subscribed: %s", topic)

        spin_target = self._make_spin_target(rclpy, self._node)
        self._thread = self._thread_factory(target=spin_target, daemon=True)
        self._thread.start()
        if self.publish_cameras and self.camera_streams:
            self._camera_stop_event.clear()
            self._camera_thread = self._camera_thread_factory(
                target=self._run_camera_publisher,
                daemon=True,
            )
            self._camera_thread.start()

    def close(self) -> None:
        self._camera_stop_event.set()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=2.0)
            self._camera_thread = None

        executor = self._executor
        if executor is not None:
            shutdown = getattr(executor, "shutdown", None)
            if callable(shutdown):
                shutdown()
            self._executor = None

        node = self._node
        if node is not None:
            destroy = getattr(node, "destroy_node", None)
            if callable(destroy):
                destroy()
            self._node = None

        rclpy = self._rclpy
        if rclpy is not None and self._initialized_rclpy:
            ok = getattr(rclpy, "ok", None)
            if not callable(ok) or ok():
                rclpy.shutdown()
            self._initialized_rclpy = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def publish_control_states(self, states: Sequence[Any]) -> None:
        if self._node is None or self._types is None:
            return
        if not self.publish_state and not self.publish_sensors:
            return
        if not self._should_publish_state_now():
            return

        stamp = self._stamp()
        for state in states:
            articulation = _state_articulation_name(state)
            art = state.art
            if self.publish_state:
                self._publish_joint_state(articulation, art, stamp)
                self._publish_odometry(articulation, art, stamp)
            if self.publish_sensors:
                self._publish_sensors(articulation, art)

    def publish_camera_frames_once(self) -> int:
        if self._node is None or self._types is None or not self.publish_cameras:
            return 0
        stamp = self._stamp()
        published = 0
        for articulation, robot_cameras in self.camera_streams.items():
            streams = getattr(robot_cameras, "streams", {})
            for camera_name, stream in streams.items():
                view = getattr(stream, "view", None)
                if view is None:
                    continue
                frame = getattr(view, "latest_frame", None)
                if frame is None:
                    continue
                source_key = (int(getattr(view, "frame_count", 0)), id(frame))
                camera_key = (str(articulation), str(camera_name))
                if self._camera_source_keys.get(camera_key) == source_key:
                    continue
                image = self._build_image_message(
                    articulation=str(articulation),
                    camera_name=str(camera_name),
                    view=view,
                    frame=np.asarray(frame),
                    stamp=stamp,
                )
                camera_info = self._build_camera_info_message(
                    articulation=str(articulation),
                    camera_name=str(camera_name),
                    view=view,
                    stamp=stamp,
                )
                base = f"/{articulation}/camera/{camera_name}"
                camera_qos = self._camera_qos()
                self._publisher(f"{base}/image_raw", self._require_type("Image"), qos=camera_qos).publish(image)
                self._publisher(f"{base}/camera_info", self._require_type("CameraInfo"), qos=camera_qos).publish(
                    camera_info
                )
                self._camera_source_keys[camera_key] = source_key
                published += 1
        return published

    def _make_twist_callback(self, articulation: str, port: Any) -> Callable[[Any], None]:
        def callback(msg: Any) -> None:
            twist = (
                float(msg.linear.x),
                float(msg.linear.y),
                float(msg.angular.z),
            )
            port.apply_twist(twist)
            if articulation not in self._seen_commands:
                self._seen_commands.add(articulation)
                self._logger.info("ROS2 cmd_vel active for %s: %s", articulation, twist)
            else:
                self._logger.debug("ROS2 cmd_vel for %s: %s", articulation, twist)

        return callback

    def _publish_joint_state(self, articulation: str, art: Any, stamp: Any) -> None:
        msg = self._require_type("JointState")()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{articulation}/base"
        names, positions, velocities = _joint_state_arrays(art)
        msg.name = names
        msg.position = positions
        msg.velocity = velocities
        msg.effort = []
        self._publisher(f"/{articulation}/joint_states", self._require_type("JointState")).publish(msg)

    def _publish_odometry(self, articulation: str, art: Any, stamp: Any) -> None:
        msg = self._require_type("Odometry")()
        msg.header.stamp = stamp
        msg.header.frame_id = "urlab_world"
        msg.child_frame_id = f"{articulation}/base"
        pos = np.asarray(getattr(art, "root_pos_w"), dtype=np.float64)
        quat = np.asarray(getattr(art, "root_quat_xyzw"), dtype=np.float64)
        lin = np.asarray(getattr(art, "root_lin_vel_w"), dtype=np.float64)
        ang = np.asarray(getattr(art, "root_ang_vel_w"), dtype=np.float64)
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.x = float(quat[0])
        msg.pose.pose.orientation.y = float(quat[1])
        msg.pose.pose.orientation.z = float(quat[2])
        msg.pose.pose.orientation.w = float(quat[3])
        msg.twist.twist.linear.x = float(lin[0])
        msg.twist.twist.linear.y = float(lin[1])
        msg.twist.twist.linear.z = float(lin[2])
        msg.twist.twist.angular.x = float(ang[0])
        msg.twist.twist.angular.y = float(ang[1])
        msg.twist.twist.angular.z = float(ang[2])
        self._publisher(f"/{articulation}/odom", self._require_type("Odometry")).publish(msg)

    def _publish_sensors(self, articulation: str, art: Any) -> None:
        for sensor_name, sensor in getattr(art, "sensors", {}).items():
            latest = getattr(sensor, "latest", None)
            if latest is None:
                continue
            msg = self._require_type("Float64MultiArray")()
            msg.data = [float(value) for value in np.asarray(latest, dtype=np.float64).ravel()]
            topic_name = _topic_part(str(sensor_name))
            self._publisher(
                f"/{articulation}/sensors/{topic_name}",
                self._require_type("Float64MultiArray"),
            ).publish(msg)

    def _build_image_message(
        self,
        *,
        articulation: str,
        camera_name: str,
        view: Any,
        frame: np.ndarray,
        stamp: Any,
    ) -> Any:
        msg = self._require_type("Image")()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{articulation}/{camera_name}_optical_frame"
        mode_name = str(getattr(getattr(view, "mode", "real"), "value", getattr(view, "mode", "real"))).lower()
        if mode_name == "real":
            pixels = linear_rgb_to_srgb_u8(frame)
            msg.height = int(pixels.shape[0])
            msg.width = int(pixels.shape[1])
            msg.encoding = "rgb8"
            msg.is_bigendian = 0
            msg.step = int(msg.width) * 3
            msg.data = np.ascontiguousarray(pixels, dtype=np.uint8).tobytes()
            return msg

        if mode_name == "depth":
            pixels = np.asarray(frame, dtype=np.float32)
            if pixels.ndim == 3 and pixels.shape[2] == 1:
                pixels = pixels[..., 0]
            if pixels.ndim != 2:
                raise ValueError(f"depth camera frame must have shape HxW, got {pixels.shape}")
            meters = np.ascontiguousarray(pixels / 100.0, dtype=np.float32)
            msg.height = int(meters.shape[0])
            msg.width = int(meters.shape[1])
            msg.encoding = "32FC1"
            msg.is_bigendian = 0
            msg.step = int(msg.width) * 4
            msg.data = meters.tobytes()
            return msg

        pixels = np.asarray(frame, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError(f"segmentation camera frame must have shape HxWx4, got {pixels.shape}")
        msg.height = int(pixels.shape[0])
        msg.width = int(pixels.shape[1])
        msg.encoding = "bgra8"
        msg.is_bigendian = 0
        msg.step = int(msg.width) * 4
        msg.data = np.ascontiguousarray(pixels, dtype=np.uint8).tobytes()
        return msg

    def _build_camera_info_message(
        self,
        *,
        articulation: str,
        camera_name: str,
        view: Any,
        stamp: Any,
    ) -> Any:
        msg = self._require_type("CameraInfo")()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{articulation}/{camera_name}_optical_frame"
        width, height = getattr(view, "resolution", (0, 0))
        msg.width = int(width)
        msg.height = int(height)
        return msg

    def _run_camera_publisher(self) -> None:
        interval_s = 1.0 / self.camera_fps
        while not self._camera_stop_event.is_set():
            started = self._monotonic()
            try:
                self.publish_camera_frames_once()
            except Exception as exc:  # pragma: no cover - live ROS camera path
                self._logger.error("ROS2 camera publish failed: %s", exc)
            elapsed = max(0.0, self._monotonic() - started)
            self._camera_stop_event.wait(max(0.0, interval_s - elapsed))

    def _publisher(self, topic: str, msg_type: Any, *, qos: Any = 10) -> Any:
        if self._node is None:
            raise RuntimeError("ROS2 gateway has not been started")
        key = (msg_type, topic)
        publisher = self._publishers.get(key)
        if publisher is None:
            publisher = self._node.create_publisher(msg_type, topic, qos)
            self._publishers[key] = publisher
            self._logger.info("ROS2 publisher ready: %s", topic)
        return publisher

    def _camera_qos(self) -> Any:
        if self._types is None:
            return 1
        return getattr(self._types, "CameraQoS", None) or 1

    def _require_type(self, name: str) -> Any:
        if self._types is None:
            raise RuntimeError("ROS2 gateway has not been started")
        msg_type = getattr(self._types, name, None)
        if msg_type is None:
            raise RuntimeError(f"ROS2 message type {name} is not available")
        return msg_type

    def _stamp(self) -> Any:
        node = self._node
        if node is None:
            return None
        get_clock = getattr(node, "get_clock", None)
        if not callable(get_clock):
            return None
        return get_clock().now().to_msg()

    def _should_publish_state_now(self) -> bool:
        if self.state_hz is None:
            return True
        now = self._monotonic()
        if self._last_state_publish_at is None:
            self._last_state_publish_at = now
            return True
        interval = 1.0 / self.state_hz
        if now - self._last_state_publish_at < interval:
            return False
        self._last_state_publish_at = now
        return True

    @staticmethod
    def _coerce_message_types(message_types: Any) -> Any:
        if hasattr(message_types, "Twist"):
            return message_types
        return SimpleNamespace(Twist=message_types)

    def _make_spin_target(self, rclpy: Any, node: Any) -> Callable[[], None]:
        executor_type = getattr(getattr(rclpy, "executors", None), "SingleThreadedExecutor", None)
        if executor_type is None:
            return lambda: rclpy.spin(node)

        executor = executor_type()
        executor.add_node(node)
        self._executor = executor
        return executor.spin


def _state_articulation_name(state: Any) -> str:
    return str(getattr(state, "articulation", getattr(state, "prefix", "")))


def _joint_state_arrays(art: Any) -> tuple[list[str], list[float], list[float]]:
    qpos = np.asarray(getattr(art, "dof_qpos"), dtype=np.float64)
    qvel = np.asarray(getattr(art, "dof_qvel"), dtype=np.float64)
    names: list[str] = []
    positions: list[float] = []
    velocities: list[float] = []
    qpos_idx = 0
    qvel_idx = 0
    for name, joint in getattr(art, "joints", {}).items():
        qpos_dim = int(getattr(joint, "qpos_dim", 0))
        qvel_dim = int(getattr(joint, "qvel_dim", 0))
        if qpos_dim == 7 and qvel_dim == 6:
            continue
        if qpos_dim == 1 and qvel_dim == 1:
            if qpos_idx < qpos.size and qvel_idx < qvel.size:
                names.append(str(name))
                positions.append(float(qpos[qpos_idx]))
                velocities.append(float(qvel[qvel_idx]))
        qpos_idx += max(0, qpos_dim)
        qvel_idx += max(0, qvel_dim)
    return names, positions, velocities


def _topic_part(value: str) -> str:
    cleaned = str(value).strip().strip("/").replace("/", "_").replace(" ", "_")
    return cleaned or "unnamed"
