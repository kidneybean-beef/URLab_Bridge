from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from urlab_bridge.control_server.commands import CommandHub
from urlab_bridge.control_server.models import WebPolicyTarget
from urlab_bridge.control_server.server import URLabControlServer


class FakeTwist:
    def __init__(self, *, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        self.linear = SimpleNamespace(x=x, y=y, z=99.0)
        self.angular = SimpleNamespace(x=99.0, y=99.0, z=yaw)


class FakeHeader:
    def __init__(self) -> None:
        self.stamp = None
        self.frame_id = ""


class FakeJointState:
    def __init__(self) -> None:
        self.header = FakeHeader()
        self.name = []
        self.position = []
        self.velocity = []
        self.effort = []


class FakeOdometry:
    def __init__(self) -> None:
        self.header = FakeHeader()
        self.child_frame_id = ""
        self.pose = SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
        self.twist = SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            )
        )


class FakeFloat64MultiArray:
    def __init__(self) -> None:
        self.data = []


class FakeImage:
    def __init__(self) -> None:
        self.header = FakeHeader()
        self.height = 0
        self.width = 0
        self.encoding = ""
        self.is_bigendian = 0
        self.step = 0
        self.data = b""


class FakeCameraInfo:
    def __init__(self) -> None:
        self.header = FakeHeader()
        self.height = 0
        self.width = 0
        self.distortion_model = ""
        self.d = []
        self.k = [0.0] * 9
        self.r = [0.0] * 9
        self.p = [0.0] * 12


class FakeCompressedImage:
    def __init__(self) -> None:
        self.header = FakeHeader()
        self.format = ""
        self.data = b""


class FakeRos2Types(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            Twist=FakeTwist,
            JointState=FakeJointState,
            Odometry=FakeOdometry,
            Float64MultiArray=FakeFloat64MultiArray,
            Image=FakeImage,
            CompressedImage=FakeCompressedImage,
            CameraInfo=FakeCameraInfo,
            CameraQoS="sensor-data-qos",
        )


class FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.subscriptions: list[tuple[object, str, object, int]] = []
        self.publishers: list[FakePublisher] = []
        self.destroyed = False

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions.append((msg_type, topic, callback, qos))
        return SimpleNamespace(topic=topic)

    def create_publisher(self, msg_type, topic, qos):
        publisher = FakePublisher(msg_type=msg_type, topic=topic, qos=qos)
        self.publishers.append(publisher)
        return publisher

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp-1"))

    def destroy_node(self) -> None:
        self.destroyed = True


class FakePublisher:
    def __init__(self, *, msg_type, topic: str, qos: int) -> None:
        self.msg_type = msg_type
        self.topic = topic
        self.qos = qos
        self.messages: list[object] = []

    def publish(self, msg) -> None:
        self.messages.append(msg)


class FakeRclpy:
    def __init__(self, *, already_ok: bool = False) -> None:
        self._ok = already_ok
        self.init_calls = 0
        self.shutdown_calls = 0
        self.created_nodes: list[FakeNode] = []
        self.spun_nodes: list[FakeNode] = []

    def ok(self) -> bool:
        return self._ok

    def init(self, args=None) -> None:
        self.init_calls += 1
        self._ok = True

    def create_node(self, name: str) -> FakeNode:
        node = FakeNode(name)
        self.created_nodes.append(node)
        return node

    def spin(self, node: FakeNode) -> None:
        self.spun_nodes.append(node)

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._ok = False


class ImmediateThread:
    instances: list["ImmediateThread"] = []

    def __init__(self, *, target, daemon) -> None:
        self.target = target
        self.daemon = daemon
        self.started = False
        self.joined = False
        ImmediateThread.instances.append(self)

    def start(self) -> None:
        self.started = True
        self.target()

    def join(self, timeout=None) -> None:
        self.joined = True


class PassiveThread:
    def __init__(self, *, target, daemon) -> None:
        self.target = target
        self.daemon = daemon
        self.started = False
        self.joined = False

    def start(self) -> None:
        self.started = True

    def join(self, timeout=None) -> None:
        self.joined = True


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class RecordingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []

    def info(self, message: str, *args) -> None:
        self.info_messages.append(message % args if args else message)

    def debug(self, _message: str, *args) -> None:
        pass

    def error(self, _message: str, *args) -> None:
        pass


def test_ros2_gateway_subscribes_to_each_target_and_applies_twist() -> None:
    from urlab_bridge.control_server.ros2_gateway import Ros2Gateway

    fake_rclpy = FakeRclpy()
    hub = CommandHub(["dog_a", "dog_b"])
    gateway = Ros2Gateway(
        targets=[WebPolicyTarget("dog_a", 8099), WebPolicyTarget("dog_b", 8100)],
        command_hub=hub,
        node_name="urlab_test_node",
        ros2_loader=lambda: (fake_rclpy, FakeRos2Types()),
        thread_factory=ImmediateThread,
    )

    gateway.start()

    node = fake_rclpy.created_nodes[0]
    assert node.name == "urlab_test_node"
    assert [topic for _msg_type, topic, _callback, _qos in node.subscriptions] == [
        "/dog_a/cmd_vel",
        "/dog_b/cmd_vel",
    ]
    assert all(msg_type is FakeTwist for msg_type, _topic, _callback, _qos in node.subscriptions)
    assert fake_rclpy.init_calls == 1
    assert fake_rclpy.spun_nodes == [node]

    _msg_type, _topic, dog_a_callback, _qos = node.subscriptions[0]
    dog_a_callback(FakeTwist(x=0.2, y=-0.1, yaw=0.35))

    assert hub.port("dog_a").poll() == pytest.approx((0.2, -0.1, 0.35))
    assert hub.port("dog_a").status()["source"] == "ros2"
    assert hub.port("dog_b").poll() == pytest.approx((0.0, 0.0, 0.0))

    gateway.close()

    assert node.destroyed is True
    assert fake_rclpy.shutdown_calls == 1
    assert ImmediateThread.instances[-1].joined is True


def test_ros2_gateway_does_not_shutdown_ros2_it_did_not_initialize() -> None:
    from urlab_bridge.control_server.ros2_gateway import Ros2Gateway

    fake_rclpy = FakeRclpy(already_ok=True)
    gateway = Ros2Gateway(
        targets=[WebPolicyTarget("dog_a", 8099)],
        command_hub=CommandHub(["dog_a"]),
        ros2_loader=lambda: (fake_rclpy, FakeRos2Types()),
        thread_factory=ImmediateThread,
    )

    gateway.start()
    gateway.close()

    assert fake_rclpy.init_calls == 0
    assert fake_rclpy.shutdown_calls == 0


def test_ros2_gateway_reports_missing_ros2_dependency() -> None:
    from urlab_bridge.control_server.ros2_gateway import Ros2Gateway

    def missing_ros2():
        raise ImportError("No module named 'rclpy'")

    gateway = Ros2Gateway(
        targets=[WebPolicyTarget("dog_a", 8099)],
        command_hub=CommandHub(["dog_a"]),
        ros2_loader=missing_ros2,
        thread_factory=ImmediateThread,
    )

    with pytest.raises(SystemExit, match="source /opt/ros"):
        gateway.start()


def _publisher_by_topic(node: FakeNode) -> dict[str, FakePublisher]:
    return {publisher.topic: publisher for publisher in node.publishers}


def test_ros2_gateway_publishes_joint_state_odom_and_sensors() -> None:
    from urlab_bridge.control_server.ros2_gateway import Ros2Gateway

    fake_rclpy = FakeRclpy()
    gateway = Ros2Gateway(
        targets=[WebPolicyTarget("dog_a", 8099)],
        command_hub=CommandHub(["dog_a"]),
        publish_state=True,
        publish_sensors=True,
        ros2_loader=lambda: (fake_rclpy, FakeRos2Types()),
        thread_factory=ImmediateThread,
    )
    gateway.start()

    art = SimpleNamespace(
        joints=OrderedDict(
            [
                ("root", SimpleNamespace(qpos_dim=7, qvel_dim=6)),
                ("FR_hip", SimpleNamespace(qpos_dim=1, qvel_dim=1)),
                ("FR_thigh", SimpleNamespace(qpos_dim=1, qvel_dim=1)),
            ]
        ),
        dof_qpos=np.array([0.11, 0.22], dtype=np.float64),
        dof_qvel=np.array([1.1, 2.2], dtype=np.float64),
        root_pos_w=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        root_quat_xyzw=np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float64),
        root_lin_vel_w=np.array([0.4, 0.5, 0.6], dtype=np.float64),
        root_ang_vel_w=np.array([0.7, 0.8, 0.9], dtype=np.float64),
        sensors={
            "imu": SimpleNamespace(latest=np.array([3.0, 4.0], dtype=np.float64)),
            "empty": SimpleNamespace(latest=None),
        },
    )

    gateway.publish_control_states([SimpleNamespace(prefix="dog_a", art=art)])

    publishers = _publisher_by_topic(fake_rclpy.created_nodes[0])
    joint_msg = publishers["/dog_a/joint_states"].messages[-1]
    odom_msg = publishers["/dog_a/odom"].messages[-1]
    sensor_msg = publishers["/dog_a/sensors/imu"].messages[-1]

    assert publishers["/dog_a/joint_states"].qos == 10
    assert publishers["/dog_a/odom"].qos == 10
    assert publishers["/dog_a/sensors/imu"].qos == 10

    assert joint_msg.header.stamp == "stamp-1"
    assert joint_msg.header.frame_id == "dog_a/base"
    assert joint_msg.name == ["FR_hip", "FR_thigh"]
    assert joint_msg.position == pytest.approx([0.11, 0.22])
    assert joint_msg.velocity == pytest.approx([1.1, 2.2])
    assert joint_msg.effort == []

    assert odom_msg.header.frame_id == "urlab_world"
    assert odom_msg.child_frame_id == "dog_a/base"
    assert odom_msg.pose.pose.position.x == pytest.approx(1.0)
    assert odom_msg.pose.pose.position.y == pytest.approx(2.0)
    assert odom_msg.pose.pose.position.z == pytest.approx(3.0)
    assert odom_msg.pose.pose.orientation.x == pytest.approx(0.1)
    assert odom_msg.pose.pose.orientation.y == pytest.approx(0.2)
    assert odom_msg.pose.pose.orientation.z == pytest.approx(0.3)
    assert odom_msg.pose.pose.orientation.w == pytest.approx(0.9)
    assert odom_msg.twist.twist.linear.x == pytest.approx(0.4)
    assert odom_msg.twist.twist.angular.z == pytest.approx(0.9)

    assert sensor_msg.data == pytest.approx([3.0, 4.0])
    assert "/dog_a/sensors/empty" not in publishers


def test_ros2_gateway_camera_publishers_map_modes_and_skip_duplicate_frames() -> None:
    from urlab_bridge.control_server.ros2_gateway import Ros2Gateway

    fake_rclpy = FakeRclpy()
    real_view = SimpleNamespace(
        latest_frame=np.array([[[128, 64, 0, 255]]], dtype=np.uint8),
        frame_count=1,
        resolution=(1, 1),
        fovy=90.0,
        mode="real",
        payload_encoding="bgra8_srgb",
    )
    depth_view = SimpleNamespace(
        latest_frame=np.array([[150.0, 250.0]], dtype=np.float32),
        frame_count=1,
        resolution=(2, 1),
        fovy=90.0,
        mode="depth",
    )
    semantic_view = SimpleNamespace(
        latest_frame=np.array([[[32, 64, 224, 255]]], dtype=np.uint8),
        frame_count=1,
        resolution=(1, 1),
        fovy=90.0,
        mode="semantic",
    )
    streams = SimpleNamespace(
        streams={
            "front_rgb": SimpleNamespace(view=real_view),
            "front_depth": SimpleNamespace(view=depth_view),
            "front_semantic": SimpleNamespace(view=semantic_view),
        }
    )
    gateway = Ros2Gateway(
        targets=[WebPolicyTarget("dog_a", 8099)],
        command_hub=CommandHub(["dog_a"]),
        publish_cameras=True,
        camera_streams={"dog_a": streams},
        ros2_loader=lambda: (fake_rclpy, FakeRos2Types()),
        thread_factory=ImmediateThread,
        camera_thread_factory=PassiveThread,
    )
    gateway.start()

    assert gateway.publish_camera_frames_once() == 3
    assert gateway.publish_camera_frames_once() == 0

    publishers = _publisher_by_topic(fake_rclpy.created_nodes[0])
    rgb_msg = publishers["/dog_a/camera/front_rgb/image_raw"].messages[-1]
    depth_msg = publishers["/dog_a/camera/front_depth/image_raw"].messages[-1]
    semantic_msg = publishers["/dog_a/camera/front_semantic/image_raw"].messages[-1]
    rgb_info = publishers["/dog_a/camera/front_rgb/camera_info"].messages[-1]

    assert publishers["/dog_a/camera/front_rgb/image_raw"].qos == "sensor-data-qos"
    assert publishers["/dog_a/camera/front_rgb/camera_info"].qos == "sensor-data-qos"
    assert publishers["/dog_a/camera/front_depth/image_raw"].qos == "sensor-data-qos"
    assert publishers["/dog_a/camera/front_semantic/image_raw"].qos == "sensor-data-qos"

    assert rgb_msg.encoding == "rgb8"
    assert rgb_msg.height == 1
    assert rgb_msg.width == 1
    assert rgb_msg.step == 3
    assert list(rgb_msg.data) == [128, 64, 0]
    assert rgb_msg.header.frame_id == "dog_a/front_rgb_optical_frame"

    assert depth_msg.encoding == "32FC1"
    assert depth_msg.height == 1
    assert depth_msg.width == 2
    assert depth_msg.step == 8
    assert np.frombuffer(depth_msg.data, dtype=np.float32).tolist() == pytest.approx([1.5, 2.5])

    assert semantic_msg.encoding == "bgra8"
    assert semantic_msg.step == 4
    assert list(semantic_msg.data) == [32, 64, 224, 255]

    assert rgb_info.header.frame_id == "dog_a/front_rgb_optical_frame"
    assert rgb_info.height == 1
    assert rgb_info.width == 1
    assert rgb_info.distortion_model == "plumb_bob"
    assert rgb_info.d == pytest.approx([0.0] * 5)
    assert rgb_info.k == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    assert rgb_info.r == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    assert rgb_info.p == pytest.approx(
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )


def test_centered_pinhole_calibration_matches_urlab_vertical_fov_contract() -> None:
    from urlab_bridge.control_server.ros2_gateway import centered_pinhole_calibration

    calibration = centered_pinhole_calibration(640, 480, 90.0)

    assert calibration.d == pytest.approx([0.0] * 5)
    assert calibration.k == pytest.approx(
        [240.0, 0.0, 319.5, 0.0, 240.0, 239.5, 0.0, 0.0, 1.0]
    )
    assert calibration.r == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )
    assert calibration.p == pytest.approx(
        [240.0, 0.0, 319.5, 0.0, 0.0, 240.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
    )


@pytest.mark.parametrize(
    ("resolution", "fovy"),
    [((0, 480), 90.0), ((640, 0), 90.0), ((640, 480), 0.0), ((640, 480), 180.0)],
)
def test_ros2_gateway_rejects_invalid_camera_calibration_at_startup(
    resolution: tuple[int, int],
    fovy: float,
) -> None:
    from urlab_bridge.control_server.ros2_gateway import Ros2Gateway

    view = SimpleNamespace(
        latest_frame=None,
        frame_count=0,
        resolution=resolution,
        fovy=fovy,
        mode="real",
    )
    streams = SimpleNamespace(streams={"front_rgb": SimpleNamespace(view=view)})
    gateway = Ros2Gateway(
        targets=[WebPolicyTarget("dog_a", 8099)],
        command_hub=CommandHub(["dog_a"]),
        publish_cameras=True,
        camera_streams={"dog_a": streams},
        ros2_loader=lambda: (FakeRclpy(), FakeRos2Types()),
        thread_factory=ImmediateThread,
        camera_thread_factory=PassiveThread,
    )

    with pytest.raises(ValueError, match="invalid camera calibration for dog_a:front_rgb"):
        gateway.start()


def test_ros2_gateway_publishes_compressed_camera_preview() -> None:
    from urlab_bridge.control_server.ros2_gateway import Ros2Gateway

    fake_rclpy = FakeRclpy()
    real_view = SimpleNamespace(
        latest_frame=np.array([[[0, 64, 128, 255]]], dtype=np.uint8),
        frame_count=1,
        resolution=(1, 1),
        fovy=90.0,
        mode="real",
    )
    streams = SimpleNamespace(streams={"front_rgb": SimpleNamespace(view=real_view)})

    def fake_encoder(frame, quality, **kwargs):
        assert frame.shape == (1, 1, 4)
        assert quality == 72
        assert kwargs["mode"] == "real"
        return b"jpeg-bytes"

    gateway = Ros2Gateway(
        targets=[WebPolicyTarget("dog_a", 8099)],
        command_hub=CommandHub(["dog_a"]),
        publish_compressed_cameras=True,
        camera_jpeg_quality=72,
        camera_streams={"dog_a": streams},
        ros2_loader=lambda: (fake_rclpy, FakeRos2Types()),
        thread_factory=ImmediateThread,
        camera_thread_factory=PassiveThread,
        camera_encoder=fake_encoder,
    )
    gateway.start()

    assert gateway.publish_camera_frames_once() == 1

    publishers = _publisher_by_topic(fake_rclpy.created_nodes[0])
    compressed = publishers["/dog_a/camera/front_rgb/image_raw/compressed"].messages[-1]
    camera_info = publishers["/dog_a/camera/front_rgb/camera_info"].messages[-1]

    assert compressed.header.stamp == "stamp-1"
    assert compressed.header.frame_id == "dog_a/front_rgb_optical_frame"
    assert compressed.format == "jpeg"
    assert compressed.data == b"jpeg-bytes"
    assert publishers["/dog_a/camera/front_rgb/image_raw/compressed"].qos == "sensor-data-qos"
    assert camera_info.width == 1
    assert camera_info.height == 1
    assert "/dog_a/camera/front_rgb/image_raw" not in publishers


def test_ros2_gateway_logs_raw_camera_publish_metrics() -> None:
    from urlab_bridge.control_server.ros2_gateway import Ros2Gateway

    fake_rclpy = FakeRclpy()
    clock = ManualClock()
    log = RecordingLogger()
    depth_view = SimpleNamespace(
        latest_frame=np.array([[100.0, 200.0]], dtype=np.float32),
        frame_count=1,
        resolution=(2, 1),
        fovy=90.0,
        mode="depth",
    )
    streams = SimpleNamespace(streams={"front_depth": SimpleNamespace(view=depth_view)})
    gateway = Ros2Gateway(
        targets=[WebPolicyTarget("dog_a", 8099)],
        command_hub=CommandHub(["dog_a"]),
        publish_cameras=True,
        camera_streams={"dog_a": streams},
        camera_log_interval_s=1.0,
        ros2_loader=lambda: (fake_rclpy, FakeRos2Types()),
        thread_factory=ImmediateThread,
        camera_thread_factory=PassiveThread,
        monotonic=clock,
        log=log,
    )
    gateway.start()

    assert gateway.publish_camera_frames_once() == 1
    clock.advance(1.0)
    depth_view.latest_frame = np.array([[300.0, 400.0]], dtype=np.float32)
    depth_view.frame_count = 2
    assert gateway.publish_camera_frames_once() == 1

    metrics = [message for message in log.info_messages if message.startswith("ROS2 camera metrics:")]
    assert len(metrics) == 1
    assert "published_fps=2.000" in metrics[0]
    assert "published=2" in metrics[0]
    assert "duplicate=0" in metrics[0]
    assert "dog_a/front_depth:2" in metrics[0]


def test_control_server_starts_ros2_gateway_only_when_enabled() -> None:
    events: list[str] = []
    client = object()
    captured: dict[str, object] = {}

    class FakeSession:
        def __init__(self, **_kwargs) -> None:
            pass

        def connect(self):
            events.append("session.connect")
            return client

        def close(self) -> None:
            events.append("session.close")

    class FakeGateway:
        def __init__(self, **kwargs) -> None:
            captured["web_command_hub"] = kwargs["command_hub"]
            self.target_sources = [(WebPolicyTarget("dog_a", 8099), object())]

        def start(self) -> None:
            events.append("web.start")

        def close(self) -> None:
            events.append("web.close")

    class FakeRos2Gateway:
        def __init__(self, **kwargs) -> None:
            captured["ros_command_hub"] = kwargs["command_hub"]
            captured["ros_targets"] = kwargs["targets"]
            captured["ros_node_name"] = kwargs["node_name"]
            captured["subscribe_cmd_vel"] = kwargs["subscribe_cmd_vel"]

        def start(self) -> None:
            events.append("ros2.start")

        def close(self) -> None:
            events.append("ros2.close")

    class FakeLoop:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, loop_client) -> int:
            assert loop_client is client
            events.append("loop.run")
            return 23

    args = SimpleNamespace(
        address="tcp://127.0.0.1",
        step_port=5559,
        state_port=5555,
        web_bind="127.0.0.1",
        web_stale_timeout_s=0.5,
        freq=50.0,
        metrics_log_interval_s=0.0,
        ros2_cmd_vel=True,
        ros2_publish_state=False,
        ros2_publish_sensors=False,
        ros2_publish_cameras=False,
        ros2_node_name="urlab_test_ros2",
    )
    server = URLabControlServer(
        args=args,
        targets=[WebPolicyTarget("dog_a", 8099)],
        limit_mode=object(),
        web_config=object(),
        dependencies=SimpleNamespace(select_articulation=lambda _client, name: name),
        session_factory=FakeSession,
        web_gateway_factory=FakeGateway,
        ros2_gateway_factory=FakeRos2Gateway,
        control_loop_factory=FakeLoop,
    )

    assert server.run() == 23
    assert captured["web_command_hub"] is captured["ros_command_hub"]
    assert captured["ros_targets"] == (WebPolicyTarget("dog_a", 8099),)
    assert captured["ros_node_name"] == "urlab_test_ros2"
    assert captured["subscribe_cmd_vel"] is True
    assert events == [
        "session.connect",
        "web.start",
        "ros2.start",
        "loop.run",
        "ros2.close",
        "web.close",
        "session.close",
    ]


def test_control_server_skips_ros2_gateway_by_default() -> None:
    events: list[str] = []

    class FakeSession:
        def __init__(self, **_kwargs) -> None:
            pass

        def connect(self):
            return object()

        def close(self) -> None:
            events.append("session.close")

    class FakeGateway:
        def __init__(self, **_kwargs) -> None:
            self.target_sources = [(WebPolicyTarget("dog_a", 8099), object())]

        def start(self) -> None:
            events.append("web.start")

        def close(self) -> None:
            events.append("web.close")

    class ExplodingRos2Gateway:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("ROS2 gateway must not be constructed by default")

    class FakeLoop:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, _client) -> int:
            events.append("loop.run")
            return 19

    args = SimpleNamespace(
        address="tcp://127.0.0.1",
        step_port=5559,
        state_port=5555,
        web_bind="127.0.0.1",
        web_stale_timeout_s=0.5,
        freq=50.0,
        metrics_log_interval_s=0.0,
    )
    server = URLabControlServer(
        args=args,
        targets=[WebPolicyTarget("dog_a", 8099)],
        limit_mode=object(),
        web_config=object(),
        dependencies=SimpleNamespace(select_articulation=lambda _client, name: name),
        session_factory=FakeSession,
        web_gateway_factory=FakeGateway,
        ros2_gateway_factory=ExplodingRos2Gateway,
        control_loop_factory=FakeLoop,
    )

    assert server.run() == 19
    assert events == ["web.start", "loop.run", "web.close", "session.close"]


def test_control_server_starts_ros2_for_publish_only_mode_and_wires_post_step_hook() -> None:
    events: list[str] = []
    captured: dict[str, object] = {}

    class FakeSession:
        def __init__(self, **_kwargs) -> None:
            pass

        def connect(self):
            return object()

        def close(self) -> None:
            events.append("session.close")

    class FakeGateway:
        def __init__(self, **_kwargs) -> None:
            self.target_sources = [(WebPolicyTarget("dog_a", 8099), object())]

        def start(self) -> None:
            events.append("web.start")

        def close(self) -> None:
            events.append("web.close")

    class FakeRos2Gateway:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.published_states: list[object] = []

        def start(self) -> None:
            events.append("ros2.start")

        def close(self) -> None:
            events.append("ros2.close")

        def publish_control_states(self, states) -> None:
            self.published_states.append(tuple(states))
            events.append("ros2.publish")

    class FakeLoop:
        def __init__(self, *_args, **kwargs) -> None:
            captured["post_step_hook"] = kwargs.get("post_step_hook")

        def run(self, _client) -> int:
            captured["post_step_hook"](["state-one"])
            events.append("loop.run")
            return 29

    args = SimpleNamespace(
        address="tcp://127.0.0.1",
        step_port=5559,
        state_port=5555,
        web_bind="127.0.0.1",
        web_stale_timeout_s=0.5,
        freq=50.0,
        metrics_log_interval_s=0.0,
        ros2_cmd_vel=False,
        ros2_publish_state=True,
        ros2_publish_sensors=False,
        ros2_publish_cameras=False,
        ros2_state_hz=None,
        ros2_camera_fps=None,
        ros2_node_name="urlab_publish_only",
    )
    server = URLabControlServer(
        args=args,
        targets=[WebPolicyTarget("dog_a", 8099)],
        limit_mode=object(),
        web_config=object(),
        dependencies=SimpleNamespace(select_articulation=lambda _client, name: name),
        session_factory=FakeSession,
        web_gateway_factory=FakeGateway,
        ros2_gateway_factory=FakeRos2Gateway,
        control_loop_factory=FakeLoop,
    )

    assert server.run() == 29
    assert captured["node_name"] == "urlab_publish_only"
    assert captured["subscribe_cmd_vel"] is False
    assert captured["publish_state"] is True
    assert captured["publish_sensors"] is False
    assert captured["publish_cameras"] is False
    assert events == [
        "web.start",
        "ros2.start",
        "ros2.publish",
        "loop.run",
        "ros2.close",
        "web.close",
        "session.close",
    ]
