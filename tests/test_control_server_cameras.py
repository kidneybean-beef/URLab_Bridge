from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from urlab_bridge.control_server.cameras import (
    CameraHub,
    MjpegCameraStream,
    RobotCameraStreams,
    depth_to_grayscale_u8,
    encode_camera_jpeg,
    linear_rgb_to_srgb_u8,
    segmentation_bgra_to_rgb_u8,
)
from urlab_bridge.control_server.models import (
    WebCameraTarget,
    WebPolicyTarget,
    parse_web_camera_target,
)
from urlab_bridge.control_server.server import URLabControlServer


def test_linear_rgb_to_srgb_u8_applies_display_transfer_function() -> None:
    linear = np.array([[[0, 64, 128, 255]]], dtype=np.uint8)

    srgb = linear_rgb_to_srgb_u8(linear)

    assert srgb.tolist() == [[[0, 137, 188]]]


def test_parse_web_camera_target() -> None:
    target = parse_web_camera_target("dog_a:front_rgb")

    assert target == WebCameraTarget("dog_a", "front_rgb")


@pytest.mark.parametrize("raw", ["dog_a", ":front_rgb", "dog_a:"])
def test_parse_web_camera_target_rejects_invalid_value(raw: str) -> None:
    with pytest.raises(SystemExit, match="ARTICULATION:CAMERA"):
        parse_web_camera_target(raw)


def test_mjpeg_stream_encodes_newest_camera_frame() -> None:
    view = SimpleNamespace(
        latest_frame=np.zeros((2, 3, 4), dtype=np.uint8),
        frame_count=1,
        recv_monotonic=10.0,
        capture_unix_time=20.0,
        resolution=(3, 2),
    )
    encoded_frames: list[np.ndarray] = []

    def encoder(frame: np.ndarray, quality: int) -> bytes:
        assert quality == 77
        encoded_frames.append(frame.copy())
        return b"jpeg-one"

    stream = MjpegCameraStream(
        articulation="dog_a",
        camera_name="front_rgb",
        view=view,
        fps=20.0,
        jpeg_quality=77,
        encoder=encoder,
    )

    assert stream.encode_latest() is True
    assert stream.wait_for_frame(after_sequence=0, timeout_s=0.0) == (1, b"jpeg-one")

    # The same URLab frame is not encoded twice.
    assert stream.encode_latest() is False

    view.latest_frame = np.full((2, 3, 4), 4, dtype=np.uint8)
    view.frame_count = 2
    assert stream.encode_latest() is True
    assert stream.wait_for_frame(after_sequence=1, timeout_s=0.0) == (2, b"jpeg-one")
    assert len(encoded_frames) == 2


def test_mjpeg_stream_does_not_gate_received_frames_on_source_enabled_flag() -> None:
    view = SimpleNamespace(
        enabled=False,
        latest_frame=np.zeros((2, 3, 4), dtype=np.uint8),
        frame_count=1,
        resolution=(3, 2),
    )
    stream = MjpegCameraStream(
        articulation="dog_a",
        camera_name="front_depth",
        view=view,
        encoder=lambda _frame, _quality: b"jpeg",
    )

    assert stream.encode_latest() is True
    assert stream.status()["enabled"] is False


def test_depth_camera_encoder_produces_displayable_jpeg() -> None:
    frame = np.array([[1.0, 2.0], [3.0, np.inf]], dtype=np.float32)

    jpeg = encode_camera_jpeg(frame, 80, mode="depth")

    assert jpeg.startswith(b"\xff\xd8")


def test_depth_grayscale_matches_urlab_fixed_near_far_mapping() -> None:
    depth = np.array([[10.0, 55.0, 100.0, 120.0]], dtype=np.float32)

    grayscale = depth_to_grayscale_u8(depth, near=10.0, far=100.0)

    assert grayscale.tolist() == [[0, 128, 255, 255]]


def test_semantic_and_instance_display_swaps_bgra_to_rgb() -> None:
    bgra = np.array([[[32, 64, 224, 255]]], dtype=np.uint8)

    rgb = segmentation_bgra_to_rgb_u8(bgra)

    assert rgb.tolist() == [[[224, 64, 32]]]


def test_robot_camera_streams_exposes_real_inventory_and_default() -> None:
    front_rgb = SimpleNamespace(
        camera_name="front_rgb",
        status=lambda: {"camera": "front_rgb", "mode": "real", "enabled": True},
    )
    front_depth = SimpleNamespace(
        camera_name="front_depth",
        status=lambda: {"camera": "front_depth", "mode": "depth", "enabled": False},
    )
    cameras = RobotCameraStreams(
        articulation="dog_a",
        streams={"front_rgb": front_rgb, "front_depth": front_depth},
        default_camera="front_rgb",
    )

    inventory = cameras.inventory()

    assert inventory["default_camera"] == "front_rgb"
    assert [camera["camera"] for camera in inventory["cameras"]] == [
        "front_rgb",
        "front_depth",
    ]
    assert cameras.stream_for("front_depth") is front_depth


def test_camera_hub_discovers_all_real_cameras_for_explicit_target() -> None:
    front = SimpleNamespace(
        latest_frame=None,
        frame_count=0,
        resolution=(640, 480),
        mode="real",
    )
    depth = SimpleNamespace(
        latest_frame=None,
        frame_count=0,
        resolution=(320, 240),
        mode="depth",
    )
    client = SimpleNamespace(
        articulations={
            "dog_a_runtime": SimpleNamespace(
                cameras={"front_rgb": front, "front_depth": depth}
            ),
        },
    )
    hub = CameraHub(
        client=client,
        targets=[WebCameraTarget("dog_a", "front_rgb")],
        articulation_resolver=lambda _client, name: f"{name}_runtime",
        fps=15.0,
        jpeg_quality=80,
        encoder=lambda _frame, _quality: b"jpeg",
    )

    cameras = hub.stream_for("dog_a")
    stream = cameras.stream_for("front_rgb")

    assert cameras.default_camera == "front_rgb"
    assert tuple(cameras.streams) == ("front_rgb", "front_depth")
    assert stream.view is front
    assert stream.articulation == "dog_a"
    assert stream.camera_name == "front_rgb"


def test_camera_hub_keeps_same_named_cameras_separate_per_robot() -> None:
    dog_a_front = SimpleNamespace(latest_frame=None, frame_count=0, resolution=(640, 480))
    dog_b_front = SimpleNamespace(latest_frame=None, frame_count=0, resolution=(640, 480))
    client = SimpleNamespace(
        articulations={
            "dog_a": SimpleNamespace(cameras={"front_rgb": dog_a_front}),
            "dog_b": SimpleNamespace(cameras={"front_rgb": dog_b_front}),
        }
    )

    hub = CameraHub(
        client=client,
        targets=[
            WebCameraTarget("dog_a", "front_rgb"),
            WebCameraTarget("dog_b", "front_rgb"),
        ],
        articulation_resolver=lambda _client, name: name,
    )

    assert hub.stream_for("dog_a").stream_for("front_rgb").view is dog_a_front
    assert hub.stream_for("dog_b").stream_for("front_rgb").view is dog_b_front
    assert hub.stream_for("dog_a") is not hub.stream_for("dog_b")


def test_camera_hub_rejects_missing_camera_with_available_names() -> None:
    client = SimpleNamespace(
        articulations={"dog_a": SimpleNamespace(cameras={"other": object()})}
    )

    with pytest.raises(KeyError, match=r"available cameras: \['other'\]"):
        CameraHub(
            client=client,
            targets=[WebCameraTarget("dog_a", "front_rgb")],
            articulation_resolver=lambda _client, name: name,
        )


def test_control_server_shares_session_client_with_camera_hub_and_policy_loop() -> None:
    events: list[str] = []
    client = object()

    class FakeSession:
        def __init__(self, **_kwargs) -> None:
            pass

        def connect(self):
            events.append("session.connect")
            return client

        def close(self) -> None:
            events.append("session.close")

    class FakeCameraHub:
        def __init__(self, **kwargs) -> None:
            assert kwargs["client"] is client
            assert kwargs["targets"] == (WebCameraTarget("dog_a", "front_rgb"),)
            self.streams = {"dog_a": "camera-stream"}

        def start(self) -> None:
            events.append("camera.start")

        def close(self) -> None:
            events.append("camera.close")

    class FakeGateway:
        def __init__(self, **kwargs) -> None:
            assert kwargs["camera_streams"] == {"dog_a": "camera-stream"}
            self.target_sources = [(WebPolicyTarget("dog_a", 8099), object())]

        def start(self) -> None:
            events.append("gateway.start")

        def close(self) -> None:
            events.append("gateway.close")

    class FakeLoop:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, loop_client) -> int:
            assert loop_client is client
            events.append("loop.run")
            return 17

    args = SimpleNamespace(
        address="tcp://127.0.0.1",
        step_port=5559,
        state_port=5555,
        web_bind="0.0.0.0",
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
        camera_targets=[WebCameraTarget("dog_a", "front_rgb")],
        session_factory=FakeSession,
        camera_hub_factory=FakeCameraHub,
        web_gateway_factory=FakeGateway,
        control_loop_factory=FakeLoop,
    )

    assert server.run() == 17
    assert events == [
        "session.connect",
        "camera.start",
        "gateway.start",
        "loop.run",
        "camera.close",
        "gateway.close",
        "session.close",
    ]
