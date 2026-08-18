from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from typing import Any

import numpy as np

from .models import WebCameraTarget

logger = logging.getLogger(__name__)

JpegEncoder = Callable[[np.ndarray, int], bytes]

REAL_PAYLOAD_SRGB = "bgra8_srgb"
REAL_PAYLOAD_LINEAR = "bgra8_linear"

_LINEAR_U8 = np.arange(256, dtype=np.float32) / 255.0
_LINEAR_TO_SRGB_U8 = np.clip(
    np.rint(
        np.where(
            _LINEAR_U8 <= 0.0031308,
            12.92 * _LINEAR_U8,
            1.055 * np.power(_LINEAR_U8, 1.0 / 2.4) - 0.055,
        )
        * 255.0
    ),
    0.0,
    255.0,
).astype(np.uint8)


def linear_rgb_to_srgb_u8(frame: np.ndarray) -> np.ndarray:
    """Convert legacy URLab linear RGBA bytes to display-encoded RGB."""
    return _LINEAR_TO_SRGB_U8[frame[..., :3]]


def real_rgba_to_display_rgb_u8(
    frame: np.ndarray,
    *,
    payload_encoding: str,
) -> np.ndarray:
    """Return display-ready RGB from a decoded URLab Real-camera frame.

    ``URLabCameraView.latest_frame`` is already RGBA because the client has
    swizzled URLab's native BGRA wire bytes. New plugins advertise
    ``bgra8_srgb`` and require only that swizzle. Older plugins advertise
    ``bgra8_linear`` and retain the legacy display-transfer LUT.
    """
    pixels = np.asarray(frame)
    if pixels.dtype != np.uint8:
        raise ValueError(f"Real camera frame must be uint8, got {pixels.dtype}")
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError(
            "Real camera frame must have shape HxWx3 or HxWx4, "
            f"got {pixels.shape}"
        )

    if payload_encoding == REAL_PAYLOAD_SRGB:
        return pixels[..., :3]
    if payload_encoding == REAL_PAYLOAD_LINEAR:
        return linear_rgb_to_srgb_u8(pixels)
    raise ValueError(
        "unsupported Real camera payload encoding "
        f"{payload_encoding!r}; expected {REAL_PAYLOAD_SRGB!r} or "
        f"{REAL_PAYLOAD_LINEAR!r}"
    )


def encode_rgb_jpeg(
    frame: np.ndarray,
    quality: int,
    *,
    payload_encoding: str = REAL_PAYLOAD_LINEAR,
) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "web camera streaming requires Pillow; install the web-camera extra"
        ) from exc

    image = Image.fromarray(
        real_rgba_to_display_rgb_u8(
            frame,
            payload_encoding=payload_encoding,
        ),
        mode="RGB",
    )
    output = BytesIO()
    image.save(output, format="JPEG", quality=int(quality), subsampling=2)
    return output.getvalue()


def depth_to_grayscale_u8(
    frame: np.ndarray,
    *,
    near: float,
    far: float,
) -> np.ndarray:
    """Match URLab's fixed-range depth preview conversion."""
    pixels = np.asarray(frame, dtype=np.float32)
    near_value = max(0.1, float(near))
    far_value = max(near_value + 1.0, float(far))
    finite = np.nan_to_num(
        pixels,
        nan=far_value,
        posinf=far_value,
        neginf=near_value,
    )
    normalized = np.clip(
        (finite - near_value) / (far_value - near_value),
        0.0,
        1.0,
    )
    return np.rint(normalized * 255.0).astype(np.uint8)


def segmentation_bgra_to_rgb_u8(frame: np.ndarray) -> np.ndarray:
    """Convert URLab's ID-preserving BGRA segmentation frame for display."""
    pixels = np.asarray(frame)
    if pixels.dtype != np.uint8:
        raise ValueError(f"segmentation camera frame must be uint8, got {pixels.dtype}")
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError(
            f"segmentation camera frame must have shape HxWx4, got {pixels.shape}"
        )
    return pixels[..., [2, 1, 0]]


def encode_camera_jpeg(
    frame: np.ndarray,
    quality: int,
    *,
    mode: object,
    payload_encoding: str = REAL_PAYLOAD_LINEAR,
    depth_near_cm: float = 10.0,
    depth_far_cm: float = 10000.0,
) -> bytes:
    """Encode a URLab camera mode into a browser-displayable JPEG."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "web camera streaming requires Pillow; install the web-camera extra"
        ) from exc

    mode_name = str(getattr(mode, "value", mode)).lower()
    pixels = np.asarray(frame)
    if mode_name == "real":
        return encode_rgb_jpeg(
            pixels,
            quality,
            payload_encoding=payload_encoding,
        )

    if mode_name == "depth":
        if pixels.ndim == 3 and pixels.shape[2] == 1:
            pixels = pixels[..., 0]
        if pixels.ndim != 2:
            raise ValueError(f"depth camera frame must have shape HxW, got {pixels.shape}")
        display = depth_to_grayscale_u8(
            pixels,
            near=depth_near_cm,
            far=depth_far_cm,
        )
        image = Image.fromarray(display, mode="L")
    else:
        image = Image.fromarray(segmentation_bgra_to_rgb_u8(pixels), mode="RGB")

    output = BytesIO()
    image.save(output, format="JPEG", quality=int(quality), subsampling=2)
    return output.getvalue()


class MjpegCameraStream:
    """Encode the newest URLab RGB frame without queueing stale frames."""

    def __init__(
        self,
        *,
        articulation: str,
        camera_name: str,
        view: Any,
        fps: float = 20.0,
        jpeg_quality: int = 80,
        encoder: JpegEncoder | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        thread_factory: Callable[..., Any] = threading.Thread,
    ) -> None:
        if float(fps) <= 0.0:
            raise ValueError("camera fps must be greater than zero")
        if int(jpeg_quality) < 1 or int(jpeg_quality) > 100:
            raise ValueError("camera JPEG quality must be between 1 and 100")

        self.articulation = str(articulation)
        self.camera_name = str(camera_name)
        self.view = view
        self.fps = float(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.mode = getattr(view, "mode", "real")
        self.payload_encoding = str(
            getattr(view, "payload_encoding", REAL_PAYLOAD_LINEAR)
        )
        self._encoder = encoder or (
            lambda frame, quality: encode_camera_jpeg(
                frame,
                quality,
                mode=self.mode,
                payload_encoding=self.payload_encoding,
                depth_near_cm=float(getattr(view, "depth_near_cm", 10.0)),
                depth_far_cm=float(getattr(view, "depth_far_cm", 10000.0)),
            )
        )
        self._monotonic = monotonic
        self._thread_factory = thread_factory
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._thread: Any | None = None
        self._source_key: tuple[int, int] | None = None
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._encoded_at: float | None = None
        self._last_error: str | None = None
        self._viewer_count = 0

    @property
    def closed(self) -> bool:
        return self._stop_event.is_set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = self._thread_factory(
            target=self._run,
            name=f"urlab-camera-{self.articulation}-{self.camera_name}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def acquire_viewer(self) -> None:
        with self._condition:
            self._viewer_count += 1
            self._condition.notify_all()

    def release_viewer(self) -> None:
        with self._condition:
            self._viewer_count = max(0, self._viewer_count - 1)
            self._condition.notify_all()

    def encode_latest(self) -> bool:
        frame = getattr(self.view, "latest_frame", None)
        if frame is None:
            return False
        source_key = (int(getattr(self.view, "frame_count", 0)), id(frame))
        if source_key == self._source_key:
            return False

        jpeg = self._encoder(np.asarray(frame), self.jpeg_quality)
        if not jpeg:
            raise ValueError("JPEG encoder returned an empty frame")
        with self._condition:
            self._source_key = source_key
            self._jpeg = bytes(jpeg)
            self._sequence += 1
            self._encoded_at = self._monotonic()
            self._last_error = None
            self._condition.notify_all()
        return True

    def wait_for_frame(
        self,
        *,
        after_sequence: int,
        timeout_s: float = 1.0,
    ) -> tuple[int, bytes] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > int(after_sequence) or self.closed,
                timeout=max(0.0, float(timeout_s)),
            )
            if self._sequence <= int(after_sequence) or self._jpeg is None:
                return None
            return self._sequence, self._jpeg

    def status(self) -> dict[str, Any]:
        with self._condition:
            age = None
            if self._encoded_at is not None:
                age = max(0.0, self._monotonic() - self._encoded_at)
            resolution = tuple(getattr(self.view, "resolution", (0, 0)))
            return {
                "ok": self._last_error is None,
                "articulation": self.articulation,
                "camera": self.camera_name,
                "mode": str(getattr(self.mode, "value", self.mode)),
                "payload_encoding": self.payload_encoding,
                "available": self._jpeg is not None,
                "enabled": bool(getattr(self.view, "enabled", True)),
                "closed": self.closed,
                "frame_age_s": age,
                "encoded_frames": self._sequence,
                "source_frame_count": int(getattr(self.view, "frame_count", 0)),
                "resolution": list(resolution),
                "fps": self.fps,
                "jpeg_quality": self.jpeg_quality,
                "error": self._last_error,
                "viewers": self._viewer_count,
            }

    def _run(self) -> None:
        interval_s = 1.0 / self.fps
        reported_error: str | None = None
        while not self._stop_event.is_set():
            with self._condition:
                has_viewer = self._viewer_count > 0
            if not has_viewer:
                self._stop_event.wait(min(0.1, interval_s))
                continue
            started = self._monotonic()
            try:
                self.encode_latest()
                reported_error = None
            except Exception as exc:  # pragma: no cover - live frame failure path
                message = str(exc)
                with self._condition:
                    self._last_error = message
                if message != reported_error:
                    logger.error(
                        "camera encode failed for %s/%s: %s",
                        self.articulation,
                        self.camera_name,
                        message,
                    )
                    reported_error = message
            elapsed = self._monotonic() - started
            self._stop_event.wait(max(0.0, interval_s - elapsed))


class RobotCameraStreams:
    """The cameras URLab advertised for one articulation."""

    def __init__(
        self,
        *,
        articulation: str,
        streams: Mapping[str, MjpegCameraStream],
        default_camera: str,
    ) -> None:
        self.articulation = str(articulation)
        self._streams = dict(streams)
        if default_camera not in self._streams:
            raise KeyError(f"default camera {default_camera!r} is not available")
        self.default_camera = str(default_camera)

    @property
    def streams(self) -> dict[str, MjpegCameraStream]:
        return dict(self._streams)

    @property
    def closed(self) -> bool:
        return all(stream.closed for stream in self._streams.values())

    def stream_for(self, camera: str | None = None) -> MjpegCameraStream | None:
        return self._streams.get(str(camera or self.default_camera))

    def inventory(self) -> dict[str, Any]:
        return {
            "ok": True,
            "articulation": self.articulation,
            "default_camera": self.default_camera,
            "cameras": [stream.status() for stream in self._streams.values()],
        }

    def start(self) -> None:
        for stream in self._streams.values():
            stream.start()

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()


class CameraHub:
    def __init__(
        self,
        *,
        client: Any,
        targets: Sequence[WebCameraTarget],
        articulation_resolver: Callable[[Any, str], str],
        fps: float = 20.0,
        jpeg_quality: int = 80,
        encoder: JpegEncoder | None = None,
        stream_factory: Callable[..., MjpegCameraStream] = MjpegCameraStream,
        log: logging.Logger = logger,
    ) -> None:
        self._logger = log
        self._streams: dict[str, RobotCameraStreams] = {}
        for target in targets:
            if target.articulation in self._streams:
                raise ValueError(
                    f"duplicate camera target for articulation {target.articulation!r}"
                )
            prefix = articulation_resolver(client, target.articulation)
            articulation = client.articulations[prefix]
            cameras: Mapping[str, Any] = getattr(articulation, "cameras", {})
            if target.camera not in cameras:
                raise KeyError(
                    f"camera {target.camera!r} not found on {target.articulation!r}; "
                    f"available cameras: {sorted(cameras)}"
                )
            streams: dict[str, MjpegCameraStream] = {}
            for camera_name, view in cameras.items():
                streams[camera_name] = stream_factory(
                    articulation=target.articulation,
                    camera_name=camera_name,
                    view=view,
                    fps=fps,
                    jpeg_quality=jpeg_quality,
                    encoder=encoder,
                )
            self._streams[target.articulation] = RobotCameraStreams(
                articulation=target.articulation,
                streams=streams,
                default_camera=target.camera,
            )

    @property
    def streams(self) -> dict[str, RobotCameraStreams]:
        return dict(self._streams)

    def stream_for(self, articulation: str) -> RobotCameraStreams | None:
        return self._streams.get(str(articulation))

    def start(self) -> None:
        for cameras in self._streams.values():
            cameras.start()
            self._logger.info(
                "camera streams ready: articulation=%s default=%s available=%s",
                cameras.articulation,
                cameras.default_camera,
                list(cameras.streams),
            )

    def close(self) -> None:
        for cameras in self._streams.values():
            cameras.close()
