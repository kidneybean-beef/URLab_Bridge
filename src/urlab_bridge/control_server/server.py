from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from typing import Any

from .cameras import CameraHub
from .go2_moe import Go2MoeControlLoop, Go2MoeDependencies
from .commands import CommandHub
from .metrics import ControlLoopMetrics
from .models import WebCameraTarget, WebPolicyTarget
from .ros2_gateway import Ros2Gateway
from .session import SessionManager
from .web_gateway import WebGateway

logger = logging.getLogger(__name__)


class URLabControlServer:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        targets: Sequence[WebPolicyTarget],
        limit_mode: Any,
        web_config: Any,
        dependencies: Go2MoeDependencies,
        camera_targets: Sequence[WebCameraTarget] = (),
        camera_fps: float = 20.0,
        camera_jpeg_quality: int = 80,
        session_factory: Callable[..., SessionManager] = SessionManager,
        camera_hub_factory: Callable[..., CameraHub] = CameraHub,
        web_gateway_factory: Callable[..., WebGateway] = WebGateway,
        ros2_gateway_factory: Callable[..., Ros2Gateway] = Ros2Gateway,
        control_loop_factory: Callable[..., Go2MoeControlLoop] = Go2MoeControlLoop,
        log: logging.Logger = logger,
    ) -> None:
        self.args = args
        self.targets = tuple(targets)
        self.limit_mode = limit_mode
        self.web_config = web_config
        self.dependencies = dependencies
        self.camera_targets = tuple(camera_targets)
        self.camera_fps = float(camera_fps)
        self.camera_jpeg_quality = int(camera_jpeg_quality)
        self._session_factory = session_factory
        self._camera_hub_factory = camera_hub_factory
        self._web_gateway_factory = web_gateway_factory
        self._ros2_gateway_factory = ros2_gateway_factory
        self._control_loop_factory = control_loop_factory
        self._logger = log

    def run(self) -> int:
        session = self._session_factory(
            address=self.args.address,
            step_port=self.args.step_port,
            state_port=self.args.state_port,
        )
        command_hub = CommandHub(
            [target.articulation for target in self.targets],
            config=self.web_config,
            stale_timeout_s=self.args.web_stale_timeout_s,
        )
        metrics = ControlLoopMetrics(freq_hz=self.args.freq)
        gateway: WebGateway | None = None
        ros2_gateway: Ros2Gateway | None = None
        camera_hub: CameraHub | None = None
        try:
            client = session.connect()
            camera_streams: dict[str, object] = {}
            if self.camera_targets:
                camera_hub = self._camera_hub_factory(
                    client=client,
                    targets=self.camera_targets,
                    articulation_resolver=self.dependencies.select_articulation,
                    fps=self.camera_fps,
                    jpeg_quality=self.camera_jpeg_quality,
                    log=self._logger,
                )
                camera_hub.start()
                camera_streams = camera_hub.streams

            gateway = self._web_gateway_factory(
                targets=self.targets,
                bind=self.args.web_bind,
                web_config=self.web_config,
                stale_timeout_s=self.args.web_stale_timeout_s,
                command_hub=command_hub,
                metrics_provider=metrics.snapshot,
                camera_streams=camera_streams,
                log=self._logger,
            )
            gateway.start()
            ros2_cmd_vel = bool(getattr(self.args, "ros2_cmd_vel", False))
            ros2_publish_state = bool(getattr(self.args, "ros2_publish_state", False))
            ros2_publish_sensors = bool(getattr(self.args, "ros2_publish_sensors", False))
            ros2_publish_cameras = bool(getattr(self.args, "ros2_publish_cameras", False))
            ros2_publish_compressed_cameras = bool(
                getattr(self.args, "ros2_publish_compressed_cameras", False)
            )
            ros2_enabled = any(
                (
                    ros2_cmd_vel,
                    ros2_publish_state,
                    ros2_publish_sensors,
                    ros2_publish_cameras,
                    ros2_publish_compressed_cameras,
                )
            )
            if ros2_enabled:
                ros2_gateway = self._ros2_gateway_factory(
                    targets=self.targets,
                    command_hub=command_hub,
                    subscribe_cmd_vel=ros2_cmd_vel,
                    publish_state=ros2_publish_state,
                    publish_sensors=ros2_publish_sensors,
                    publish_cameras=ros2_publish_cameras,
                    publish_compressed_cameras=ros2_publish_compressed_cameras,
                    state_hz=getattr(self.args, "ros2_state_hz", None),
                    camera_fps=(
                        getattr(self.args, "ros2_camera_fps", None)
                        or self.camera_fps
                    ),
                    camera_jpeg_quality=(
                        getattr(self.args, "ros2_camera_jpeg_quality", None)
                        or self.camera_jpeg_quality
                    ),
                    camera_log_interval_s=getattr(
                        self.args,
                        "ros2_camera_log_interval_s",
                        2.0,
                    ),
                    camera_streams=camera_streams,
                    node_name=getattr(
                        self.args,
                        "ros2_node_name",
                        "urlab_go2_control_server",
                    ),
                    log=self._logger,
                )
                ros2_gateway.start()
            post_step_hook = (
                ros2_gateway.publish_control_states
                if ros2_gateway is not None
                and (ros2_publish_state or ros2_publish_sensors)
                else None
            )
            loop = self._control_loop_factory(
                self.args,
                gateway.target_sources,
                self.limit_mode,
                self.dependencies,
                metrics=metrics,
                metrics_log_interval_s=getattr(
                    self.args,
                    "metrics_log_interval_s",
                    1.0,
                ),
                post_step_hook=post_step_hook,
                log=self._logger,
            )
            return loop.run(client)
        finally:
            if ros2_gateway is not None:
                ros2_gateway.close()
            if camera_hub is not None:
                camera_hub.close()
            if gateway is not None:
                gateway.close()
            session.close()
