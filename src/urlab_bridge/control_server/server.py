from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from typing import Any

from .go2_moe import Go2MoeControlLoop, Go2MoeDependencies
from .commands import CommandHub
from .metrics import ControlLoopMetrics
from .models import WebPolicyTarget
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
        session_factory: Callable[..., SessionManager] = SessionManager,
        web_gateway_factory: Callable[..., WebGateway] = WebGateway,
        control_loop_factory: Callable[..., Go2MoeControlLoop] = Go2MoeControlLoop,
        log: logging.Logger = logger,
    ) -> None:
        self.args = args
        self.targets = tuple(targets)
        self.limit_mode = limit_mode
        self.web_config = web_config
        self.dependencies = dependencies
        self._session_factory = session_factory
        self._web_gateway_factory = web_gateway_factory
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
        gateway = self._web_gateway_factory(
            targets=self.targets,
            bind=self.args.web_bind,
            web_config=self.web_config,
            stale_timeout_s=self.args.web_stale_timeout_s,
            command_hub=command_hub,
            metrics_provider=metrics.snapshot,
            log=self._logger,
        )
        try:
            gateway.start()
            client = session.connect()
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
                log=self._logger,
            )
            return loop.run(client)
        finally:
            gateway.close()
            session.close()
