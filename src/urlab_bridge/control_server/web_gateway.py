from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from http.server import ThreadingHTTPServer
from typing import Any

from urlab_bridge.web_control import make_handler

from .commands import CommandHub
from .models import WebPolicyTarget

logger = logging.getLogger(__name__)


class _WebServerHandle:
    def __init__(
        self,
        *,
        bind: str,
        target: WebPolicyTarget,
        command_source: Any,
        stale_timeout_s: float,
        server_factory: Callable[..., Any],
        handler_factory: Callable[..., Any],
        thread_factory: Callable[..., Any],
        event_factory: Callable[[], Any],
        metrics_provider: Callable[[], Any] | object | None,
    ) -> None:
        self.target = target
        self.command_source = command_source
        self.server = server_factory(
            (bind, int(target.port)),
            handler_factory(command_source, metrics_provider=metrics_provider),
        )
        self._stop_event = event_factory()
        interval_s = max(
            0.05,
            min(0.25, stale_timeout_s / 2.0 if stale_timeout_s else 0.05),
        )

        def watchdog() -> None:
            while not self._stop_event.wait(interval_s):
                command_source.stop_if_stale()

        self._web_thread = thread_factory(
            target=self.server.serve_forever,
            daemon=True,
        )
        self._watchdog_thread = thread_factory(target=watchdog, daemon=True)

    def start(self) -> None:
        self._web_thread.start()
        self._watchdog_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        self.server.shutdown()
        self.server.server_close()
        self._web_thread.join(timeout=2.0)
        self._watchdog_thread.join(timeout=1.0)
        self.command_source.close()


class WebGateway:
    def __init__(
        self,
        *,
        targets: Sequence[WebPolicyTarget],
        bind: str,
        web_config: Any,
        stale_timeout_s: float,
        command_hub: CommandHub | None = None,
        command_hub_factory: Callable[..., CommandHub] = CommandHub,
        command_source_factory: Callable[..., Any] | None = None,
        metrics_provider: Callable[[], Any] | object | None = None,
        server_factory: Callable[..., Any] = ThreadingHTTPServer,
        handler_factory: Callable[..., Any] = make_handler,
        thread_factory: Callable[..., Any] = threading.Thread,
        event_factory: Callable[[], Any] = threading.Event,
        log: logging.Logger = logger,
    ) -> None:
        self.targets = tuple(targets)
        self.bind = bind
        self.web_config = web_config
        self.stale_timeout_s = float(stale_timeout_s)
        self.command_hub = command_hub or command_hub_factory(
            [target.articulation for target in self.targets],
            config=self.web_config,
            stale_timeout_s=self.stale_timeout_s,
        )
        self._command_source_factory = command_source_factory
        self._metrics_provider = metrics_provider
        self._server_factory = server_factory
        self._handler_factory = handler_factory
        self._thread_factory = thread_factory
        self._event_factory = event_factory
        self._logger = log
        self._handles: list[_WebServerHandle] = []
        self._target_sources: list[tuple[WebPolicyTarget, Any]] = []

    @property
    def handles(self) -> list[_WebServerHandle]:
        return list(self._handles)

    @property
    def target_sources(self) -> list[tuple[WebPolicyTarget, Any]]:
        return list(self._target_sources)

    def start(self) -> None:
        if self._handles:
            return
        for target in self.targets:
            if self._command_source_factory is None:
                command_source = self.command_hub.port(target.articulation, source="web")
            else:
                command_source = self._command_source_factory(
                    config=self.web_config,
                    stale_timeout_s=self.stale_timeout_s,
                )
            handle = _WebServerHandle(
                bind=self.bind,
                target=target,
                command_source=command_source,
                stale_timeout_s=self.stale_timeout_s,
                server_factory=self._server_factory,
                handler_factory=self._handler_factory,
                thread_factory=self._thread_factory,
                event_factory=self._event_factory,
                metrics_provider=self._metrics_provider,
            )
            self._target_sources.append((target, command_source))
            self._handles.append(handle)
            handle.start()
            self._logger.info(
                "open http://%s:%d for %s",
                self.bind,
                target.port,
                target.articulation,
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.close()
