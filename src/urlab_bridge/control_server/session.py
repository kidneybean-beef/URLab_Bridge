from __future__ import annotations

from collections.abc import Callable
from typing import Any

from urlab_client import URLabClient


class SessionManager:
    def __init__(
        self,
        *,
        address: str,
        step_port: int,
        state_port: int,
        step_mode: str = "live",
        client_factory: Callable[..., Any] = URLabClient,
    ) -> None:
        self.address = address
        self.step_port = int(step_port)
        self.state_port = int(state_port)
        self.step_mode = step_mode
        self._client_factory = client_factory
        self._client: Any | None = None
        self._connected = False

    @property
    def client(self) -> Any:
        if self._client is None:
            raise RuntimeError("URLab session has not been connected")
        return self._client

    @property
    def runtime(self) -> Any:
        return self.client.runtime

    def connect(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(
                self.address,
                step_mode=self.step_mode,
                step_port=self.step_port,
                state_port=self.state_port,
            )
        if not self._connected:
            self._client.connect(observations="standard")
            self._connected = True
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._connected = False
