from __future__ import annotations

import select
import sys
import termios
import tty
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

import numpy as np


@dataclass(frozen=True)
class KeyboardCommandConfig:
    step_vx: float = 0.1
    step_vy: float = 0.1
    step_yaw: float = 0.2
    max_vx: float = 0.8
    max_vy: float = 0.6
    max_yaw: float = 1.0

    @property
    def limits(self) -> np.ndarray:
        return np.array([self.max_vx, self.max_vy, self.max_yaw], dtype=np.float32)


class FixedCommandSource:
    def __init__(self, command: Sequence[float]) -> None:
        self._command = _checked_command(command)

    def poll(self) -> np.ndarray:
        return self._command.copy()

    @property
    def quit_requested(self) -> bool:
        return False

    def close(self) -> None:
        return None


class URLabTwistCommandSource:
    def __init__(
        self,
        articulation: object,
        *,
        axis_signs: Sequence[float] = (1.0, -1.0, -1.0),
    ) -> None:
        self._articulation = articulation
        self._axis_signs = _checked_command(axis_signs)

    def poll(self) -> np.ndarray:
        lin = np.asarray(getattr(self._articulation, "twist_linear"), dtype=np.float32)
        ang = np.asarray(getattr(self._articulation, "twist_angular"), dtype=np.float32)
        if lin.shape != (3,):
            raise ValueError(f"twist_linear must have shape (3,), got {lin.shape}")
        if ang.shape != (3,):
            raise ValueError(f"twist_angular must have shape (3,), got {ang.shape}")
        command = np.array([lin[0], lin[1], ang[2]], dtype=np.float32)
        command *= self._axis_signs
        if not np.all(np.isfinite(command)):
            raise ValueError(f"twist command must be finite, got {command}")
        return command

    @property
    def quit_requested(self) -> bool:
        return False

    def close(self) -> None:
        return None


class KeyboardCommandSource:
    def __init__(
        self,
        *,
        initial_command: Sequence[float] = (0.0, 0.0, 0.0),
        config: KeyboardCommandConfig | None = None,
        reader: object | None = None,
    ) -> None:
        self.config = config or KeyboardCommandConfig()
        self.reader = reader
        self._command = _checked_command(initial_command)
        self._quit_requested = False
        self._clip()

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def poll(self) -> np.ndarray:
        if self.reader is not None:
            for key in self.reader.poll_keys():
                self.handle_key(str(key))
        return self._command.copy()

    def handle_key(self, key: str) -> None:
        key = key.lower()
        if key == "w":
            self._command[0] += self.config.step_vx
        elif key == "s":
            self._command[0] -= self.config.step_vx
        elif key == "a":
            self._command[1] += self.config.step_vy
        elif key == "d":
            self._command[1] -= self.config.step_vy
        elif key == "q":
            self._command[2] += self.config.step_yaw
        elif key == "e":
            self._command[2] -= self.config.step_yaw
        elif key in (" ", "r", "x"):
            self._command[:] = 0.0
        elif key in ("\x1b", "\x03"):
            self._quit_requested = True
        self._clip()

    def close(self) -> None:
        close = getattr(self.reader, "close", None)
        if callable(close):
            close()

    def _clip(self) -> None:
        limits = self.config.limits
        self._command[:] = np.clip(self._command, -limits, limits)


class TerminalKeyReader:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdin
        self._fd = self._stream.fileno()
        self._old_settings: list[int | bytes] | None = None

    def __enter__(self) -> TerminalKeyReader:
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def start(self) -> None:
        if not self._stream.isatty():
            raise RuntimeError("keyboard control requires an interactive terminal")
        if self._old_settings is None:
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

    def poll_keys(self) -> list[str]:
        keys: list[str] = []
        while select.select([self._stream], [], [], 0.0)[0]:
            keys.append(self._stream.read(1))
        return keys

    def close(self) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None


def format_keyboard_help(config: KeyboardCommandConfig) -> str:
    return (
        "keyboard: W/S vx +/-{:.2f}, A/D vy +/-{:.2f}, Q/E yaw +/-{:.2f}, "
        "Space/R/X zero, Esc quit; limits vx={:.2f}, vy={:.2f}, yaw={:.2f}"
    ).format(
        config.step_vx,
        config.step_vy,
        config.step_yaw,
        config.max_vx,
        config.max_vy,
        config.max_yaw,
    )


def _checked_command(command: Sequence[float]) -> np.ndarray:
    arr = np.asarray(command, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"command must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"command must be finite, got {arr}")
    return arr.copy()
