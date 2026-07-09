from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .models import WebPolicyTarget, parse_web_target


class RobotRegistry:
    def __init__(self, targets: Sequence[WebPolicyTarget]) -> None:
        self._targets = tuple(targets)
        self._control_articulations: tuple[str, ...] = tuple(
            target.articulation for target in self._targets
        )
        self._validate_unique()

    @classmethod
    def from_arg_strings(cls, raw_targets: Sequence[object]) -> "RobotRegistry":
        return cls([parse_web_target(raw) for raw in raw_targets])

    @classmethod
    def from_args(cls, args: object) -> "RobotRegistry":
        raw_targets = list(getattr(args, "web_target", []) or [])
        articulation = str(getattr(args, "articulation", "") or "")
        if not raw_targets and articulation:
            raw_targets = [f"{articulation}:{int(getattr(args, 'web_port'))}"]
        if not raw_targets:
            raise SystemExit("pass at least one --web-target ARTICULATION:PORT")
        return cls.from_arg_strings(raw_targets)

    @property
    def targets(self) -> tuple[WebPolicyTarget, ...]:
        return self._targets

    @property
    def control_articulations(self) -> tuple[str, ...]:
        return self._control_articulations

    def resolve(
        self,
        client: Any,
        resolver: Callable[[Any, str], str],
    ) -> tuple[str, ...]:
        self._control_articulations = tuple(
            resolver(client, target.articulation) for target in self._targets
        )
        return self._control_articulations

    def _validate_unique(self) -> None:
        seen_articulations: set[str] = set()
        seen_ports: set[int] = set()
        for target in self._targets:
            if target.articulation in seen_articulations:
                raise SystemExit(f"duplicate articulation target {target.articulation!r}")
            if target.port in seen_ports:
                raise SystemExit(f"duplicate web port {target.port}")
            seen_articulations.add(target.articulation)
            seen_ports.add(target.port)
