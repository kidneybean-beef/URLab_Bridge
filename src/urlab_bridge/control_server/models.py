from __future__ import annotations

from typing import NamedTuple


class WebPolicyTarget(NamedTuple):
    articulation: str
    port: int


def parse_web_target(raw: object) -> WebPolicyTarget:
    text = str(raw)
    articulation, sep, port_text = text.rpartition(":")
    if not sep or not articulation or not port_text:
        raise SystemExit(f"invalid --web-target {text!r}; expected ARTICULATION:PORT")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise SystemExit(
            f"invalid --web-target {text!r}; port must be an integer"
        ) from exc
    if port <= 0 or port > 65535:
        raise SystemExit(f"invalid --web-target {text!r}; port out of range")
    return WebPolicyTarget(articulation=articulation, port=port)
