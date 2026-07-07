from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET


Float3 = tuple[float, float, float]
Float4 = tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneTexture:
    name: str
    texture_type: str | None = None
    builtin: str | None = None
    file: Path | None = None
    rgb1: Float3 | None = None
    rgb2: Float3 | None = None
    markrgb: Float3 | None = None
    width: int | None = None
    height: int | None = None

    def as_unreal_payload(self) -> dict:
        return {
            "name": self.name,
            "texture_type": self.texture_type,
            "builtin": self.builtin,
            "file": str(self.file) if self.file else None,
            "rgb1": list(self.rgb1) if self.rgb1 else None,
            "rgb2": list(self.rgb2) if self.rgb2 else None,
            "markrgb": list(self.markrgb) if self.markrgb else None,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class SceneMaterial:
    name: str
    texture: SceneTexture | None = None
    rgba: Float4 | None = None
    texrepeat: tuple[float, float] | None = None

    def as_unreal_payload(self) -> dict:
        return {
            "name": self.name,
            "texture": self.texture.as_unreal_payload() if self.texture else None,
            "rgba": list(self.rgba) if self.rgba else None,
            "texrepeat": list(self.texrepeat) if self.texrepeat else None,
        }


@dataclass(frozen=True)
class SceneBoxGeom:
    actor_id: str
    name: str
    location: Float3
    rotation_quat_wxyz: Float4
    half_size: Float3
    friction: Float3
    source_index: int
    visual_only: bool = False
    material: SceneMaterial | None = None
    rgba: Float4 | None = None

    def as_unreal_payload(self) -> dict:
        payload = {
            "actor_id": self.actor_id,
            "name": self.name,
            "location": list(self.location),
            "rotation_quat_wxyz": list(self.rotation_quat_wxyz),
            "half_size": list(self.half_size),
            "friction": list(self.friction),
            "source_index": self.source_index,
            "visual_only": self.visual_only,
        }
        if self.material:
            payload["material"] = self.material.as_unreal_payload()
        if self.rgba:
            payload["rgba"] = list(self.rgba)
        return payload


def load_scene_geometries(
    xml_path: str | Path,
    *,
    actor_prefix: str | None = None,
    include_visual_geoms: bool = False,
    include_planes: bool = False,
    plane_half_size: Float3 = (10.0, 10.0, 0.025),
    default_friction: Float3 = (1.0, 1.0, 1.0),
) -> list[SceneBoxGeom]:
    """Extract static box-like scene geometry from an MJCF worldbody.

    The parser reads the XML file exactly as authored and deliberately
    ignores ``<include>`` targets, so robot includes do not get expanded
    into the output. Supported geometry is ``type="box"`` plus optional
    finite box stand-ins for ``type="plane"``.
    """

    path = Path(xml_path)
    prefix = _sanitize_id(actor_prefix or path.stem)
    root = ET.parse(path).getroot()
    materials = _parse_scene_materials(root, path.parent)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return []

    out: list[SceneBoxGeom] = []
    used: set[str] = set()
    for source_index, geom in enumerate(worldbody.iter("geom")):
        geom_type = (geom.get("type") or "").strip().lower()
        visual_only = _is_visual_only(geom)
        if visual_only and not include_visual_geoms:
            continue

        if geom_type == "box":
            half_size = _parse_float_tuple(
                geom.get("size"), 3, field="size", default=None
            )
            location = _parse_float_tuple(
                geom.get("pos"), 3, field="pos", default=(0.0, 0.0, 0.0)
            )
        elif geom_type == "plane" and include_planes:
            half_size = plane_half_size
            pos = _parse_float_tuple(
                geom.get("pos"), 3, field="pos", default=(0.0, 0.0, 0.0)
            )
            location = (pos[0], pos[1], pos[2] - half_size[2])
        else:
            continue

        name = geom.get("name") or f"geom_{source_index:04d}"
        out.append(
            SceneBoxGeom(
                actor_id=_unique_actor_id(prefix, name, used),
                name=name,
                location=location,
                rotation_quat_wxyz=_parse_float_tuple(
                    geom.get("quat"),
                    4,
                    field="quat",
                    default=(1.0, 0.0, 0.0, 0.0),
                ),
                half_size=half_size,
                friction=_parse_float_tuple(
                    geom.get("friction"),
                    3,
                    field="friction",
                    default=default_friction,
                ),
                source_index=source_index,
                visual_only=visual_only,
                material=materials.get((geom.get("material") or "").strip()),
                rgba=_parse_optional_float_tuple(
                    geom.get("rgba"),
                    4,
                    field="rgba",
                ),
            )
        )
    return out


def _parse_scene_materials(root: ET.Element, xml_dir: Path) -> dict[str, SceneMaterial]:
    textures: dict[str, SceneTexture] = {}
    materials: dict[str, SceneMaterial] = {}
    for asset in root.findall("asset"):
        for texture in asset.findall("texture"):
            name = (texture.get("name") or "").strip()
            if not name:
                continue
            textures[name] = SceneTexture(
                name=name,
                texture_type=_clean_optional_string(texture.get("type")),
                builtin=_clean_optional_string(texture.get("builtin")),
                file=_resolve_texture_file(xml_dir, texture.get("file")),
                rgb1=_parse_optional_float_tuple(
                    texture.get("rgb1"),
                    3,
                    field=f"texture {name} rgb1",
                ),
                rgb2=_parse_optional_float_tuple(
                    texture.get("rgb2"),
                    3,
                    field=f"texture {name} rgb2",
                ),
                markrgb=_parse_optional_float_tuple(
                    texture.get("markrgb"),
                    3,
                    field=f"texture {name} markrgb",
                ),
                width=_parse_optional_int(texture.get("width")),
                height=_parse_optional_int(texture.get("height")),
            )

        for material in asset.findall("material"):
            name = (material.get("name") or "").strip()
            if not name:
                continue
            texture_name = (material.get("texture") or "").strip()
            materials[name] = SceneMaterial(
                name=name,
                texture=textures.get(texture_name),
                rgba=_parse_optional_float_tuple(
                    material.get("rgba"),
                    4,
                    field=f"material {name} rgba",
                ),
                texrepeat=_parse_optional_float_tuple(
                    material.get("texrepeat"),
                    2,
                    field=f"material {name} texrepeat",
                ),
            )
    return materials


def _parse_float_tuple(
    value: str | None,
    count: int,
    *,
    field: str,
    default: tuple[float, ...] | None,
) -> tuple[float, ...]:
    if value is None or not value.strip():
        if default is None:
            raise ValueError(f"missing required MJCF geom '{field}'")
        return default
    parts = value.split()
    if len(parts) != count:
        raise ValueError(
            f"MJCF geom '{field}' expected {count} floats, got {len(parts)}: {value!r}"
        )
    return tuple(float(part) for part in parts)


def _parse_optional_float_tuple(
    value: str | None,
    count: int,
    *,
    field: str,
) -> tuple[float, ...] | None:
    if value is None or not value.strip():
        return None
    return _parse_float_tuple(value, count, field=field, default=None)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _clean_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _resolve_texture_file(xml_dir: Path, value: str | None) -> Path | None:
    value = _clean_optional_string(value)
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (xml_dir / path).resolve()


def _is_visual_only(geom: ET.Element) -> bool:
    return geom.get("contype") == "0" and geom.get("conaffinity") == "0"


def _sanitize_id(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "mjcf_geom"


def _unique_actor_id(prefix: str, name: str, used: set[str]) -> str:
    base = f"{prefix}_{_sanitize_id(name)}"
    actor_id = base
    suffix = 1
    while actor_id in used:
        actor_id = f"{base}_{suffix:02d}"
        suffix += 1
    used.add(actor_id)
    return actor_id
