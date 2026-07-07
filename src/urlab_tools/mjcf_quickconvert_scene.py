from __future__ import annotations

from dataclasses import dataclass

from urlab_tools.mjcf_scene_geoms import Float3, Float4, SceneBoxGeom, SceneMaterial


@dataclass(frozen=True)
class QuickConvertBoxActorSpec:
    actor_id: str
    source_actor_id: str
    location: Float3
    rotation_quat_wxyz: Float4
    scale_xyz: Float3
    friction: Float3
    visual_only: bool = False
    material: SceneMaterial | None = None
    rgba: Float4 | None = None


@dataclass(frozen=True)
class QuickConvertSceneSpec:
    scene_actor_id: str
    actors: tuple[QuickConvertBoxActorSpec, ...]


def build_quickconvert_scene(
    geoms: list[SceneBoxGeom],
    *,
    scene_actor_id: str,
    actor_prefix: str,
) -> QuickConvertSceneSpec:
    actors = []
    for index, geom in enumerate(geoms, start=1):
        actors.append(
            QuickConvertBoxActorSpec(
                actor_id=f"{actor_prefix}_{index:04d}_{geom.actor_id}",
                source_actor_id=geom.actor_id,
                location=geom.location,
                rotation_quat_wxyz=geom.rotation_quat_wxyz,
                scale_xyz=(
                    geom.half_size[0] * 2.0,
                    geom.half_size[1] * 2.0,
                    geom.half_size[2] * 2.0,
                ),
                friction=geom.friction,
                visual_only=geom.visual_only,
                material=geom.material,
                rgba=geom.rgba,
            )
        )
    return QuickConvertSceneSpec(
        scene_actor_id=scene_actor_id,
        actors=tuple(actors),
    )
