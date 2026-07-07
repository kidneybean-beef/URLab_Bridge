from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


ACTOR_ID_TAG_PREFIX = "URLab.ActorId="


@dataclass(frozen=True)
class MaterialImportConfig:
    apply_mjcf_materials: bool
    material_dest_path: str
    texture_dest_path: str
    master_material_path: str
    checker_material_path: str
    default_material_path: str


@dataclass(frozen=True)
class SceneSpawnResult:
    folder_path: str
    spawned_actors: int
    total_actors: int
    materials_applied: int
    removed_old_actors: int


def spawn_scene_in_unreal(
    unreal: Any,
    scene: Any,
    *,
    xml_stem: str,
    clean_existing: bool,
    clean_old_articulation_import: bool,
    save_level: bool,
    material_config: MaterialImportConfig,
) -> SceneSpawnResult:
    removed = _cleanup_existing(
        unreal,
        scene,
        xml_stem=xml_stem,
        clean_existing=clean_existing,
        clean_old_articulation_import=clean_old_articulation_import,
    )
    cube_mesh = unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    if not cube_mesh:
        raise RuntimeError("failed to load /Engine/BasicShapes/Cube.Cube")

    errors = []
    spawned = 0
    materials_applied = 0
    material_cache: dict[str, Any] = {}
    texture_cache: dict[str, Any] = {}
    for spec in scene.actors:
        try:
            _, material_applied = _spawn_or_update_step(
                unreal,
                cube_mesh,
                "",
                spec,
                material_config,
                material_cache,
                texture_cache,
            )
            spawned += 1
            if material_applied:
                materials_applied += 1
        except Exception as exc:
            errors.append(f"{spec.actor_id}: {exc}")

    if save_level:
        unreal.EditorLevelLibrary.save_current_level()

    if errors:
        raise RuntimeError(
            "Some QuickConvert stair actors failed:\n" + _format_errors(errors)
        )
    return SceneSpawnResult(
        folder_path="",
        spawned_actors=spawned,
        total_actors=len(scene.actors),
        materials_applied=materials_applied,
        removed_old_actors=removed,
    )


def _mj_pos_to_ue(unreal: Any, pos: tuple[float, float, float]) -> Any:
    return unreal.Vector(pos[0] * 100.0, -pos[1] * 100.0, pos[2] * 100.0)


def _mj_quat_to_ue_rotator(unreal: Any, quat_wxyz: tuple[float, float, float, float]) -> Any:
    quat = unreal.Quat(
        -quat_wxyz[1],
        quat_wxyz[2],
        -quat_wxyz[3],
        quat_wxyz[0],
    )
    return quat.rotator()


def _tag_value(actor_id: str) -> str:
    return ACTOR_ID_TAG_PREFIX + actor_id


def _actor_tags(actor: Any) -> list[str]:
    return [str(item) for item in actor.tags]


def _actor_has_id(actor: Any, actor_id: str) -> bool:
    return actor.get_actor_label() == actor_id or _tag_value(actor_id) in _actor_tags(actor)


def _actor_folder_path(actor: Any) -> str:
    try:
        return str(actor.get_folder_path())
    except Exception:
        pass
    try:
        return str(actor.get_editor_property("folder_path"))
    except Exception:
        return ""


def _all_level_actors(unreal: Any) -> list[Any]:
    try:
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        if subsystem:
            return list(subsystem.get_all_level_actors())
    except Exception:
        pass
    return list(unreal.EditorLevelLibrary.get_all_level_actors())


def _find_actor(unreal: Any, actor_id: str) -> Any | None:
    for actor in _all_level_actors(unreal):
        if _actor_has_id(actor, actor_id):
            return actor
    return None


def _destroy_actor(unreal: Any, actor: Any) -> bool:
    try:
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        if subsystem:
            return bool(subsystem.destroy_actor(actor))
    except Exception:
        pass
    return bool(unreal.EditorLevelLibrary.destroy_actor(actor))


def _spawn_actor_from_class(unreal: Any, actor_class: Any, location: Any, rotation: Any) -> Any:
    try:
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        if subsystem:
            return subsystem.spawn_actor_from_class(actor_class, location, rotation, False)
    except Exception:
        pass
    return unreal.EditorLevelLibrary.spawn_actor_from_class(actor_class, location, rotation)


def _tag_actor(unreal: Any, actor: Any, actor_id: str) -> None:
    tag = unreal.Name(_tag_value(actor_id))
    tags = list(actor.tags)
    if tag not in tags:
        tags.append(tag)
        actor.tags = tags
    actor.set_actor_label(actor_id)


def _set_folder_path(unreal: Any, actor: Any, folder: str) -> None:
    try:
        actor.set_folder_path(unreal.Name(folder))
    except Exception:
        try:
            actor.set_folder_path(folder)
        except Exception:
            pass


def _set_property(obj: Any, names: tuple[str, ...], value: Any) -> None:
    errors = []
    for name in names:
        try:
            obj.set_editor_property(name, value)
            return
        except Exception as exc:
            errors.append(f"{name}: {exc}")
        try:
            setattr(obj, name, value)
            return
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("; ".join(errors))


def _get_component_by_class(unreal: Any, actor: Any, component_class: Any) -> Any | None:
    method = getattr(actor, "get_component_by_class", None)
    if method:
        try:
            component = method(component_class)
            if component:
                return component
        except Exception:
            pass
    method = getattr(actor, "get_components_by_class", None)
    if method:
        try:
            components = list(method(component_class))
            if components:
                return components[0]
        except Exception:
            pass
    try:
        components = []
        actor.get_components(component_class, components)
        if components:
            return components[0]
    except Exception:
        pass
    return None


def _get_static_mesh_component(unreal: Any, actor: Any) -> Any:
    component = _get_component_by_class(unreal, actor, unreal.StaticMeshComponent)
    if component:
        return component
    for prop_name in ("static_mesh_component", "StaticMeshComponent"):
        try:
            component = actor.get_editor_property(prop_name)
            if component:
                return component
        except Exception:
            pass
    raise RuntimeError(f"no StaticMeshComponent on {actor.get_actor_label()}")


def _cleanup_existing(
    unreal: Any,
    scene: Any,
    *,
    xml_stem: str,
    clean_existing: bool,
    clean_old_articulation_import: bool,
) -> int:
    if not clean_existing:
        return 0
    ids = {scene.scene_actor_id, *(spec.actor_id for spec in scene.actors)}
    removed = 0
    for actor in list(_all_level_actors(unreal)):
        label = actor.get_actor_label()
        tags = set(_actor_tags(actor))
        has_scene_id = label in ids or any(_tag_value(actor_id) in tags for actor_id in ids)
        is_in_scene_folder = _actor_folder_path(actor).strip("/") == scene.scene_actor_id
        is_old_articulation_scene = (
            clean_old_articulation_import
            and label == xml_stem
            and "MjArticulation" in actor.get_class().get_name()
        )
        if has_scene_id or is_in_scene_folder or is_old_articulation_scene:
            if _destroy_actor(unreal, actor):
                removed += 1
    return removed


def _sanitize_asset_name(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "mjcf_material"


def _ensure_asset_directory(unreal: Any, path: str) -> None:
    try:
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            unreal.EditorAssetLibrary.make_directory(path)
    except Exception:
        pass


def _load_asset(unreal: Any, asset_path: str) -> Any | None:
    try:
        asset = unreal.load_asset(asset_path)
        if asset:
            return asset
    except Exception:
        pass
    return None


def _rgba_to_linear_color(unreal: Any, rgba: tuple[float, float, float, float]) -> Any:
    return unreal.LinearColor(
        float(rgba[0]),
        float(rgba[1]),
        float(rgba[2]),
        float(rgba[3]),
    )


# def _visual_rgba(spec: Any) -> tuple[float, float, float, float]:
#     if spec.rgba:
#         return spec.rgba
#     material = spec.material
#     if material and material.rgba:
#         return material.rgba
#     texture = material.texture if material else None
#     if texture and texture.rgb1:
#         return (texture.rgb1[0], texture.rgb1[1], texture.rgb1[2], 1.0)
#     return (0.72, 0.72, 0.72, 1.0)

def _visual_rgba(spec: Any) -> tuple[float, float, float, float]:
    if spec.rgba:
        return spec.rgba
    material = spec.material
    texture = material.texture if material else None
    if texture and texture.rgb1 and not texture.file:
        return (texture.rgb1[0], texture.rgb1[1], texture.rgb1[2], 1.0)
    if material and material.rgba:
        return material.rgba
    if texture and texture.rgb1:
        return (texture.rgb1[0], texture.rgb1[1], texture.rgb1[2], 1.0)
    return (0.72, 0.72, 0.72, 1.0)


def _texture_cache_key(texture: Any) -> str:
    if texture.file:
        return f"file:{texture.file}"
    return f"builtin:{texture.name}:{texture.builtin}:{texture.rgb1}:{texture.rgb2}"


def _material_cache_key(spec: Any) -> str:
    material = spec.material
    if not material:
        return f"rgba:{spec.rgba}"
    texture_key = _texture_cache_key(material.texture) if material.texture else "none"
    return f"material:{material.name}:{texture_key}:{material.rgba}:{spec.rgba}"


def _import_texture(
    unreal: Any,
    texture: Any,
    config: MaterialImportConfig,
    cache: dict[str, Any],
) -> Any | None:
    if not texture or not texture.file:
        return None
    key = _texture_cache_key(texture)
    if key in cache:
        return cache[key]

    texture_path = Path(texture.file)
    if not texture_path.exists():
        print(f"URLab MJCF material warning: texture file not found: {texture_path}")
        cache[key] = None
        return None

    _ensure_asset_directory(unreal, config.texture_dest_path)
    asset_name = _sanitize_asset_name(texture_path.stem)
    existing = _load_asset(unreal, f"{config.texture_dest_path}/{asset_name}")
    if existing:
        cache[key] = existing
        return existing

    data = unreal.AutomatedAssetImportData()
    _set_property(data, ("DestinationPath", "destination_path"), config.texture_dest_path)
    _set_property(data, ("Filenames", "filenames"), [str(texture_path)])
    _set_property(data, ("bReplaceExisting", "replace_existing"), True)

    imported = []
    try:
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
        imported = list(asset_tools.import_assets_automated(data) or [])
    except Exception as exc:
        print(f"URLab MJCF material warning: failed to import {texture_path}: {exc}")

    asset = imported[0] if imported else _load_asset(unreal, f"{config.texture_dest_path}/{asset_name}")
    cache[key] = asset
    return asset


def _set_material_instance_params(
    unreal: Any,
    material_instance: Any,
    color: Any,
    texture_asset: Any | None,
) -> None:
    edit = unreal.MaterialEditingLibrary
    try:
        edit.set_material_instance_vector_parameter_value(
            material_instance,
            "BaseColor",
            color,
        )
    except Exception as exc:
        print(f"URLab MJCF material warning: could not set BaseColor: {exc}")
    if texture_asset:
        try:
            edit.set_material_instance_texture_parameter_value(
                material_instance,
                "BaseColorTexture",
                texture_asset,
            )
        except Exception as exc:
            print(f"URLab MJCF material warning: could not set texture: {exc}")
    set_switch = getattr(edit, "set_material_instance_static_switch_parameter_value", None)
    if set_switch:
        try:
            set_switch(material_instance, "bUseTexture", bool(texture_asset))
        except Exception as exc:
            print(f"URLab MJCF material warning: could not set bUseTexture: {exc}")
    try:
        edit.update_material_instance(material_instance)
    except Exception:
        pass


def _make_material_instance(
    unreal: Any,
    asset_name: str,
    color: Any,
    texture_asset: Any | None,
    config: MaterialImportConfig,
) -> Any | None:
    master_material = _load_asset(unreal, config.master_material_path)
    if not master_material:
        print(
            "URLab MJCF material warning: "
            f"missing master material {config.master_material_path}"
        )
        return None

    _ensure_asset_directory(unreal, config.material_dest_path)
    existing = _load_asset(unreal, f"{config.material_dest_path}/{asset_name}")
    if existing:
        try:
            _set_material_instance_params(unreal, existing, color, texture_asset)
        except Exception as exc:
            print(f"URLab MJCF material warning: failed to update {asset_name}: {exc}")
        return existing

    try:
        factory = unreal.MaterialInstanceConstantFactoryNew()
        material_instance = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            asset_name,
            config.material_dest_path,
            unreal.MaterialInstanceConstant,
            factory,
        )
        if not material_instance:
            return None
        try:
            unreal.MaterialEditingLibrary.set_material_instance_parent(
                material_instance,
                master_material,
            )
        except Exception:
            material_instance.set_editor_property("parent", master_material)
        _set_material_instance_params(unreal, material_instance, color, texture_asset)
        try:
            unreal.EditorAssetLibrary.save_asset(f"{config.material_dest_path}/{asset_name}")
        except Exception:
            pass
        return material_instance
    except Exception as exc:
        print(f"URLab MJCF material warning: failed to create {asset_name}: {exc}")
        return None


def _dynamic_fallback_material(
    unreal: Any,
    outer: Any,
    color: Any,
    texture_asset: Any | None,
    config: MaterialImportConfig,
) -> Any | None:
    master_material = _load_asset(unreal, config.master_material_path)
    if not master_material:
        return None
    try:
        dynamic_material = unreal.MaterialInstanceDynamic.create(master_material, outer)
        dynamic_material.set_vector_parameter_value("BaseColor", color)
        if texture_asset:
            dynamic_material.set_texture_parameter_value(
                "BaseColorTexture",
                texture_asset,
            )
        return dynamic_material
    except Exception:
        return None


def _resolve_visual_material(
    unreal: Any,
    actor: Any,
    spec: Any,
    config: MaterialImportConfig,
    material_cache: dict[str, Any],
    texture_cache: dict[str, Any],
) -> Any | None:
    if not config.apply_mjcf_materials:
        return None
    if not spec.material and not spec.rgba:
        return None

    key = _material_cache_key(spec)
    if key in material_cache:
        return material_cache[key]

    material = spec.material
    texture = material.texture if material else None
    builtin = (texture.builtin or "").lower() if texture else ""
    if builtin == "checker":
        checker_material = _load_asset(unreal, config.checker_material_path)
        if checker_material:
            material_cache[key] = checker_material
            return checker_material

    color = _rgba_to_linear_color(unreal, _visual_rgba(spec))
    texture_asset = _import_texture(unreal, texture, config, texture_cache) if texture else None
    name_hint = material.name if material else spec.actor_id
    material_name = f"MI_{_sanitize_asset_name(name_hint)}"
    material_asset = _make_material_instance(unreal, material_name, color, texture_asset, config)
    if not material_asset:
        material_asset = _dynamic_fallback_material(unreal, actor, color, texture_asset, config)
    if not material_asset:
        material_asset = _load_asset(unreal, config.default_material_path)
    material_cache[key] = material_asset
    return material_asset


def _apply_visual_material(
    unreal: Any,
    actor: Any,
    mesh_component: Any,
    spec: Any,
    config: MaterialImportConfig,
    material_cache: dict[str, Any],
    texture_cache: dict[str, Any],
) -> bool:
    material = _resolve_visual_material(
        unreal,
        actor,
        spec,
        config,
        material_cache,
        texture_cache,
    )
    if not material:
        return False
    mesh_component.set_material(0, material)
    return True


def _spawn_or_update_step(
    unreal: Any,
    cube_mesh: Any,
    folder_path: str,
    spec: Any,
    material_config: MaterialImportConfig,
    material_cache: dict[str, Any],
    texture_cache: dict[str, Any],
) -> tuple[Any, bool]:
    location = _mj_pos_to_ue(unreal, spec.location)
    rotation = _mj_quat_to_ue_rotator(unreal, spec.rotation_quat_wxyz)
    actor = _find_actor(unreal, spec.actor_id)
    if actor is None:
        actor = _spawn_actor_from_class(unreal, unreal.StaticMeshActor, location, rotation)
    else:
        actor.set_actor_location(location, False, False)
        actor.set_actor_rotation(rotation, False)
    if not actor:
        raise RuntimeError(f"failed to spawn {spec.actor_id}")

    _tag_actor(unreal, actor, spec.actor_id)
    if folder_path:
        _set_folder_path(unreal, actor, folder_path)
    actor.set_actor_scale3d(
        unreal.Vector(
            float(spec.scale_xyz[0]),
            float(spec.scale_xyz[1]),
            float(spec.scale_xyz[2]),
        )
    )

    mesh_component = _get_static_mesh_component(unreal, actor)
    mesh_component.set_static_mesh(cube_mesh)
    try:
        mesh_component.set_mobility(unreal.ComponentMobility.STATIC)
    except Exception:
        pass
    material_applied = _apply_visual_material(
        unreal,
        actor,
        mesh_component,
        spec,
        material_config,
        material_cache,
        texture_cache,
    )
    return actor, material_applied


def _format_errors(errors: list[str], limit: int = 10) -> str:
    shown = "\n".join(errors[:limit])
    if len(errors) > limit:
        shown += f"\n... {len(errors) - limit} more"
    return shown
