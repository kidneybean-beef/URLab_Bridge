from __future__ import annotations

from typing import Any

from urlab_tools.mjcf_quickconvert_scene import QuickConvertSceneSpec

Float3 = tuple[float, float, float]


def _literal(value: Any) -> str:
    return repr(value)


def build_unreal_import_exec(
    *,
    ue_script: str,
    urlab_bridge_root: str,
    xml_path: str,
    scene_actor_id: str,
    actor_prefix: str,
    include_visual_geoms: bool,
    include_planes: bool,
    plane_half_size: Float3,
    clean_existing: bool,
    clean_old_articulation_import: bool,
    save_level: bool,
    apply_mjcf_materials: bool,
) -> str:
    settings = {
        "URLAB_BRIDGE_ROOT": urlab_bridge_root,
        "XML_PATH": xml_path,
        "SCENE_ACTOR_ID": scene_actor_id,
        "ACTOR_PREFIX": actor_prefix,
        "INCLUDE_VISUAL_GEOMS": include_visual_geoms,
        "INCLUDE_PLANES": include_planes,
        "PLANE_HALF_SIZE": tuple(float(x) for x in plane_half_size),
        "CLEAN_EXISTING": clean_existing,
        "CLEAN_OLD_ARTICULATION_IMPORT": clean_old_articulation_import,
        "SAVE_LEVEL": save_level,
        "APPLY_MJCF_MATERIALS": apply_mjcf_materials,
    }
    lines = [f"{key} = {_literal(value)}" for key, value in settings.items()]
    lines.append(f"exec(open({_literal(ue_script)}).read())")
    return "\n".join(lines)


def quickconvert_payloads(
    scene: QuickConvertSceneSpec,
    *,
    static: bool,
    complex_mesh: bool,
    coacd_threshold: float,
    driven_by_unreal: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "target": spec.actor_id,
            "by_name": False,
            "static": bool(static),
            "complex_mesh": bool(complex_mesh),
            "coacd_threshold": float(coacd_threshold),
            "driven_by_unreal": bool(driven_by_unreal),
            "friction": tuple(float(x) for x in spec.friction),
        }
        for spec in scene.actors
        if not spec.visual_only
    ]


def apply_quickconvert_with_urlab_client(
    client: Any,
    scene: QuickConvertSceneSpec,
    *,
    static: bool,
    complex_mesh: bool,
    coacd_threshold: float,
    driven_by_unreal: bool,
) -> int:
    payloads = quickconvert_payloads(
        scene,
        static=static,
        complex_mesh=complex_mesh,
        coacd_threshold=coacd_threshold,
        driven_by_unreal=driven_by_unreal,
    )
    result = client.outliner.add_quick_convert_many(payloads)
    if result.failed:
        failures = [
            f"{item.target}: {item.error}"
            for item in result.results
            if not item.ok
        ]
        shown = "\n".join(failures[:10])
        if len(failures) > 10:
            shown += f"\n... {len(failures) - 10} more"
        raise RuntimeError(
            "Some QuickConvert actors failed:\n"
            f"{shown}"
        )
    return result.converted


def remote_discovery_error_message(
    *,
    timeout_s: float,
    multicast_group_endpoint: str,
    multicast_bind_address: str,
) -> str:
    return (
        "no UE Python remote execution node found "
        f"(timeout={timeout_s:g}s, multicast_group={multicast_group_endpoint}, "
        f"bind_address={multicast_bind_address}).\n"
        "Check UE Project Settings > Plugins > Python > Enable Remote Execution, "
        "and make sure the multicast group/bind address match this command.\n"
        "Manual fallback: in the UE Python console run:\n"
        "  exec(open('/sata/axin/URLab_Bridge/scripts/ue_import_mjcf_scene_quickconvert.py').read())\n"
        "Then rerun this terminal command with --skip-ue-spawn so URLabClient "
        "applies QuickConvert/friction through the documented outliner API."
    )
