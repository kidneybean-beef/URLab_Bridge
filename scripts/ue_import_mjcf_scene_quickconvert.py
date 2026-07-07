#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time
from typing import Any


IMPORT_VERSION = "2026-07-03-urlab-client-qc-v9"
URLAB_BRIDGE_ROOT = Path(globals().get("URLAB_BRIDGE_ROOT", "/sata/axin/URLab_Bridge"))
ENGINE_PYTHON_PLUGIN_PATH = Path(
    globals().get(
        "ENGINE_PYTHON_PLUGIN_PATH",
        "/sata/app/UnrealEngine-bin-5.7.4/Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python",
    )
)
XML_PATH = Path(
    globals().get(
        "XML_PATH",
        # "/sata/axin/robot_controllers/go2_rl_gym/resources/robots/go2/stairs.xml",
        "/sata/axin/robot_controllers/go2_rl_gym/resources/robots/go2/cross_stairs.xml",
    )
)
SCENE_ACTOR_ID = str(
    globals().get("SCENE_ACTOR_ID", f"{XML_PATH.stem}_quickconvert")
)
ACTOR_PREFIX = str(globals().get("ACTOR_PREFIX", f"{XML_PATH.stem}_qc"))
INCLUDE_VISUAL_GEOMS = bool(globals().get("INCLUDE_VISUAL_GEOMS", False))
INCLUDE_PLANES = bool(globals().get("INCLUDE_PLANES", True))
PLANE_HALF_SIZE = tuple(globals().get("PLANE_HALF_SIZE", (10.0, 10.0, 0.025)))
SAVE_LEVEL = bool(globals().get("SAVE_LEVEL", False))
CLEAN_EXISTING = bool(globals().get("CLEAN_EXISTING", True))
CLEAN_OLD_ARTICULATION_IMPORT = bool(
    globals().get("CLEAN_OLD_ARTICULATION_IMPORT", True)
)
APPLY_MJCF_MATERIALS = bool(globals().get("APPLY_MJCF_MATERIALS", True))
MATERIAL_ROOT = str(
    globals().get("MATERIAL_ROOT", f"/Game/URLabImportedMJCF/{SCENE_ACTOR_ID}")
)
MATERIAL_DEST_PATH = str(
    globals().get("MATERIAL_DEST_PATH", f"{MATERIAL_ROOT}/Materials")
)
TEXTURE_DEST_PATH = str(
    globals().get("TEXTURE_DEST_PATH", f"{MATERIAL_ROOT}/Textures")
)
MASTER_MATERIAL_PATH = str(
    globals().get(
        "MASTER_MATERIAL_PATH",
        "/UnrealRoboticsLab/Materials/M_MuJoCo_Master.M_MuJoCo_Master",
    )
)
CHECKER_MATERIAL_PATH = str(
    globals().get(
        "CHECKER_MATERIAL_PATH",
        "/Engine/EngineMaterials/WorldGridMaterial.WorldGridMaterial",
    )
)
DEFAULT_MATERIAL_PATH = str(
    globals().get(
        "DEFAULT_MATERIAL_PATH",
        "/Engine/EngineMaterials/DefaultMaterial.DefaultMaterial",
    )
)

def _ensure_urlab_tools_path() -> None:
    src = URLAB_BRIDGE_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _run_inside_unreal() -> None:
    import unreal

    _ensure_urlab_tools_path()
    from urlab_tools.mjcf_quickconvert_scene import build_quickconvert_scene
    from urlab_tools.mjcf_scene_geoms import load_scene_geometries
    from urlab_tools.ue_mjcf_scene_spawn import (
        MaterialImportConfig,
        spawn_scene_in_unreal,
    )

    print(
        "URLab MJCF QuickConvert scene importer: "
        f"version={IMPORT_VERSION} file={Path(__file__).resolve() if '__file__' in globals() else 'exec-string'}"
    )

    geoms = load_scene_geometries(
        XML_PATH,
        actor_prefix=XML_PATH.stem,
        include_visual_geoms=INCLUDE_VISUAL_GEOMS,
        include_planes=INCLUDE_PLANES,
        plane_half_size=PLANE_HALF_SIZE,  # type: ignore[arg-type]
    )
    if not geoms:
        raise RuntimeError(f"no supported scene geometry found in {XML_PATH}")

    scene = build_quickconvert_scene(
        geoms,
        scene_actor_id=SCENE_ACTOR_ID,
        actor_prefix=ACTOR_PREFIX,
    )
    result = spawn_scene_in_unreal(
        unreal,
        scene,
        xml_stem=XML_PATH.stem,
        clean_existing=CLEAN_EXISTING,
        clean_old_articulation_import=CLEAN_OLD_ARTICULATION_IMPORT,
        save_level=SAVE_LEVEL,
        material_config=MaterialImportConfig(
            apply_mjcf_materials=APPLY_MJCF_MATERIALS,
            material_dest_path=MATERIAL_DEST_PATH,
            texture_dest_path=TEXTURE_DEST_PATH,
            master_material_path=MASTER_MATERIAL_PATH,
            checker_material_path=CHECKER_MATERIAL_PATH,
            default_material_path=DEFAULT_MATERIAL_PATH,
        ),
    )

    print(
        "URLab MJCF QuickConvert scene import: "
        f"scene_id={SCENE_ACTOR_ID} "
        f"spawned_actors={result.spawned_actors}/{result.total_actors} "
        f"materials_applied={result.materials_applied} "
        f"removed_old_actors={result.removed_old_actors}"
    )


def _parse_float3(value: str) -> tuple[float, float, float]:
    parts = [part for part in re.split(r"[,\s]+", value.strip()) if part]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three floats")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _parse_endpoint(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("expected endpoint in IP:PORT form")
    return (host, int(port))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import MJCF box/plane scene geoms into UE, then apply URLab "
            "QuickConvert through the documented Python outliner API."
        )
    )
    parser.add_argument("--xml", default=str(XML_PATH), help="MJCF scene XML path")
    parser.add_argument("--address", default="tcp://127.0.0.1", help="URLab bridge address")
    parser.add_argument("--step-port", type=int, default=5559, help="URLab bridge RPC port")
    parser.add_argument("--remote-timeout", type=float, default=10.0, help="UE remote node discovery timeout")
    parser.add_argument("--remote-node", default="", help="Specific UE remote execution node id")
    parser.add_argument(
        "--remote-group-endpoint",
        default="239.0.0.1:6766",
        help="UE Python Remote Execution multicast group endpoint",
    )
    parser.add_argument(
        "--remote-bind-address",
        default="127.0.0.1",
        help="UE Python Remote Execution multicast bind address",
    )
    parser.add_argument(
        "--remote-command-endpoint",
        default="127.0.0.1:6776",
        help="Local TCP endpoint for UE Python Remote Execution command replies",
    )
    parser.add_argument("--engine-python-plugin-path", default=str(ENGINE_PYTHON_PLUGIN_PATH))
    parser.add_argument("--urlab-bridge-root", default=str(URLAB_BRIDGE_ROOT))
    parser.add_argument("--scene-actor-id", default="", help="Scene folder/id; defaults to <xml_stem>_quickconvert")
    parser.add_argument("--actor-prefix", default="", help="Child actor id prefix; defaults to <xml_stem>_qc")
    parser.add_argument("--include-visual-geoms", action="store_true")
    parser.add_argument("--no-planes", action="store_true")
    parser.add_argument("--plane-half-size", type=_parse_float3, default=PLANE_HALF_SIZE)
    parser.add_argument("--no-clean-existing", action="store_true")
    parser.add_argument("--keep-old-articulation-import", action="store_true")
    parser.add_argument("--save-level", action="store_true")
    parser.add_argument("--complex-mesh", action="store_true")
    parser.add_argument("--coacd-threshold", type=float, default=0.05)
    parser.add_argument("--driven-by-unreal", action="store_true")
    parser.add_argument("--no-materials", action="store_true")
    parser.add_argument(
        "--skip-ue-spawn",
        action="store_true",
        help="Only apply QuickConvert via URLab client to actors already in the level.",
    )
    return parser


def _run_unreal_remote_command(
    command: str,
    *,
    plugin_path: str,
    timeout_s: float,
    node_id: str = "",
    multicast_group_endpoint: str = "239.0.0.1:6766",
    multicast_bind_address: str = "127.0.0.1",
    command_endpoint: str = "127.0.0.1:6776",
) -> dict[str, Any]:
    if plugin_path not in sys.path:
        sys.path.insert(0, plugin_path)
    import remote_execution

    config = remote_execution.RemoteExecutionConfig()
    config.multicast_group_endpoint = _parse_endpoint(multicast_group_endpoint)
    config.multicast_bind_address = multicast_bind_address
    config.command_endpoint = _parse_endpoint(command_endpoint)
    remote = remote_execution.RemoteExecution(config)
    remote.start()
    try:
        deadline = time.monotonic() + timeout_s
        while not remote.remote_nodes and time.monotonic() < deadline:
            time.sleep(0.1)
        nodes = list(remote.remote_nodes)
        if not nodes:
            from urlab_tools.mjcf_quickconvert_workflow import remote_discovery_error_message

            raise RuntimeError(
                remote_discovery_error_message(
                    timeout_s=timeout_s,
                    multicast_group_endpoint=multicast_group_endpoint,
                    multicast_bind_address=multicast_bind_address,
                )
            )
        selected = None
        if node_id:
            selected = next((node for node in nodes if node.get("node_id") == node_id), None)
            if selected is None:
                raise RuntimeError(f"UE remote node {node_id!r} not found")
        else:
            selected = nodes[0]
        remote.open_command_connection(selected["node_id"])
        return dict(
            remote.run_command(
                command,
                unattended=True,
                exec_mode=remote_execution.MODE_EXEC_FILE,
                raise_on_failure=True,
            )
        )
    finally:
        remote.stop()


def _run_external() -> None:
    args = _build_arg_parser().parse_args()
    bridge_root = Path(args.urlab_bridge_root)
    src = bridge_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from urlab_client import URLabClient
    from urlab_tools.mjcf_quickconvert_scene import build_quickconvert_scene
    from urlab_tools.mjcf_quickconvert_workflow import (
        apply_quickconvert_with_urlab_client,
        build_unreal_import_exec,
    )
    from urlab_tools.mjcf_scene_geoms import load_scene_geometries

    xml_path = Path(args.xml)
    scene_actor_id = args.scene_actor_id or f"{xml_path.stem}_quickconvert"
    actor_prefix = args.actor_prefix or f"{xml_path.stem}_qc"
    include_planes = not args.no_planes
    clean_existing = not args.no_clean_existing
    clean_old_articulation_import = not args.keep_old_articulation_import
    apply_mjcf_materials = not args.no_materials

    geoms = load_scene_geometries(
        xml_path,
        actor_prefix=xml_path.stem,
        include_visual_geoms=args.include_visual_geoms,
        include_planes=include_planes,
        plane_half_size=args.plane_half_size,
    )
    if not geoms:
        raise RuntimeError(f"no supported scene geometry found in {xml_path}")
    scene = build_quickconvert_scene(
        geoms,
        scene_actor_id=scene_actor_id,
        actor_prefix=actor_prefix,
    )

    if not args.skip_ue_spawn:
        command = build_unreal_import_exec(
            ue_script=str(Path(__file__).resolve()),
            urlab_bridge_root=str(bridge_root),
            xml_path=str(xml_path),
            scene_actor_id=scene_actor_id,
            actor_prefix=actor_prefix,
            include_visual_geoms=args.include_visual_geoms,
            include_planes=include_planes,
            plane_half_size=args.plane_half_size,
            clean_existing=clean_existing,
            clean_old_articulation_import=clean_old_articulation_import,
            save_level=False,
            apply_mjcf_materials=apply_mjcf_materials,
        )
        result = _run_unreal_remote_command(
            command,
            plugin_path=args.engine_python_plugin_path,
            timeout_s=args.remote_timeout,
            node_id=args.remote_node,
            multicast_group_endpoint=args.remote_group_endpoint,
            multicast_bind_address=args.remote_bind_address,
            command_endpoint=args.remote_command_endpoint,
        )
        print(f"UE remote import: success={result.get('success')} result={result.get('result', '')}")

    with URLabClient(
        args.address,
        step_port=args.step_port,
        local_model=False,
        mujoco_version_check=False,
    ) as client:
        client.connect(observations="minimal")
        applied = apply_quickconvert_with_urlab_client(
            client,
            scene,
            static=True,
            complex_mesh=args.complex_mesh,
            coacd_threshold=args.coacd_threshold,
            driven_by_unreal=args.driven_by_unreal,
        )
        if args.save_level:
            client.scene.save_level()
    visual_actors = sum(1 for actor in scene.actors if actor.visual_only)
    physics_actors = len(scene.actors) - visual_actors
    print(
        "URLab MJCF QuickConvert scene import via URLab client: "
        f"scene_id={scene.scene_actor_id} quickconvert_actors={applied}/{physics_actors} "
        f"visual_actors={visual_actors} spawned_actors={len(scene.actors)} "
        "friction=from_mjcf"
    )


def main() -> None:
    try:
        import unreal  # noqa: F401
    except ImportError:
        _run_external()
    else:
        _run_inside_unreal()


main()
