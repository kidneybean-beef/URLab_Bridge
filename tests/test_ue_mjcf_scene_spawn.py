from __future__ import annotations

from types import SimpleNamespace

from urlab_tools.mjcf_quickconvert_scene import (
    QuickConvertBoxActorSpec,
    QuickConvertSceneSpec,
)
from urlab_tools.ue_mjcf_scene_spawn import MaterialImportConfig, spawn_scene_in_unreal


class _FakeQuat:
    def __init__(self, *_values: float) -> None:
        pass

    def rotator(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)


class _FakeActorClass:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


class _FakeStaticMeshComponent:
    def __init__(self) -> None:
        self.mesh = None
        self.mobility = None

    def set_static_mesh(self, mesh: object) -> None:
        self.mesh = mesh

    def set_mobility(self, mobility: object) -> None:
        self.mobility = mobility

    def set_material(self, _index: int, _material: object) -> None:
        pass


class _FakeActor:
    def __init__(self, class_name: str, location: object, rotation: object) -> None:
        self.class_name = class_name
        self.label = class_name
        self.tags = []
        self.folder_path = None
        self.location = location
        self.rotation = rotation
        self.scale = None
        self.static_mesh_component = _FakeStaticMeshComponent()

    def get_actor_label(self) -> str:
        return self.label

    def set_actor_label(self, label: str) -> None:
        self.label = label

    def set_folder_path(self, folder: object) -> None:
        self.folder_path = str(folder)

    def get_folder_path(self) -> object | None:
        return self.folder_path

    def set_actor_location(self, location: object, *_args: object) -> None:
        self.location = location

    def set_actor_rotation(self, rotation: object, *_args: object) -> None:
        self.rotation = rotation

    def set_actor_scale3d(self, scale: object) -> None:
        self.scale = scale

    def get_component_by_class(self, _component_class: object) -> _FakeStaticMeshComponent:
        return self.static_mesh_component

    def get_class(self) -> _FakeActorClass:
        return _FakeActorClass(self.class_name)

    def attach_to_actor(self, *_args: object) -> None:
        raise AssertionError("scene props should not attach to a marker actor")


class _FakeActorSubsystem:
    def __init__(self, unreal: "_FakeUnreal") -> None:
        self.unreal = unreal

    def get_all_level_actors(self) -> list[_FakeActor]:
        return list(self.unreal.actors)

    def spawn_actor_from_class(
        self,
        actor_class: object,
        location: object,
        rotation: object,
        _transient: bool = False,
    ) -> _FakeActor:
        class_name = "Actor" if actor_class is self.unreal.Actor else "StaticMeshActor"
        actor = _FakeActor(class_name, location, rotation)
        self.unreal.actors.append(actor)
        return actor

    def destroy_actor(self, actor: _FakeActor) -> bool:
        self.unreal.actors.remove(actor)
        return True


class _FakeUnreal:
    Actor = object()
    StaticMeshActor = object()
    StaticMeshComponent = object()
    AttachmentRule = SimpleNamespace(KEEP_WORLD="keep_world")
    ComponentMobility = SimpleNamespace(STATIC="static")
    EditorActorSubsystem = object()
    EditorLevelLibrary = SimpleNamespace(get_all_level_actors=lambda: [])
    Quat = _FakeQuat

    def __init__(self) -> None:
        self.actors: list[_FakeActor] = []

    def get_editor_subsystem(self, _subsystem_class: object) -> _FakeActorSubsystem:
        return _FakeActorSubsystem(self)

    def load_asset(self, path: str) -> object | None:
        if path == "/Engine/BasicShapes/Cube.Cube":
            return object()
        return None

    def Name(self, value: str) -> str:
        return value

    def Vector(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        return (x, y, z)

    def Rotator(self, pitch: float, yaw: float, roll: float) -> tuple[float, float, float]:
        return (pitch, yaw, roll)


def test_spawn_scene_does_not_create_folder_or_marker_actor():
    scene = QuickConvertSceneSpec(
        scene_actor_id="cross_stairs_quickconvert",
        actors=(
            QuickConvertBoxActorSpec(
                actor_id="cross_stairs_qc_0001_floor",
                source_actor_id="floor",
                location=(0.0, 0.0, 0.0),
                rotation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                scale_xyz=(1.0, 1.0, 0.05),
                friction=(0.8, 0.01, 0.001),
            ),
        ),
    )

    unreal = _FakeUnreal()

    result = spawn_scene_in_unreal(
        unreal,
        scene,
        xml_stem="cross_stairs",
        clean_existing=True,
        clean_old_articulation_import=True,
        save_level=False,
        material_config=MaterialImportConfig(
            apply_mjcf_materials=False,
            material_dest_path="/Game/URLab/MJCFMaterials",
            texture_dest_path="/Game/URLab/MJCFTextures",
            master_material_path="/Game/URLab/M_Master",
            checker_material_path="/Game/URLab/M_Checker",
            default_material_path="/Engine/BasicShapes/BasicShapeMaterial",
        ),
    )

    assert result.spawned_actors == 1
    assert result.total_actors == 1
    assert getattr(result, "folder_path", None) == ""
    assert [actor.class_name for actor in unreal.actors] == ["StaticMeshActor"]
    assert [actor.folder_path for actor in unreal.actors] == [None]


def test_cleanup_removes_stale_marker_actor_from_scene_folder():
    scene = QuickConvertSceneSpec(
        scene_actor_id="cross_stairs_quickconvert",
        actors=(
            QuickConvertBoxActorSpec(
                actor_id="cross_stairs_qc_0001_floor",
                source_actor_id="floor",
                location=(0.0, 0.0, 0.0),
                rotation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                scale_xyz=(1.0, 1.0, 0.05),
                friction=(0.8, 0.01, 0.001),
            ),
        ),
    )
    unreal = _FakeUnreal()
    stale_marker = _FakeActor(
        "Actor",
        location=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
    )
    stale_marker.set_actor_label("stale_marker_label")
    stale_marker.set_folder_path("cross_stairs_quickconvert")
    unreal.actors.append(stale_marker)

    result = spawn_scene_in_unreal(
        unreal,
        scene,
        xml_stem="cross_stairs",
        clean_existing=True,
        clean_old_articulation_import=True,
        save_level=False,
        material_config=MaterialImportConfig(
            apply_mjcf_materials=False,
            material_dest_path="/Game/URLab/MJCFMaterials",
            texture_dest_path="/Game/URLab/MJCFTextures",
            master_material_path="/Game/URLab/M_Master",
            checker_material_path="/Game/URLab/M_Checker",
            default_material_path="/Engine/BasicShapes/BasicShapeMaterial",
        ),
    )

    assert result.removed_old_actors == 1
    assert [actor.class_name for actor in unreal.actors] == ["StaticMeshActor"]
    assert [actor.get_actor_label() for actor in unreal.actors] == [
        "cross_stairs_qc_0001_floor"
    ]
