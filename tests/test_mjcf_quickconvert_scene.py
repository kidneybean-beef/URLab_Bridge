from __future__ import annotations

from pathlib import Path

from urlab_tools.mjcf_quickconvert_scene import build_quickconvert_scene
from urlab_tools.mjcf_scene_geoms import SceneBoxGeom, SceneMaterial, SceneTexture


def test_build_quickconvert_scene_maps_boxes_to_prop_actors():
    geoms = [
        SceneBoxGeom(
            actor_id="stairs_geom_0001",
            name="geom_0001",
            location=(1.3, 0.0, 0.025),
            rotation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            half_size=(0.15, 0.75, 0.025),
            friction=(1.0, 0.01, 0.001),
            source_index=1,
        ),
        SceneBoxGeom(
            actor_id="stairs_geom_0002",
            name="geom_0002",
            location=(1.6, 0.0, 0.075),
            rotation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            half_size=(0.15, 0.75, 0.025),
            friction=(0.8, 0.02, 0.002),
            source_index=2,
        ),
    ]

    scene = build_quickconvert_scene(
        geoms,
        scene_actor_id="stairs_quickconvert",
        actor_prefix="stairs_qc",
    )

    assert scene.scene_actor_id == "stairs_quickconvert"
    assert [spec.actor_id for spec in scene.actors] == [
        "stairs_qc_0001_stairs_geom_0001",
        "stairs_qc_0002_stairs_geom_0002",
    ]
    assert scene.actors[0].scale_xyz == (0.3, 1.5, 0.05)
    assert scene.actors[1].location == (1.6, 0.0, 0.075)
    assert scene.actors[1].friction == (0.8, 0.02, 0.002)


def test_build_quickconvert_scene_preserves_visual_material_metadata():
    texture = SceneTexture(
        name="label_tex",
        texture_type="2d",
        file=Path("/scene/imgs/label.png"),
    )
    material = SceneMaterial(
        name="label_mat",
        texture=texture,
        rgba=(1.0, 1.0, 1.0, 1.0),
    )
    geoms = [
        SceneBoxGeom(
            actor_id="cross_label",
            name="label",
            location=(0.0, 0.0, 0.1),
            rotation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            half_size=(1.0, 1.0, 0.001),
            friction=(1.0, 1.0, 1.0),
            source_index=1,
            material=material,
            rgba=(0.5, 0.6, 0.7, 1.0),
        )
    ]

    scene = build_quickconvert_scene(
        geoms,
        scene_actor_id="cross_quickconvert",
        actor_prefix="cross_qc",
    )

    assert scene.actors[0].material == material
    assert scene.actors[0].rgba == (0.5, 0.6, 0.7, 1.0)
    assert scene.actors[0].visual_only is False


def test_build_quickconvert_scene_preserves_visual_only_flag():
    geoms = [
        SceneBoxGeom(
            actor_id="cross_label",
            name="label",
            location=(0.0, 0.0, 0.1),
            rotation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            half_size=(1.0, 1.0, 0.001),
            friction=(1.0, 1.0, 1.0),
            source_index=1,
            visual_only=True,
        )
    ]

    scene = build_quickconvert_scene(
        geoms,
        scene_actor_id="cross_quickconvert",
        actor_prefix="cross_qc",
    )

    assert scene.actors[0].visual_only is True


def test_ue_import_script_does_not_have_direct_quickconvert_component_path():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "ue_import_mjcf_scene_quickconvert.py"

    script = script_path.read_text()

    assert "MjQuickConvertComponent" not in script
    assert "_ensure_quick_convert" not in script
    assert "APPLY_QUICK_CONVERT_IN_UE" not in script
