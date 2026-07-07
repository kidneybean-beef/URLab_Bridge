from __future__ import annotations

from pathlib import Path

from urlab_tools.mjcf_scene_geoms import (
    SceneBoxGeom,
    load_scene_geometries,
)


def test_load_scene_geometries_extracts_boxes_and_ignores_robot_include(
    tmp_path: Path,
):
    xml = tmp_path / "scene.xml"
    xml.write_text(
        """
<mujoco model="test scene">
  <include file="go2.xml"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <geom name="step" type="box" pos="1 2 0.25" size="0.5 0.25 0.1"
          quat="0.70710678 0 0 0.70710678" friction="0.8 0.01 0.001"/>
  </worldbody>
</mujoco>
""".strip()
    )

    geoms = load_scene_geometries(xml, actor_prefix="track")

    assert [g.actor_id for g in geoms] == ["track_step"]
    assert geoms[0].location == (1.0, 2.0, 0.25)
    assert geoms[0].half_size == (0.5, 0.25, 0.1)
    assert geoms[0].rotation_quat_wxyz == (0.70710678, 0.0, 0.0, 0.70710678)
    assert geoms[0].friction == (0.8, 0.01, 0.001)


def test_load_scene_geometries_skips_visual_only_boxes_by_default(
    tmp_path: Path,
):
    xml = tmp_path / "labels.xml"
    xml.write_text(
        """
<mujoco>
  <worldbody>
    <geom type="box" pos="0 0 0.1" size="1 1 0.1"/>
    <geom type="box" pos="0 0 0.2" size="1 1 0.0001"
          contype="0" conaffinity="0" group="1"/>
  </worldbody>
</mujoco>
""".strip()
    )

    default_geoms = load_scene_geometries(xml, actor_prefix="scene")
    all_geoms = load_scene_geometries(
        xml, actor_prefix="scene", include_visual_geoms=True
    )

    assert [g.actor_id for g in default_geoms] == ["scene_geom_0000"]
    assert [g.actor_id for g in all_geoms] == [
        "scene_geom_0000",
        "scene_geom_0001",
    ]


def test_load_scene_geometries_can_turn_plane_into_floor_box(tmp_path: Path):
    xml = tmp_path / "flat.xml"
    xml.write_text(
        """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0"/>
  </worldbody>
</mujoco>
""".strip()
    )

    geoms = load_scene_geometries(
        xml,
        actor_prefix="flat",
        include_planes=True,
        plane_half_size=(5.0, 4.0, 0.025),
    )

    assert [g.actor_id for g in geoms] == ["flat_floor"]
    assert geoms[0].location == (0.0, 0.0, -0.025)
    assert geoms[0].half_size == (5.0, 4.0, 0.025)


def test_load_scene_geometries_resolves_mjcf_material_assets(tmp_path: Path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    (img_dir / "label_1.png").write_bytes(b"fake image")
    xml = tmp_path / "scene.xml"
    xml.write_text(
        """
<mujoco>
  <asset>
    <texture name="ground_tex" type="2d" builtin="checker"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="ground_tex" texrepeat="5 5"/>
    <texture name="label_tex" type="2d" file="./imgs/label_1.png"/>
    <material name="label_mat" texture="label_tex" rgba="1 1 1 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" material="groundplane"/>
    <geom name="label" type="box" pos="0 0 0.1" size="1 1 0.01"
          material="label_mat" rgba="0.5 0.6 0.7 1"/>
  </worldbody>
</mujoco>
""".strip()
    )

    geoms = load_scene_geometries(
        xml,
        actor_prefix="scene",
        include_planes=True,
        plane_half_size=(5.0, 5.0, 0.025),
    )

    assert geoms[0].material is not None
    assert geoms[0].material.name == "groundplane"
    assert geoms[0].material.texture is not None
    assert geoms[0].material.texture.builtin == "checker"
    assert geoms[0].material.texture.rgb1 == (0.2, 0.3, 0.4)
    assert geoms[0].material.texture.rgb2 == (0.1, 0.2, 0.3)
    assert geoms[0].material.texrepeat == (5.0, 5.0)
    assert geoms[1].material is not None
    assert geoms[1].material.name == "label_mat"
    assert geoms[1].material.texture is not None
    assert geoms[1].material.texture.file == img_dir / "label_1.png"
    assert geoms[1].rgba == (0.5, 0.6, 0.7, 1.0)


def test_scene_box_geom_payload_is_plain_python_data():
    geom = SceneBoxGeom(
        actor_id="stairs_step",
        name="step",
        location=(1.0, 0.0, 0.25),
        rotation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        half_size=(0.5, 0.25, 0.1),
        friction=(0.9, 0.01, 0.001),
        source_index=0,
        visual_only=False,
    )

    assert geom.as_unreal_payload() == {
        "actor_id": "stairs_step",
        "name": "step",
        "location": [1.0, 0.0, 0.25],
        "rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "half_size": [0.5, 0.25, 0.1],
        "friction": [0.9, 0.01, 0.001],
        "source_index": 0,
        "visual_only": False,
    }
