from __future__ import annotations

from urlab_tools.mjcf_quickconvert_scene import QuickConvertBoxActorSpec, QuickConvertSceneSpec
from urlab_tools.mjcf_quickconvert_workflow import (
    apply_quickconvert_with_urlab_client,
    build_unreal_import_exec,
    quickconvert_payloads,
    remote_discovery_error_message,
)


def _scene() -> QuickConvertSceneSpec:
    return QuickConvertSceneSpec(
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


def test_build_unreal_import_exec_only_configures_ue_actor_spawn():
    script = build_unreal_import_exec(
        ue_script="/sata/axin/URLab_Bridge/scripts/ue_import_mjcf_scene_quickconvert.py",
        urlab_bridge_root="/sata/axin/URLab_Bridge",
        xml_path="/tmp/cross_stairs.xml",
        scene_actor_id="cross_stairs_quickconvert",
        actor_prefix="cross_stairs_qc",
        include_visual_geoms=False,
        include_planes=True,
        plane_half_size=(10.0, 10.0, 0.025),
        clean_existing=True,
        clean_old_articulation_import=True,
        save_level=False,
        apply_mjcf_materials=True,
    )

    assert "APPLY_QUICK_CONVERT_IN_UE" not in script
    assert "COACD_THRESHOLD" not in script
    assert "COMPLEX_MESH" not in script
    assert "DRIVEN_BY_UNREAL" not in script
    assert "SAVE_LEVEL = False" in script
    assert "exec(open(" in script


def test_quickconvert_payloads_use_documented_urlab_outliner_fields():
    payloads = quickconvert_payloads(
        _scene(),
        static=True,
        complex_mesh=False,
        coacd_threshold=0.05,
        driven_by_unreal=False,
    )

    assert payloads == [
        {
            "target": "cross_stairs_qc_0001_floor",
            "by_name": False,
            "static": True,
            "complex_mesh": False,
            "coacd_threshold": 0.05,
            "driven_by_unreal": False,
            "friction": (0.8, 0.01, 0.001),
        }
    ]


def test_quickconvert_payloads_skip_visual_only_actors():
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
            QuickConvertBoxActorSpec(
                actor_id="cross_stairs_qc_0002_label",
                source_actor_id="label",
                location=(0.0, 0.0, 0.1),
                rotation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                scale_xyz=(1.0, 1.0, 0.001),
                friction=(1.0, 1.0, 1.0),
                visual_only=True,
            ),
        ),
    )

    payloads = quickconvert_payloads(
        scene,
        static=True,
        complex_mesh=False,
        coacd_threshold=0.05,
        driven_by_unreal=False,
    )

    assert [payload["target"] for payload in payloads] == [
        "cross_stairs_qc_0001_floor"
    ]


def test_apply_quickconvert_uses_single_batch_call():
    class FakeBatchResult:
        converted = 1
        failed = 0
        results = []

    class FakeOutliner:
        def __init__(self):
            self.batch_calls = []
            self.single_calls = []

        def add_quick_convert_many(self, items):
            self.batch_calls.append(list(items))
            return FakeBatchResult()

        def add_quick_convert(self, *args, **kwargs):
            self.single_calls.append((args, kwargs))

    class FakeClient:
        def __init__(self):
            self.outliner = FakeOutliner()

    client = FakeClient()
    applied = apply_quickconvert_with_urlab_client(
        client,
        _scene(),
        static=True,
        complex_mesh=False,
        coacd_threshold=0.05,
        driven_by_unreal=False,
    )

    assert applied == 1
    assert len(client.outliner.batch_calls) == 1
    assert client.outliner.batch_calls[0][0]["target"] == "cross_stairs_qc_0001_floor"
    assert client.outliner.single_calls == []


def test_apply_quickconvert_raises_on_batch_failures():
    class FakeBatchItem:
        target = "missing_actor"
        ok = False
        error = "no actor matching 'missing_actor'"

    class FakeBatchResult:
        converted = 0
        failed = 1
        results = [FakeBatchItem()]

    class FakeOutliner:
        def add_quick_convert_many(self, items):
            return FakeBatchResult()

    class FakeClient:
        outliner = FakeOutliner()

    try:
        apply_quickconvert_with_urlab_client(
            FakeClient(),
            _scene(),
            static=True,
            complex_mesh=False,
            coacd_threshold=0.05,
            driven_by_unreal=False,
        )
    except RuntimeError as exc:
        assert "missing_actor" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for failed batch item")


def test_remote_discovery_error_message_has_manual_fallback():
    message = remote_discovery_error_message(
        timeout_s=10.0,
        multicast_group_endpoint="239.0.0.1:6766",
        multicast_bind_address="127.0.0.1",
    )

    assert "Enable Remote Execution" in message
    assert "--skip-ue-spawn" in message
