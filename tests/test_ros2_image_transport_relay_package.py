from __future__ import annotations

from pathlib import Path


RELAY_ROOT = Path(__file__).parents[1] / "ros2" / "urlab_image_transport_relay"


def test_relay_package_declares_standard_image_transport_dependencies() -> None:
    package_xml = (RELAY_ROOT / "package.xml").read_text()

    assert "<name>urlab_image_transport_relay</name>" in package_xml
    assert "<depend>rclcpp</depend>" in package_xml
    assert "<depend>rmw</depend>" in package_xml
    assert "<depend>sensor_msgs</depend>" in package_xml
    assert "<depend>image_transport</depend>" in package_xml
    assert "<exec_depend>image_transport_plugins</exec_depend>" in package_xml


def test_relay_cmake_installs_image_transport_relay_executable() -> None:
    cmake = (RELAY_ROOT / "CMakeLists.txt").read_text()

    assert "add_executable(image_transport_relay" in cmake
    assert "ament_target_dependencies(image_transport_relay" in cmake
    assert "rmw" in cmake
    assert "image_transport" in cmake
    assert "install(TARGETS image_transport_relay" in cmake


def test_relay_package_declares_direct_urlab_camera_bridge_dependencies() -> None:
    package_xml = (RELAY_ROOT / "package.xml").read_text()

    assert "<depend>msgpack-cxx</depend>" in package_xml
    assert "<depend>pkg-config</depend>" in package_xml
    assert "<depend>libzmq3-dev</depend>" in package_xml


def test_relay_cmake_installs_direct_urlab_camera_bridge_executable() -> None:
    cmake = (RELAY_ROOT / "CMakeLists.txt").read_text()

    assert "find_package(PkgConfig REQUIRED)" in cmake
    assert "pkg_check_modules(ZMQ REQUIRED libzmq)" in cmake
    assert "add_executable(urlab_camera_bridge" in cmake
    assert "ament_target_dependencies(urlab_camera_bridge" in cmake
    assert "install(TARGETS image_transport_relay urlab_camera_bridge" in cmake
    assert "add_library(urlab_camera_runtime" in cmake
    assert "ament_add_gtest(test_runtime_description" in cmake


def test_direct_urlab_camera_bridge_source_uses_urlab_zmq_and_image_transport() -> None:
    source = (RELAY_ROOT / "src" / "urlab_camera_bridge.cpp").read_text()
    runtime_source = (RELAY_ROOT / "src" / "runtime_description.cpp").read_text()

    assert 'declare_parameter<std::vector<std::string>>("cameras"' in source
    assert 'declare_parameter<std::string>("urlab_address"' in source
    assert 'declare_parameter<int>("step_port"' in source
    assert "zmq_connect" in source
    assert "zmq_setsockopt" in source
    assert "image_transport::create_publisher" in source
    assert "convert_real_bgra_to_rgb" in source
    assert 'msg.encoding = "rgb8"' in source
    assert 'msg.encoding = "32FC1"' in source
    assert 'msg.encoding = "bgra8"' in source
    assert "client.describe_runtime()" in source
    assert "set_camera_enabled" not in source
    assert '"describe_runtime"' in runtime_source
    assert '"bgra8_srgb"' in runtime_source
    assert '"bgra8_linear"' in runtime_source
    assert '"hello"' not in source


def test_direct_bridge_runtime_discovery_is_session_safe() -> None:
    runtime_header = (
        RELAY_ROOT / "include" / "urlab_image_transport_relay" / "runtime_description.hpp"
    ).read_text()
    runtime_source = (RELAY_ROOT / "src" / "runtime_description.cpp").read_text()

    assert "make_describe_runtime_request" in runtime_header
    assert "packer.pack_map(1)" in runtime_source
    assert '"describe_runtime"' in runtime_source
    assert "validate_direct_bridge_options" in runtime_source
    assert "Broadcast to ZMQ disabled" in runtime_source


def test_relay_source_uses_distinct_input_and_output_topics() -> None:
    source = (RELAY_ROOT / "src" / "image_transport_relay.cpp").read_text()

    assert 'declare_parameter<std::string>("input_topic"' in source
    assert 'declare_parameter<std::string>("output_topic"' in source
    assert "image_transport::create_publisher" in source
    assert "create_subscription<sensor_msgs::msg::Image>" in source
    assert "input_topic_ == output_topic_" in source
