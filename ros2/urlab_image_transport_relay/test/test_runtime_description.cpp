#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <msgpack.hpp>

#include "urlab_image_transport_relay/runtime_description.hpp"

namespace
{

template<typename PackerT>
void pack_string(PackerT & packer, const std::string & value)
{
  packer.pack_str(static_cast<std::uint32_t>(value.size()));
  packer.pack_str_body(value.data(), static_cast<std::uint32_t>(value.size()));
}

std::string payload_for(const std::string & mode)
{
  if (mode == "depth") {
    return "float32_cm";
  }
  return mode == "real" ? "bgra8_srgb" : "bgra8";
}

std::string ros_for(const std::string & mode)
{
  if (mode == "depth") {
    return "32FC1";
  }
  return mode == "real" ? "rgb8" : "bgra8";
}

void pack_camera(msgpack::packer<msgpack::sbuffer> & packer,
  const std::string & name, const std::string & mode, bool zmq_enabled = true,
  const std::string & real_payload = "")
{
  pack_string(packer, name);
  packer.pack_map(13);
  pack_string(packer, "mode");
  pack_string(packer, mode);
  pack_string(packer, "resolution");
  packer.pack_array(2);
  // URLab serializes every FJsonValueNumber as MsgPack float64, including
  // integer-valued camera dimensions.
  packer.pack(640.0);
  packer.pack(480.0);
  pack_string(packer, "fovy");
  packer.pack(90.0);
  pack_string(packer, "depth_near_cm");
  packer.pack(10.0);
  pack_string(packer, "depth_far_cm");
  packer.pack(10000.0);
  pack_string(packer, "enabled");
  packer.pack(true);
  pack_string(packer, "streaming");
  packer.pack(true);
  pack_string(packer, "zmq_broadcast_enabled");
  packer.pack(zmq_enabled);
  pack_string(packer, "zmq_broadcast_active");
  packer.pack(true);
  pack_string(packer, "payload_encoding");
  pack_string(packer, real_payload.empty() ? payload_for(mode) : real_payload);
  pack_string(packer, "ros_encoding");
  pack_string(packer, ros_for(mode));
  pack_string(packer, "zmq_endpoint");
  pack_string(packer, "tcp://127.0.0.1:5558");
  pack_string(packer, "zmq_topic");
  pack_string(packer, "go2/camera/" + name);
}

msgpack::object_handle make_runtime_reply(
  const std::vector<std::string> & modes,
  const std::string & real_payload = "")
{
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(3);
  pack_string(packer, "op");
  pack_string(packer, "describe_runtime_ok");
  pack_string(packer, "manager_present");
  packer.pack(true);
  pack_string(packer, "articulations");
  packer.pack_array(1);
  packer.pack_map(2);
  pack_string(packer, "prefix");
  pack_string(packer, "go2");
  pack_string(packer, "camera_topics");
  packer.pack_map(static_cast<std::uint32_t>(modes.size()));
  for (const std::string & mode : modes) {
    pack_camera(packer, "front_" + mode, mode, true, real_payload);
  }
  return msgpack::unpack(buffer.data(), buffer.size());
}

}  // namespace

TEST(RuntimeDescription, PacksOnlyDescribeRuntimeOp)
{
  const msgpack::sbuffer request =
    urlab_image_transport_relay::make_describe_runtime_request();
  const msgpack::object_handle unpacked = msgpack::unpack(request.data(), request.size());
  const msgpack::object root = unpacked.get();

  ASSERT_EQ(root.type, msgpack::type::MAP);
  ASSERT_EQ(root.via.map.size, 1U);
  EXPECT_EQ(root.via.map.ptr[0].key.as<std::string>(), "op");
  EXPECT_EQ(root.via.map.ptr[0].val.as<std::string>(), "describe_runtime");
}

TEST(RuntimeDescription, ParsesAllSupportedCameraModes)
{
  const msgpack::object_handle reply =
    make_runtime_reply({"real", "depth", "semantic", "instance"});
  const std::vector<urlab_image_transport_relay::CameraSpec> cameras =
    urlab_image_transport_relay::parse_describe_runtime_reply(reply.get());

  ASSERT_EQ(cameras.size(), 4U);
  EXPECT_EQ(cameras[0].width, 640);
  EXPECT_EQ(cameras[0].height, 480);
  EXPECT_DOUBLE_EQ(cameras[0].fovy, 90.0);
  EXPECT_EQ(cameras[0].mode, urlab_image_transport_relay::CameraMode::Real);
  EXPECT_EQ(cameras[0].payload_encoding, "bgra8_srgb");
  EXPECT_EQ(
    cameras[0].real_payload_encoding,
    urlab_image_transport_relay::RealPayloadEncoding::Srgb);
  EXPECT_EQ(cameras[0].ros_encoding, "rgb8");
  EXPECT_EQ(cameras[1].mode, urlab_image_transport_relay::CameraMode::Depth);
  EXPECT_EQ(cameras[1].ros_encoding, "32FC1");
  EXPECT_EQ(cameras[2].mode, urlab_image_transport_relay::CameraMode::Semantic);
  EXPECT_EQ(cameras[2].ros_encoding, "bgra8");
  EXPECT_EQ(cameras[3].mode, urlab_image_transport_relay::CameraMode::Instance);
  EXPECT_EQ(cameras[3].ros_encoding, "bgra8");
}

TEST(RuntimeDescription, BuildsCenteredPinholeCalibrationFromVerticalFov)
{
  const auto calibration =
    urlab_image_transport_relay::make_centered_pinhole_calibration(640, 480, 90.0);

  EXPECT_EQ(calibration.d, (std::array<double, 5>{0.0, 0.0, 0.0, 0.0, 0.0}));
  EXPECT_DOUBLE_EQ(calibration.k[0], 240.0);
  EXPECT_DOUBLE_EQ(calibration.k[2], 319.5);
  EXPECT_DOUBLE_EQ(calibration.k[4], 240.0);
  EXPECT_DOUBLE_EQ(calibration.k[5], 239.5);
  EXPECT_EQ(
    calibration.r,
    (std::array<double, 9>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
  EXPECT_DOUBLE_EQ(calibration.p[0], 240.0);
  EXPECT_DOUBLE_EQ(calibration.p[2], 319.5);
  EXPECT_DOUBLE_EQ(calibration.p[5], 240.0);
  EXPECT_DOUBLE_EQ(calibration.p[6], 239.5);
  EXPECT_DOUBLE_EQ(calibration.p[10], 1.0);
}

TEST(RuntimeDescription, RejectsInvalidPinholeCalibration)
{
  EXPECT_THROW(
    urlab_image_transport_relay::make_centered_pinhole_calibration(0, 480, 90.0),
    std::invalid_argument);
  EXPECT_THROW(
    urlab_image_transport_relay::make_centered_pinhole_calibration(640, 0, 90.0),
    std::invalid_argument);
  EXPECT_THROW(
    urlab_image_transport_relay::make_centered_pinhole_calibration(640, 480, 0.0),
    std::invalid_argument);
  EXPECT_THROW(
    urlab_image_transport_relay::make_centered_pinhole_calibration(640, 480, 180.0),
    std::invalid_argument);
}

TEST(RuntimeDescription, SupportsLegacyLinearRealPayloadAndConvertsBothContracts)
{
  const msgpack::object_handle legacy_reply =
    make_runtime_reply({"real"}, "bgra8_linear");
  const std::vector<urlab_image_transport_relay::CameraSpec> legacy_cameras =
    urlab_image_transport_relay::parse_describe_runtime_reply(legacy_reply.get());
  ASSERT_EQ(legacy_cameras.size(), 1U);
  EXPECT_EQ(
    legacy_cameras[0].real_payload_encoding,
    urlab_image_transport_relay::RealPayloadEncoding::Linear);

  const std::vector<std::uint8_t> bgra{0, 64, 128, 255};
  auto srgb_spec = legacy_cameras[0];
  srgb_spec.width = 1;
  srgb_spec.height = 1;
  srgb_spec.real_payload_encoding = urlab_image_transport_relay::RealPayloadEncoding::Srgb;
  auto legacy_spec = srgb_spec;
  legacy_spec.real_payload_encoding = urlab_image_transport_relay::RealPayloadEncoding::Linear;
  EXPECT_EQ(
    urlab_image_transport_relay::convert_real_bgra_to_rgb(bgra, srgb_spec),
    (std::vector<std::uint8_t>{128, 64, 0}));
  EXPECT_EQ(
    urlab_image_transport_relay::convert_real_bgra_to_rgb(bgra, legacy_spec),
    (std::vector<std::uint8_t>{188, 137, 0}));
}

TEST(RuntimeDescription, RejectsErrorAndMalformedReplies)
{
  msgpack::sbuffer error_buffer;
  msgpack::pack(error_buffer, std::map<std::string, std::string>{
    {"op", "error"}, {"code", "no_active_manager"}, {"message", "PIE not running"}});
  const msgpack::object_handle error_reply =
    msgpack::unpack(error_buffer.data(), error_buffer.size());
  EXPECT_THROW(
    urlab_image_transport_relay::parse_describe_runtime_reply(error_reply.get()),
    std::runtime_error);

  msgpack::sbuffer malformed_buffer;
  msgpack::pack(malformed_buffer, std::map<std::string, std::string>{{"op", "hello_ok"}});
  const msgpack::object_handle malformed_reply =
    msgpack::unpack(malformed_buffer.data(), malformed_buffer.size());
  EXPECT_THROW(
    urlab_image_transport_relay::parse_describe_runtime_reply(malformed_reply.get()),
    std::runtime_error);

  const msgpack::object_handle unknown_real_payload =
    make_runtime_reply({"real"}, "rgba16f");
  EXPECT_THROW(
    urlab_image_transport_relay::parse_describe_runtime_reply(unknown_real_payload.get()),
    std::runtime_error);
}

TEST(RuntimeDescription, RejectsUnsafeOrUnavailableCameraControls)
{
  EXPECT_THROW(
    urlab_image_transport_relay::validate_direct_bridge_options(true),
    std::invalid_argument);

  urlab_image_transport_relay::CameraSpec spec;
  spec.articulation = "go2";
  spec.camera = "front_rgb";
  spec.zmq_broadcast_enabled = false;
  EXPECT_THROW(
    urlab_image_transport_relay::validate_camera_zmq_configuration(spec),
    std::runtime_error);
}
