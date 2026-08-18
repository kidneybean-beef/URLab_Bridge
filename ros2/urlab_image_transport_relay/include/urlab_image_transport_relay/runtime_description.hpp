#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <msgpack.hpp>

namespace urlab_image_transport_relay
{

enum class CameraMode
{
  Real,
  Depth,
  Semantic,
  Instance,
};

enum class RealPayloadEncoding
{
  Srgb,
  Linear,
};

struct CameraSpec
{
  std::string articulation;
  std::string camera;
  CameraMode mode = CameraMode::Real;
  std::string mode_name = "real";
  int width = 0;
  int height = 0;
  double fovy = 0.0;
  double depth_near_cm = 10.0;
  double depth_far_cm = 10000.0;
  bool enabled = false;
  bool streaming = false;
  bool zmq_broadcast_enabled = false;
  bool zmq_broadcast_active = false;
  std::string payload_encoding;
  std::string ros_encoding;
  std::string zmq_endpoint;
  std::string zmq_topic;
  // Meaningful only for Real cameras. The direct bridge accepts the current
  // display-ready contract and the legacy linear contract explicitly.
  RealPayloadEncoding real_payload_encoding = RealPayloadEncoding::Srgb;
};

struct PinholeCalibration
{
  std::array<double, 5> d{};
  std::array<double, 9> k{};
  std::array<double, 9> r{};
  std::array<double, 12> p{};
};

std::string camera_mode_to_string(CameraMode mode);

/** Packs the only valid sessionless discovery request: {"op":"describe_runtime"}. */
msgpack::sbuffer make_describe_runtime_request();

/** Parses URLab's read-only runtime description and validates its camera
 *  payload contract before a stream worker is started. */
std::vector<CameraSpec> parse_describe_runtime_reply(const msgpack::object & reply);

/** Converts URLab's native BGRA Real-camera wire payload into ROS rgb8.
 *  bgra8_srgb needs only channel reordering; bgra8_linear takes the retained
 *  legacy LUT path before that reorder. */
std::vector<std::uint8_t> convert_real_bgra_to_rgb(
  std::span<const std::uint8_t> pixels,
  const CameraSpec & spec);

/** Builds the centered, square-pixel ROS pinhole model from MuJoCo's vertical
 *  field of view. Invalid dimensions or perspective FOV values are rejected. */
PinholeCalibration make_centered_pinhole_calibration(
  int width,
  int height,
  double vertical_fov_degrees);

/** The direct sidecar is read-only. It must not acquire a controller session
 *  merely to toggle capture. */
void validate_direct_bridge_options(bool auto_enable_cameras);

/** Fails early for a camera that cannot publish ZMQ frames. An inactive
 *  publisher is allowed: the camera can be enabled later by its owner. */
void validate_camera_zmq_configuration(const CameraSpec & spec);

}  // namespace urlab_image_transport_relay
