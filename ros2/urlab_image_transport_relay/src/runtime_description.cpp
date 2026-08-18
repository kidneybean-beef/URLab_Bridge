#include "urlab_image_transport_relay/runtime_description.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace urlab_image_transport_relay
{
namespace
{

const msgpack::object * map_get(const msgpack::object & object, const std::string & key)
{
  if (object.type != msgpack::type::MAP) {
    return nullptr;
  }
  for (std::uint32_t i = 0; i < object.via.map.size; ++i) {
    const msgpack::object_kv & kv = object.via.map.ptr[i];
    if (kv.key.type != msgpack::type::STR) {
      continue;
    }
    if (kv.key.via.str.size == key.size() &&
      std::memcmp(kv.key.via.str.ptr, key.data(), key.size()) == 0)
    {
      return &kv.val;
    }
  }
  return nullptr;
}

std::string object_string(const msgpack::object * object, const std::string & fallback = "")
{
  if (object == nullptr) {
    return fallback;
  }
  if (object->type == msgpack::type::STR) {
    return std::string(object->via.str.ptr, object->via.str.size);
  }
  if (object->type == msgpack::type::BIN) {
    return std::string(object->via.bin.ptr, object->via.bin.size);
  }
  return fallback;
}

bool object_bool(const msgpack::object * object, bool fallback = false)
{
  return object != nullptr && object->type == msgpack::type::BOOLEAN ?
    object->via.boolean : fallback;
}

double object_double(const msgpack::object * object, double fallback = 0.0)
{
  if (object == nullptr) {
    return fallback;
  }
  try {
    double value = fallback;
    object->convert(value);
    return value;
  } catch (const std::exception &) {
    return fallback;
  }
}

int object_int(const msgpack::object * object, int fallback = 0)
{
	const double value = object_double(
		object, std::numeric_limits<double>::quiet_NaN());
	if (!std::isfinite(value)
		|| std::trunc(value) != value
		|| value < static_cast<double>(std::numeric_limits<int>::min())
		|| value > static_cast<double>(std::numeric_limits<int>::max()))
	{
		return fallback;
	}
	return static_cast<int>(value);
}

std::pair<int, int> object_resolution(const msgpack::object * object)
{
  if (object == nullptr || object->type != msgpack::type::ARRAY || object->via.array.size < 2) {
    return {0, 0};
  }
  return {
    object_int(&object->via.array.ptr[0]),
    object_int(&object->via.array.ptr[1]),
  };
}

template<typename PackerT>
void pack_string(PackerT & packer, const std::string & value)
{
  packer.pack_str(static_cast<std::uint32_t>(value.size()));
  packer.pack_str_body(value.data(), static_cast<std::uint32_t>(value.size()));
}

CameraMode parse_camera_mode(const std::string & value)
{
  if (value == "real") {
    return CameraMode::Real;
  }
  if (value == "depth") {
    return CameraMode::Depth;
  }
  if (value == "semantic") {
    return CameraMode::Semantic;
  }
  if (value == "instance") {
    return CameraMode::Instance;
  }
  throw std::runtime_error("unsupported URLab camera mode '" + value + "'");
}

std::string expected_payload_encoding(CameraMode mode)
{
  switch (mode) {
    case CameraMode::Real:
      return "";
    case CameraMode::Depth:
      return "float32_cm";
    case CameraMode::Semantic:
    case CameraMode::Instance:
      return "bgra8";
  }
  return "";
}

std::string expected_ros_encoding(CameraMode mode)
{
  switch (mode) {
    case CameraMode::Real:
      return "rgb8";
    case CameraMode::Depth:
      return "32FC1";
    case CameraMode::Semantic:
    case CameraMode::Instance:
      return "bgra8";
  }
  return "";
}

void validate_camera_payload_contract(CameraSpec & spec)
{
  const std::string expected_payload = expected_payload_encoding(spec.mode);
  const std::string expected_ros = expected_ros_encoding(spec.mode);
  const bool real_payload_supported = spec.mode != CameraMode::Real ||
    spec.payload_encoding == "bgra8_srgb" || spec.payload_encoding == "bgra8_linear";
  if (!real_payload_supported ||
    (spec.mode != CameraMode::Real && spec.payload_encoding != expected_payload) ||
    spec.ros_encoding != expected_ros)
  {
    throw std::runtime_error(
      "camera " + spec.articulation + ":" + spec.camera +
      " advertises an unsupported payload contract '" + spec.payload_encoding +
      "' -> '" + spec.ros_encoding + "'");
  }
  if (spec.mode == CameraMode::Real) {
    spec.real_payload_encoding = spec.payload_encoding == "bgra8_linear"
      ? RealPayloadEncoding::Linear
      : RealPayloadEncoding::Srgb;
  }
}

const std::array<std::uint8_t, 256> & legacy_linear_to_srgb_lut()
{
  static const std::array<std::uint8_t, 256> lut = []() {
      std::array<std::uint8_t, 256> values{};
      for (std::size_t index = 0; index < values.size(); ++index) {
        const double linear = static_cast<double>(index) / 255.0;
        const double srgb = linear <= 0.0031308
          ? 12.92 * linear
          : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
        values[index] = static_cast<std::uint8_t>(
          std::clamp(std::lround(srgb * 255.0), 0L, 255L));
      }
      return values;
    }();
  return lut;
}

}  // namespace

std::string camera_mode_to_string(CameraMode mode)
{
  switch (mode) {
    case CameraMode::Real:
      return "real";
    case CameraMode::Depth:
      return "depth";
    case CameraMode::Semantic:
      return "semantic";
    case CameraMode::Instance:
      return "instance";
  }
  return "real";
}

msgpack::sbuffer make_describe_runtime_request()
{
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(1);
  pack_string(packer, "op");
  pack_string(packer, "describe_runtime");
  return buffer;
}

std::vector<CameraSpec> parse_describe_runtime_reply(const msgpack::object & reply)
{
  const std::string op = object_string(map_get(reply, "op"));
  if (op == "error") {
    throw std::runtime_error(
      "URLab describe_runtime failed: " +
      object_string(map_get(reply, "code"), "error") + ": " +
      object_string(map_get(reply, "message"), "unknown error"));
  }
  if (op != "describe_runtime_ok") {
    throw std::runtime_error("unexpected URLab describe_runtime reply op: " + op);
  }
  if (!object_bool(map_get(reply, "manager_present"))) {
    throw std::runtime_error(
      "URLab describe_runtime reports no active manager; start PIE and MuJoCo first");
  }

  const msgpack::object * articulations = map_get(reply, "articulations");
  if (articulations == nullptr || articulations->type != msgpack::type::ARRAY) {
    throw std::runtime_error("URLab describe_runtime reply has no articulations array");
  }

  std::vector<CameraSpec> cameras_out;
  for (std::uint32_t i = 0; i < articulations->via.array.size; ++i) {
    const msgpack::object & articulation = articulations->via.array.ptr[i];
    const std::string prefix = object_string(map_get(articulation, "prefix"));
    if (prefix.empty()) {
      throw std::runtime_error("URLab describe_runtime contains an articulation without prefix");
    }
    const msgpack::object * cameras = map_get(articulation, "camera_topics");
    if (cameras == nullptr || cameras->type != msgpack::type::MAP) {
      continue;
    }
    for (std::uint32_t camera_index = 0; camera_index < cameras->via.map.size; ++camera_index) {
      const msgpack::object_kv & camera_kv = cameras->via.map.ptr[camera_index];
      const std::string camera_name = object_string(&camera_kv.key);
      if (camera_name.empty() || camera_kv.val.type != msgpack::type::MAP) {
        throw std::runtime_error("URLab describe_runtime contains malformed camera metadata");
      }

      CameraSpec spec;
      spec.articulation = prefix;
      spec.camera = camera_name;
      spec.mode_name = object_string(map_get(camera_kv.val, "mode"));
      spec.mode = parse_camera_mode(spec.mode_name);
      const auto [width, height] = object_resolution(map_get(camera_kv.val, "resolution"));
      spec.width = width;
      spec.height = height;
      spec.fovy = object_double(map_get(camera_kv.val, "fovy"));
      spec.depth_near_cm = object_double(map_get(camera_kv.val, "depth_near_cm"), 10.0);
      spec.depth_far_cm = object_double(map_get(camera_kv.val, "depth_far_cm"), 10000.0);
      spec.enabled = object_bool(map_get(camera_kv.val, "enabled"));
      spec.streaming = object_bool(map_get(camera_kv.val, "streaming"));
      spec.zmq_broadcast_enabled = object_bool(
        map_get(camera_kv.val, "zmq_broadcast_enabled"));
      spec.zmq_broadcast_active = object_bool(
        map_get(camera_kv.val, "zmq_broadcast_active"));
      spec.payload_encoding = object_string(map_get(camera_kv.val, "payload_encoding"));
      spec.ros_encoding = object_string(map_get(camera_kv.val, "ros_encoding"));
      spec.zmq_endpoint = object_string(map_get(camera_kv.val, "zmq_endpoint"));
      spec.zmq_topic = object_string(map_get(camera_kv.val, "zmq_topic"));
      validate_camera_payload_contract(spec);
      cameras_out.push_back(std::move(spec));
    }
  }
  return cameras_out;
}

std::vector<std::uint8_t> convert_real_bgra_to_rgb(
  std::span<const std::uint8_t> pixels,
  const CameraSpec & spec)
{
  if (spec.mode != CameraMode::Real) {
    throw std::invalid_argument("convert_real_bgra_to_rgb requires a Real camera");
  }
  if (spec.width <= 0 || spec.height <= 0) {
    throw std::runtime_error("unexpected real camera payload size");
  }
  const std::size_t expected = static_cast<std::size_t>(spec.width) *
    static_cast<std::size_t>(spec.height) * 4U;
  if (pixels.size() != expected) {
    throw std::runtime_error("unexpected real camera payload size");
  }

  std::vector<std::uint8_t> output(static_cast<std::size_t>(spec.width) *
    static_cast<std::size_t>(spec.height) * 3U);
  const bool legacy_linear = spec.real_payload_encoding == RealPayloadEncoding::Linear;
  const auto * lut = legacy_linear ? &legacy_linear_to_srgb_lut() : nullptr;
  for (std::size_t src = 0, dst = 0; src < pixels.size(); src += 4, dst += 3) {
    output[dst + 0] = lut ? (*lut)[pixels[src + 2]] : pixels[src + 2];
    output[dst + 1] = lut ? (*lut)[pixels[src + 1]] : pixels[src + 1];
    output[dst + 2] = lut ? (*lut)[pixels[src + 0]] : pixels[src + 0];
  }
  return output;
}

PinholeCalibration make_centered_pinhole_calibration(
  int width,
  int height,
  double vertical_fov_degrees)
{
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument(
      "camera calibration requires positive resolution, got " +
      std::to_string(width) + "x" + std::to_string(height));
  }
  if (!std::isfinite(vertical_fov_degrees) ||
    vertical_fov_degrees <= 0.0 || vertical_fov_degrees >= 180.0)
  {
    throw std::invalid_argument(
      "camera calibration requires vertical FOV in (0, 180) degrees, got " +
      std::to_string(vertical_fov_degrees));
  }

  const double vertical_fov_radians =
    vertical_fov_degrees * std::numbers::pi_v<double> / 180.0;
  const double fy = (static_cast<double>(height) / 2.0) /
    std::tan(vertical_fov_radians / 2.0);
  const double fx = fy;
  const double cx = (static_cast<double>(width) - 1.0) / 2.0;
  const double cy = (static_cast<double>(height) - 1.0) / 2.0;

  PinholeCalibration calibration;
  calibration.d = {0.0, 0.0, 0.0, 0.0, 0.0};
  calibration.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
  calibration.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  calibration.p = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0};
  return calibration;
}

void validate_direct_bridge_options(bool auto_enable_cameras)
{
  if (auto_enable_cameras) {
    throw std::invalid_argument(
      "auto_enable_cameras is not supported by the session-safe direct bridge; "
      "enable the camera through URLab or its existing control-session owner");
  }
}

void validate_camera_zmq_configuration(const CameraSpec & spec)
{
  if (!spec.zmq_broadcast_enabled) {
    throw std::runtime_error(
      "camera " + spec.articulation + ":" + spec.camera +
      " has Broadcast to ZMQ disabled; enable it in URLab before starting the bridge");
  }
}

}  // namespace urlab_image_transport_relay
