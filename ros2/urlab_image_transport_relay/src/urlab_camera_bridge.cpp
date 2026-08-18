#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "image_transport/image_transport.hpp"
#include "msgpack.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rmw/qos_profiles.h"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "urlab_image_transport_relay/runtime_description.hpp"
#include "zmq.h"

namespace
{

constexpr std::uint32_t kCameraMetaMagic = 0x314D4355;  // 'UCM1'
constexpr std::size_t kCameraMetaV1Size = 32;
constexpr std::size_t kCameraMetaV2Size = 40;

using urlab_image_transport_relay::CameraMode;
using urlab_image_transport_relay::CameraSpec;
using urlab_image_transport_relay::PinholeCalibration;
using urlab_image_transport_relay::RealPayloadEncoding;
using urlab_image_transport_relay::convert_real_bgra_to_rgb;
using urlab_image_transport_relay::make_centered_pinhole_calibration;
using urlab_image_transport_relay::make_describe_runtime_request;
using urlab_image_transport_relay::parse_describe_runtime_reply;
using urlab_image_transport_relay::validate_camera_zmq_configuration;
using urlab_image_transport_relay::validate_direct_bridge_options;

class UrlabRpcClient
{
public:
  UrlabRpcClient(std::string address, int step_port)
  : address_(std::move(address)), step_port_(step_port)
  {
    context_ = zmq_ctx_new();
    if (context_ == nullptr) {
      throw std::runtime_error("failed to create ZMQ context");
    }
    socket_ = zmq_socket(context_, ZMQ_REQ);
    if (socket_ == nullptr) {
      throw std::runtime_error("failed to create URLab RPC ZMQ socket");
    }
    const int linger_ms = 0;
    const int recv_timeout_ms = 30000;
    zmq_setsockopt(socket_, ZMQ_LINGER, &linger_ms, sizeof(linger_ms));
    zmq_setsockopt(socket_, ZMQ_RCVTIMEO, &recv_timeout_ms, sizeof(recv_timeout_ms));

    const std::string endpoint = address_ + ":" + std::to_string(step_port_);
    if (zmq_connect(socket_, endpoint.c_str()) != 0) {
      throw std::runtime_error("failed to connect URLab RPC endpoint " + endpoint);
    }
  }

  ~UrlabRpcClient()
  {
    if (socket_ != nullptr) {
      zmq_close(socket_);
      socket_ = nullptr;
    }
    if (context_ != nullptr) {
      zmq_ctx_term(context_);
      context_ = nullptr;
    }
  }

  std::vector<CameraSpec> describe_runtime()
  {
    const msgpack::sbuffer request = make_describe_runtime_request();
    const msgpack::object_handle reply = rpc(request);
    return parse_describe_runtime_reply(reply.get());
  }

  const std::string & address() const { return address_; }

private:
  msgpack::object_handle rpc(const msgpack::sbuffer & buffer)
  {
    const int send_rc = zmq_send(socket_, buffer.data(), buffer.size(), 0);
    if (send_rc < 0) {
      throw std::runtime_error("URLab RPC send failed");
    }

    zmq_msg_t msg;
    zmq_msg_init(&msg);
    const int recv_rc = zmq_msg_recv(&msg, socket_, 0);
    if (recv_rc < 0) {
      zmq_msg_close(&msg);
      throw std::runtime_error("URLab RPC recv timed out or failed");
    }
    const char * data = static_cast<const char *>(zmq_msg_data(&msg));
    const std::size_t size = static_cast<std::size_t>(zmq_msg_size(&msg));
    msgpack::object_handle handle = msgpack::unpack(data, size);
    zmq_msg_close(&msg);
    return handle;
  }

  std::string address_;
  int step_port_ = 5559;
  void * context_ = nullptr;
  void * socket_ = nullptr;
};

std::string request_key(const std::string & articulation, const std::string & camera)
{
  return articulation + ":" + camera;
}

std::pair<std::string, std::string> parse_camera_request(const std::string & raw)
{
  const std::size_t pos = raw.find(':');
  if (pos == std::string::npos || pos == 0 || pos == raw.size() - 1) {
    throw std::invalid_argument(
      "invalid camera request '" + raw + "'; expected ARTICULATION:CAMERA");
  }
  return {raw.substr(0, pos), raw.substr(pos + 1)};
}

std::string join_available_cameras(const std::vector<CameraSpec> & specs)
{
  std::ostringstream oss;
  for (std::size_t i = 0; i < specs.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << request_key(specs[i].articulation, specs[i].camera);
  }
  return oss.str();
}

std::string connect_endpoint_for_client(
  const std::string & advertised_endpoint,
  const std::string & urlab_address)
{
  const std::size_t colon = advertised_endpoint.rfind(':');
  if (colon == std::string::npos || colon == advertised_endpoint.size() - 1) {
    return advertised_endpoint;
  }
  return urlab_address + advertised_endpoint.substr(colon);
}

std::span<const std::uint8_t> strip_camera_meta(std::span<const std::uint8_t> payload)
{
  if (payload.size() >= kCameraMetaV1Size) {
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::memcpy(&magic, payload.data(), sizeof(magic));
    std::memcpy(&version, payload.data() + sizeof(magic), sizeof(version));
    if (magic == kCameraMetaMagic) {
      if (version >= 2 && payload.size() >= kCameraMetaV2Size) {
        return payload.subspan(kCameraMetaV2Size);
      }
      return payload.subspan(kCameraMetaV1Size);
    }
  }
  return payload;
}

bool recv_part(void * socket, std::vector<std::uint8_t> & out, int flags)
{
  zmq_msg_t msg;
  zmq_msg_init(&msg);
  const int rc = zmq_msg_recv(&msg, socket, flags);
  if (rc < 0) {
    zmq_msg_close(&msg);
    return false;
  }
  const auto * data = static_cast<const std::uint8_t *>(zmq_msg_data(&msg));
  const std::size_t size = static_cast<std::size_t>(zmq_msg_size(&msg));
  out.assign(data, data + size);
  zmq_msg_close(&msg);
  return true;
}

bool recv_camera_payload(void * socket, std::vector<std::uint8_t> & payload, int flags)
{
  std::vector<std::uint8_t> topic;
  if (!recv_part(socket, topic, flags)) {
    return false;
  }
  return recv_part(socket, payload, flags);
}

class CameraStream
{
public:
  CameraStream(
    rclcpp::Node * node,
    CameraSpec spec,
    std::string connect_endpoint,
    std::size_t queue_size,
    const std::string & output_suffix,
    bool publish_camera_info)
  : node_(node),
    spec_(std::move(spec)),
    calibration_(make_centered_pinhole_calibration(spec_.width, spec_.height, spec_.fovy)),
    connect_endpoint_(std::move(connect_endpoint)),
    publish_camera_info_(publish_camera_info)
  {
    auto qos = rmw_qos_profile_sensor_data;
    qos.depth = queue_size;
    const std::string image_topic =
      "/" + spec_.articulation + "/camera/" + spec_.camera + "/" + output_suffix;
    image_pub_ = image_transport::create_publisher(node_, image_topic, qos);
    if (publish_camera_info_) {
      camera_info_pub_ = node_->create_publisher<sensor_msgs::msg::CameraInfo>(
        "/" + spec_.articulation + "/camera/" + spec_.camera + "/camera_info",
        rclcpp::SensorDataQoS(rclcpp::KeepLast(queue_size)));
    }
  }

  ~CameraStream()
  {
    stop();
  }

  void start()
  {
    running_.store(true, std::memory_order_release);
    worker_ = std::thread([this]() { run(); });
  }

  void stop()
  {
    running_.store(false, std::memory_order_release);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  std::uint64_t published_count() const
  {
    return published_count_.load(std::memory_order_acquire);
  }

private:
  void run()
  {
    void * context = zmq_ctx_new();
    void * socket = zmq_socket(context, ZMQ_SUB);
    const int hwm = 4;
    const int timeout_ms = 200;
    const int linger_ms = 0;
    zmq_setsockopt(socket, ZMQ_RCVHWM, &hwm, sizeof(hwm));
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    zmq_setsockopt(socket, ZMQ_LINGER, &linger_ms, sizeof(linger_ms));
    zmq_setsockopt(socket, ZMQ_SUBSCRIBE, spec_.zmq_topic.c_str(), spec_.zmq_topic.size());

    if (zmq_connect(socket, connect_endpoint_.c_str()) != 0) {
      RCLCPP_ERROR(
        node_->get_logger(),
        "camera %s/%s failed to connect %s",
        spec_.articulation.c_str(),
        spec_.camera.c_str(),
        connect_endpoint_.c_str());
      zmq_close(socket);
      zmq_ctx_term(context);
      return;
    }

    RCLCPP_INFO(
      node_->get_logger(),
      "URLab camera bridge subscribed: %s/%s mode=%s endpoint=%s topic=%s",
      spec_.articulation.c_str(),
      spec_.camera.c_str(),
      spec_.mode_name.c_str(),
      connect_endpoint_.c_str(),
      spec_.zmq_topic.c_str());

    std::vector<std::uint8_t> latest;
    std::vector<std::uint8_t> next;
    while (running_.load(std::memory_order_acquire) && rclcpp::ok()) {
      if (!recv_camera_payload(socket, latest, 0)) {
        continue;
      }
      while (recv_camera_payload(socket, next, ZMQ_DONTWAIT)) {
        latest.swap(next);
      }
      try {
        publish(latest);
      } catch (const std::exception & exc) {
        RCLCPP_WARN_THROTTLE(
          node_->get_logger(),
          *node_->get_clock(),
          2000,
          "camera %s/%s publish failed: %s",
          spec_.articulation.c_str(),
          spec_.camera.c_str(),
          exc.what());
      }
    }

    zmq_close(socket);
    zmq_ctx_term(context);
  }

  void publish(const std::vector<std::uint8_t> & frame)
  {
    const std::span<const std::uint8_t> pixels = strip_camera_meta(
      std::span<const std::uint8_t>(frame.data(), frame.size()));
    sensor_msgs::msg::Image msg;
    msg.header.stamp = node_->now();
    msg.header.frame_id = spec_.articulation + "/" + spec_.camera + "_optical_frame";
    msg.height = static_cast<std::uint32_t>(spec_.height);
    msg.width = static_cast<std::uint32_t>(spec_.width);
    msg.is_bigendian = 0;

    if (spec_.mode == CameraMode::Depth) {
      publish_depth(pixels, msg);
    } else if (spec_.mode == CameraMode::Real) {
      publish_real(pixels, msg);
    } else {
      publish_segmentation(pixels, msg);
    }

    image_pub_.publish(msg);
    if (publish_camera_info_ && camera_info_pub_) {
      sensor_msgs::msg::CameraInfo info;
      info.header = msg.header;
      info.width = msg.width;
      info.height = msg.height;
      info.distortion_model = "plumb_bob";
      info.d.assign(calibration_.d.begin(), calibration_.d.end());
      info.k = calibration_.k;
      info.r = calibration_.r;
      info.p = calibration_.p;
      camera_info_pub_->publish(info);
    }
    published_count_.fetch_add(1, std::memory_order_acq_rel);
  }

  void publish_real(
    std::span<const std::uint8_t> pixels,
    sensor_msgs::msg::Image & msg) const
  {
    msg.encoding = "rgb8";
    msg.step = msg.width * 3;
    msg.data = convert_real_bgra_to_rgb(pixels, spec_);
  }

  void publish_depth(
    std::span<const std::uint8_t> pixels,
    sensor_msgs::msg::Image & msg) const
  {
    const std::size_t expected = static_cast<std::size_t>(spec_.width) *
      static_cast<std::size_t>(spec_.height) * sizeof(float);
    if (pixels.size() != expected) {
      throw std::runtime_error("unexpected depth camera payload size");
    }
    msg.encoding = "32FC1";
    msg.step = msg.width * sizeof(float);
    msg.data.resize(pixels.size());
    const auto * in = reinterpret_cast<const float *>(pixels.data());
    auto * out = reinterpret_cast<float *>(msg.data.data());
    const std::size_t n = static_cast<std::size_t>(spec_.width) *
      static_cast<std::size_t>(spec_.height);
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = in[i] / 100.0f;
    }
  }

  void publish_segmentation(
    std::span<const std::uint8_t> pixels,
    sensor_msgs::msg::Image & msg) const
  {
    const std::size_t expected = static_cast<std::size_t>(spec_.width) *
      static_cast<std::size_t>(spec_.height) * 4;
    if (pixels.size() != expected) {
      throw std::runtime_error("unexpected segmentation camera payload size");
    }
    msg.encoding = "bgra8";
    msg.step = msg.width * 4;
    msg.data.assign(pixels.begin(), pixels.end());
  }

  rclcpp::Node * node_ = nullptr;
  CameraSpec spec_;
  PinholeCalibration calibration_;
  std::string connect_endpoint_;
  bool publish_camera_info_ = true;
  image_transport::Publisher image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  std::atomic<bool> running_{false};
  std::atomic<std::uint64_t> published_count_{0};
  std::thread worker_;
};

class UrlabCameraBridgeNode : public rclcpp::Node
{
public:
  UrlabCameraBridgeNode()
  : rclcpp::Node("urlab_camera_bridge")
  {
    urlab_address_ = declare_parameter<std::string>("urlab_address", "tcp://127.0.0.1");
    step_port_ = declare_parameter<int>("step_port", 5559);
    camera_requests_ = declare_parameter<std::vector<std::string>>("cameras", std::vector<std::string>{});
    queue_size_ = static_cast<std::size_t>(
      std::max<std::int64_t>(1, declare_parameter<int>("queue_size", 1)));
    output_suffix_ = declare_parameter<std::string>("output_suffix", "image_transport");
    publish_camera_info_ = declare_parameter<bool>("publish_camera_info", true);
    auto_enable_cameras_ = declare_parameter<bool>("auto_enable_cameras", false);
    log_interval_s_ = declare_parameter<double>("log_interval_s", 1.0);

    validate_direct_bridge_options(auto_enable_cameras_);

    if (camera_requests_.empty()) {
      throw std::invalid_argument("parameter 'cameras' must contain ARTICULATION:CAMERA entries");
    }

    UrlabRpcClient client(urlab_address_, step_port_);
    std::vector<CameraSpec> available = client.describe_runtime();
    std::vector<CameraSpec> selected = select_cameras(available);

    for (const CameraSpec & spec : selected) {
      const char * conversion = spec.mode == CameraMode::Real &&
        spec.real_payload_encoding == RealPayloadEncoding::Linear
        ? "linear_to_srgb_then_bgra_to_rgb"
        : spec.mode == CameraMode::Real ? "bgra_to_rgb" : "native";
      RCLCPP_INFO(
        get_logger(),
        "URLab camera %s discovered: resolution=%dx%d enabled=%s streaming=%s "
        "zmq_enabled=%s zmq_active=%s payload=%s conversion=%s endpoint=%s topic=%s",
        request_key(spec.articulation, spec.camera).c_str(),
        spec.width,
        spec.height,
        spec.enabled ? "true" : "false",
        spec.streaming ? "true" : "false",
        spec.zmq_broadcast_enabled ? "true" : "false",
        spec.zmq_broadcast_active ? "true" : "false",
        spec.payload_encoding.c_str(),
        conversion,
        spec.zmq_endpoint.c_str(),
        spec.zmq_topic.c_str());
      validate_camera_zmq_configuration(spec);
      if (spec.width <= 0 || spec.height <= 0) {
        throw std::runtime_error(
          "camera " + request_key(spec.articulation, spec.camera) + " has invalid resolution");
      }
      if (spec.zmq_endpoint.empty() || spec.zmq_topic.empty()) {
        throw std::runtime_error(
          "camera " + request_key(spec.articulation, spec.camera) +
          " does not advertise a ZMQ endpoint/topic");
      }
      if (!spec.zmq_broadcast_active) {
        RCLCPP_WARN(
          get_logger(),
          "camera %s is not broadcasting yet; the bridge will wait for its owner to enable capture",
          request_key(spec.articulation, spec.camera).c_str());
      }
      auto stream = std::make_unique<CameraStream>(
        this,
        spec,
        connect_endpoint_for_client(spec.zmq_endpoint, client.address()),
        queue_size_,
        output_suffix_,
        publish_camera_info_);
      stream->start();
      streams_.push_back(std::move(stream));
    }

    if (log_interval_s_ > 0.0) {
      metrics_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(log_interval_s_)),
        [this]() { log_metrics(); });
    }
  }

  ~UrlabCameraBridgeNode() override
  {
    streams_.clear();
  }

private:
  std::vector<CameraSpec> select_cameras(const std::vector<CameraSpec> & available) const
  {
    std::unordered_map<std::string, CameraSpec> by_key;
    for (const CameraSpec & spec : available) {
      by_key.emplace(request_key(spec.articulation, spec.camera), spec);
    }

    std::vector<CameraSpec> selected;
    for (const std::string & raw : camera_requests_) {
      const auto [articulation, camera] = parse_camera_request(raw);
      const std::string key = request_key(articulation, camera);
      const auto it = by_key.find(key);
      if (it == by_key.end()) {
        throw std::runtime_error(
          "camera " + key + " not found. Available cameras: " +
          join_available_cameras(available));
      }
      selected.push_back(it->second);
    }
    return selected;
  }

  void log_metrics()
  {
    std::uint64_t total = 0;
    for (const auto & stream : streams_) {
      total += stream->published_count();
    }
    const std::uint64_t delta = total - last_total_published_;
    last_total_published_ = total;
    RCLCPP_INFO(
      get_logger(),
      "URLab camera bridge metrics: published_fps=%.3f total=%lu streams=%zu",
      static_cast<double>(delta) / log_interval_s_,
      static_cast<unsigned long>(total),
      streams_.size());
  }

  std::string urlab_address_;
  int step_port_ = 5559;
  std::vector<std::string> camera_requests_;
  std::size_t queue_size_ = 1;
  std::string output_suffix_ = "image_transport";
  bool publish_camera_info_ = true;
  bool auto_enable_cameras_ = false;
  double log_interval_s_ = 1.0;
  std::vector<std::unique_ptr<CameraStream>> streams_;
  rclcpp::TimerBase::SharedPtr metrics_timer_;
  std::uint64_t last_total_published_ = 0;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<UrlabCameraBridgeNode>());
  } catch (const std::exception & exc) {
    RCLCPP_FATAL(rclcpp::get_logger("urlab_camera_bridge"), "%s", exc.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
