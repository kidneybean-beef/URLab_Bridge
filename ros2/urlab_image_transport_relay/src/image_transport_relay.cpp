#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rmw/qos_profiles.h"
#include "sensor_msgs/msg/image.hpp"

class UrlabImageTransportRelay : public rclcpp::Node
{
public:
  UrlabImageTransportRelay()
  : rclcpp::Node("urlab_image_transport_relay")
  {
    input_topic_ = declare_parameter<std::string>("input_topic", "/urlab/image_raw");
    output_topic_ = declare_parameter<std::string>("output_topic", "/urlab/image");
    queue_size_ = declare_parameter<int>("queue_size", 1);

    if (input_topic_ == output_topic_) {
      throw std::invalid_argument(
        "input_topic and output_topic must be different; publishing image_transport "
        "onto the subscribed raw topic can create a self-loop through the raw plugin");
    }

    const auto queue_size = static_cast<std::size_t>(std::max(1, queue_size_));
    auto publisher_qos = rmw_qos_profile_sensor_data;
    publisher_qos.depth = queue_size;
    publisher_ = image_transport::create_publisher(this, output_topic_, publisher_qos);

    auto subscriber_qos = rclcpp::QoS(rclcpp::KeepLast(queue_size));
    subscriber_qos.best_effort();
    subscriber_qos.durability_volatile();
    subscription_ = create_subscription<sensor_msgs::msg::Image>(
      input_topic_,
      subscriber_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        publisher_.publish(msg);
      });

    RCLCPP_INFO(
      get_logger(),
      "URLab image_transport relay: input=%s output=%s queue_size=%zu",
      input_topic_.c_str(),
      output_topic_.c_str(),
      queue_size);
  }

private:
  std::string input_topic_;
  std::string output_topic_;
  int queue_size_ = 1;
  image_transport::Publisher publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<UrlabImageTransportRelay>());
  } catch (const std::exception & exc) {
    RCLCPP_FATAL(rclcpp::get_logger("urlab_image_transport_relay"), "%s", exc.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
