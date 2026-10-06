#include "khronos_ros/measurement_sink.h"

#include <cv_bridge/cv_bridge.hpp>
#include <hydra/common/global_info.h>

namespace khronos {

MeasurementSink::MeasurementSink(ianvs::NodeHandle nh)
    : publisher_(nh.create_publisher<khronos_msgs::msg::MeasurementFrame>(
          "measurements", rclcpp::QoS(10).reliable())) {}

void MeasurementSink::call(const FrameData& data, const VolumetricMap&,
                           const Tracks& tracks) const {
  const auto sequence = sequence_++;
  if (!publisher_->get_subscription_count()) {
    return;
  }

  auto msg = std::make_unique<khronos_msgs::msg::MeasurementFrame>();
  msg->header.stamp = rclcpp::Time(data.input.timestamp_ns);
  msg->header.frame_id = hydra::GlobalInfo::instance().getFrames().odom;
  msg->sequence = sequence;
  const auto pose = data.input.getSensorPose();
  msg->world_t_sensor.position.x = pose.translation().x();
  msg->world_t_sensor.position.y = pose.translation().y();
  msg->world_t_sensor.position.z = pose.translation().z();
  const Eigen::Quaterniond rotation(pose.rotation());
  msg->world_t_sensor.orientation.x = rotation.x();
  msg->world_t_sensor.orientation.y = rotation.y();
  msg->world_t_sensor.orientation.z = rotation.z();
  msg->world_t_sensor.orientation.w = rotation.w();

  // These matrices are the exact processed inputs; no independent topic sync or
  // camera backprojection is required by the offline consumer.
  auto copy_image = [&msg](const cv::Mat& input, const char* encoding) {
    return *cv_bridge::CvImage(msg->header, encoding, input).toImageMsg();
  };
  msg->depth = copy_image(data.input.depth_image, "32FC1");
  msg->vertices = copy_image(data.input.vertex_map, "32FC3");
  msg->semantic_labels = copy_image(data.input.label_image, "32SC1");
  msg->dynamic_labels = copy_image(data.dynamic_image, "32SC1");
  msg->object_labels = copy_image(data.object_image, "32SC1");
  for (const auto& track : tracks) {
    auto& out = msg->tracks.emplace_back();
    out.id = track.id;
    out.is_dynamic = track.is_dynamic;
    out.is_active = track.is_active;
    out.confidence = track.confidence;
    out.first_seen_ns = track.first_seen;
    out.last_seen_ns = track.last_seen;
    out.semantic_cluster_id = -1;
    out.dynamic_cluster_id = -1;
    for (auto it = track.observations.rbegin(); it != track.observations.rend(); ++it) {
      if (it->stamp == data.input.timestamp_ns) {
        out.semantic_cluster_id = it->semantic_cluster_id;
        out.dynamic_cluster_id = it->dynamic_cluster_id;
        break;
      }
      if (it->stamp < data.input.timestamp_ns) {
        break;
      }
    }
  }
  publisher_->publish(std::move(msg));
}

}  // namespace khronos
