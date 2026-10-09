#pragma once

#include <khronos/active_window/motion_detection/dynamics_estimator.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

namespace khronos {

sensor_msgs::msg::PointCloud2 dynamicsCloud(const DynamicsEstimate& estimate,
                                           const std_msgs::msg::Header& header);
cv::Mat dynamicsEvidenceImage(const DynamicsEstimate& estimate);

}  // namespace khronos
