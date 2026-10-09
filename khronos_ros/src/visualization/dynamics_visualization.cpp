#include "khronos_ros/visualization/dynamics_visualization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace khronos {
namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

float scalar(const cv::Mat& image, int v, int u) {
  return image.empty() ? kNaN : image.at<float>(v, u);
}

uint32_t color(const DynamicsEstimate& result, int v, int u) {
  float score = result.legacy_mode ? scalar(result.probability, v, u)
                                  : scalar(result.regularized_evidence, v, u);
  if (!std::isfinite(score)) score = scalar(result.motion_evidence, v, u);
  if (!std::isfinite(score)) return 0xff808080;
  score = std::clamp(score, 0.f, 1.f);
  return 0xff000000 | (static_cast<uint32_t>(255 * score) << 16) |
         static_cast<uint32_t>(255 * (1.f - score));
}

}  // namespace

cv::Mat dynamicsEvidenceImage(const DynamicsEstimate& result) {
  cv::Mat image(result.reference_points.size(), CV_8UC3);
  for (int v = 0; v < image.rows; ++v) for (int u = 0; u < image.cols; ++u) {
    const auto rgba = color(result, v, u);
    image.at<cv::Vec3b>(v, u) = cv::Vec3b((rgba >> 16) & 255, (rgba >> 8) & 255, rgba & 255);
  }
  return image;
}

sensor_msgs::msg::PointCloud2 dynamicsCloud(const DynamicsEstimate& result,
                                           const std_msgs::msg::Header& header) {
  using Field = sensor_msgs::msg::PointField;
  struct Description { const char* name; uint8_t type; };
  const std::array<Description, 17> core = {{
      {"x", Field::FLOAT32}, {"y", Field::FLOAT32}, {"z", Field::FLOAT32},
      {"probability", Field::FLOAT32}, {"observations", Field::INT32}, {"scoring_mode", Field::UINT32},
      {"motion_evidence", Field::FLOAT32}, {"regularized_evidence", Field::FLOAT32},
      {"reliability", Field::FLOAT32}, {"validity", Field::UINT32}, {"spatial_support", Field::UINT32},
      {"accepted_observations", Field::INT32}, {"temporal_span", Field::FLOAT32},
      {"failure_flags", Field::UINT32}, {"failures", Field::INT32}, {"track_alive", Field::UINT32},
      {"rgba", Field::UINT32}}};
  const std::array<Description, 21> debug = {{
      {"flow_error", Field::FLOAT32}, {"residual_norm", Field::FLOAT32}, {"innovation_sigma", Field::FLOAT32},
      {"static_cost", Field::FLOAT32}, {"dynamic_cost", Field::FLOAT32}, {"delta_cost", Field::FLOAT32},
      {"static_fit_cdf", Field::FLOAT32}, {"dynamic_fit_cdf", Field::FLOAT32},
      {"fitted_x", Field::FLOAT32}, {"fitted_y", Field::FLOAT32}, {"fitted_z", Field::FLOAT32},
      {"vx", Field::FLOAT32}, {"vy", Field::FLOAT32}, {"vz", Field::FLOAT32},
      {"velocity_cov_xx", Field::FLOAT32}, {"velocity_cov_xy", Field::FLOAT32},
      {"velocity_cov_xz", Field::FLOAT32}, {"velocity_cov_yy", Field::FLOAT32},
      {"velocity_cov_yz", Field::FLOAT32}, {"velocity_cov_zz", Field::FLOAT32},
      {"measurement_sigma", Field::FLOAT32}}};
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header = header;
  cloud.height = result.reference_points.rows;
  cloud.width = result.reference_points.cols;
  cloud.is_dense = false;
  const uint16_t endian = 1;
  cloud.is_bigendian = *reinterpret_cast<const uint8_t*>(&endian) == 0;
  const auto add_field = [&](const Description& description) {
    Field field;
    field.name = description.name;
    field.datatype = description.type;
    field.offset = cloud.point_step;
    field.count = 1;
    cloud.fields.push_back(field);
    cloud.point_step += 4;
  };
  for (const auto& field : core) add_field(field);
  if (result.debug_output) for (const auto& field : debug) add_field(field);
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(static_cast<size_t>(cloud.row_step) * cloud.height);
  auto* output = cloud.data.data();
  const auto write = [&](auto value) {
    static_assert(sizeof(value) == 4);
    std::memcpy(output, &value, 4);
    output += 4;
  };
  for (int v = 0; v < result.reference_points.rows; ++v) {
    for (int u = 0; u < result.reference_points.cols; ++u) {
      const auto point = result.reference_points.at<cv::Vec3f>(v, u);
      for (int axis = 0; axis < 3; ++axis) write(point[axis]);
      write(scalar(result.probability, v, u));
      write(result.observation_count.at<int32_t>(v, u));
      write(uint32_t(result.legacy_mode ? 0 : 1));
      write(scalar(result.motion_evidence, v, u));
      write(scalar(result.regularized_evidence, v, u));
      write(scalar(result.reliability, v, u));
      write(uint32_t(result.validity.at<uint8_t>(v, u)));
      write(uint32_t(result.spatial_support.at<uint8_t>(v, u)));
      write(result.accepted_count.at<int32_t>(v, u));
      write(scalar(result.temporal_span, v, u));
      write(uint32_t(result.failure_flags.at<uint16_t>(v, u)));
      write(result.failure_count.at<int32_t>(v, u));
      write(uint32_t(result.track_alive.at<uint8_t>(v, u) ? 1 : 0));
      write(color(result, v, u));
      if (result.debug_output) {
        for (const auto* image : {&result.forward_backward_error, &result.residual_norm, &result.innovation_sigma,
                                 &result.static_cost, &result.dynamic_cost, &result.delta_cost,
                                 &result.static_consistency, &result.dynamic_consistency}) write(scalar(*image, v, u));
        const auto fitted = result.fitted_point.empty() ? cv::Vec3f::all(kNaN)
                                                       : result.fitted_point.at<cv::Vec3f>(v, u);
        const auto velocity = result.velocity.empty() ? cv::Vec3f::all(kNaN) : result.velocity.at<cv::Vec3f>(v, u);
        for (int axis = 0; axis < 3; ++axis) write(fitted[axis]);
        for (int axis = 0; axis < 3; ++axis) write(velocity[axis]);
        const auto covariance = result.velocity_covariance.empty() ? cv::Vec<float, 6>::all(kNaN)
            : result.velocity_covariance.at<cv::Vec<float, 6>>(v, u);
        for (int i = 0; i < 6; ++i) write(covariance[i]);
        write(scalar(result.measurement_sigma, v, u));
      }
    }
  }
  return cloud;
}

}  // namespace khronos
