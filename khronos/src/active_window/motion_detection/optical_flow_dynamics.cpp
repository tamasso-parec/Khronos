#include "khronos/active_window/motion_detection/optical_flow_dynamics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>
#include <boost/math/distributions/chi_squared.hpp>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace khronos {
namespace {

const auto registration =
    config::RegistrationWithConfig<DynamicsEstimator, OpticalFlowDynamics,
                                   OpticalFlowDynamics::Config>("OpticalFlowDynamics");
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

bool inside(const cv::Mat& image, const cv::Vec2f& p) {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && p[0] >= 0.f && p[1] >= 0.f &&
         p[0] < image.cols - 1 && p[1] < image.rows - 1;
}

template <typename T>
T interpolate(const cv::Mat& image, const cv::Vec2f& p) {
  const int x = static_cast<int>(p[0]);
  const int y = static_cast<int>(p[1]);
  const float dx = p[0] - x;
  const float dy = p[1] - y;
  return (image.at<T>(y, x) * (1.f - dx) + image.at<T>(y, x + 1) * dx) * (1.f - dy) +
         (image.at<T>(y + 1, x) * (1.f - dx) + image.at<T>(y + 1, x + 1) * dx) * dy;
}

struct PointSample {
  Eigen::Vector3d point;
  Eigen::Matrix3d jacobian;
  double depth_variance = 0.0;
};

uint16_t samplePoint(const hydra::InputData& input,
                     const hydra::Camera::Config& camera,
                     const cv::Vec2f& p,
                     const OpticalFlowDynamics::Config& config,
                     PointSample& sample, int footprint_radius = 0) {
  if (!inside(input.depth_image, p)) {
    return kOutsideImage;
  }
  const int x = static_cast<int>(p[0]);
  const int y = static_cast<int>(p[1]);
  const double dx = p[0] - x;
  const double dy = p[1] - y;
  double depths[2][2];
  double min_depth = std::numeric_limits<double>::infinity();
  double max_depth = 0.0;
  sample.depth_variance = 0.0;
  for (int iy = 0; iy < 2; ++iy) {
    for (int ix = 0; ix < 2; ++ix) {
      const double z = input.depth_image.at<float>(y + iy, x + ix);
      if (!std::isfinite(z) || z <= 0.0) {
        return kInvalidDepth;
      }
      depths[iy][ix] = z;
      min_depth = std::min(min_depth, z);
      max_depth = std::max(max_depth, z);
      const double weight = (ix ? dx : 1.0 - dx) * (iy ? dy : 1.0 - dy);
      const double sigma = config.depth_sigma + config.depth_sigma_scale * z * z;
      sample.depth_variance += weight * weight * sigma * sigma;
    }
  }
  if (config.scoring_mode == "temporal") {
    const int radius = std::max(config.depth_neighborhood_radius, std::min(footprint_radius, 3));
    const int y0 = std::max(0, y - radius), y1 = std::min(input.depth_image.rows - 1, y + radius + 1);
    const int x0 = std::max(0, x - radius), x1 = std::min(input.depth_image.cols - 1, x + radius + 1);
    for (int iy = y0; iy <= y1; ++iy) {
      for (int ix = x0; ix <= x1; ++ix) {
        const double z = input.depth_image.at<float>(iy, ix);
        if (!std::isfinite(z) || z <= 0.0) return kInvalidDepth;
        for (const auto& offset : {cv::Point(1, 0), cv::Point(0, 1)}) {
          if (ix + offset.x > x1 || iy + offset.y > y1) continue;
          const double neighbor = input.depth_image.at<float>(iy + offset.y, ix + offset.x);
          if (!std::isfinite(neighbor) || neighbor <= 0.0) return kInvalidDepth;
          const double tolerance = config.max_depth_jump + config.depth_jump_relative * std::min(z, neighbor);
          if (std::abs(z - neighbor) > tolerance) return kDepthDiscontinuity;
        }
      }
    }
  } else if (max_depth - min_depth > config.max_depth_jump) {
    return kDepthDiscontinuity;
  }
  const double z = interpolate<float>(input.depth_image, p);
  sample.point = Eigen::Vector3d((p[0] - camera.cx) * z / camera.fx,
                                 (p[1] - camera.cy) * z / camera.fy, z);
  if (!sample.point.allFinite() || !input.inRange(sample.point.norm())) {
    return kOutOfRange;
  }
  const Eigen::Vector2d gradient(
      (1.0 - dy) * (depths[0][1] - depths[0][0]) + dy * (depths[1][1] - depths[1][0]),
      (1.0 - dx) * (depths[1][0] - depths[0][0]) + dx * (depths[1][1] - depths[0][1]));
  sample.jacobian = rgbdPointJacobian(p[0], p[1], z, camera.fx, camera.fy,
                                     camera.cx, camera.cy, gradient);
  return 0;
}

// Mean absolute gray difference between 5x5 patches; NaN if either patch leaves the image.
double patchDifference(const cv::Mat& reference, const cv::Vec2f& p_reference,
                       const cv::Mat& current, const cv::Vec2f& p_current) {
  const cv::Vec2f margin(2.f, 2.f);
  if (!inside(reference, p_reference - margin) || !inside(reference, p_reference + margin) ||
      !inside(current, p_current - margin) || !inside(current, p_current + margin)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const auto sample = [](const cv::Mat& image, float x, float y) {
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const float dx = x - x0, dy = y - y0;
    return (image.at<uint8_t>(y0, x0) * (1.f - dx) + image.at<uint8_t>(y0, x0 + 1) * dx) * (1.f - dy) +
           (image.at<uint8_t>(y0 + 1, x0) * (1.f - dx) + image.at<uint8_t>(y0 + 1, x0 + 1) * dx) * dy;
  };
  double sum = 0.0;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      sum += std::abs(sample(reference, p_reference[0] + dx, p_reference[1] + dy) -
                      sample(current, p_current[0] + dx, p_current[1] + dy));
    }
  }
  return sum / 25.0;
}

Eigen::Matrix2d flowJacobian(const cv::Mat& flow, const cv::Vec2f& p) {
  const int x = static_cast<int>(p[0]);
  const int y = static_cast<int>(p[1]);
  const float dx = p[0] - x;
  const float dy = p[1] - y;
  const auto a = flow.at<cv::Vec2f>(y, x);
  const auto b = flow.at<cv::Vec2f>(y, x + 1);
  const auto c = flow.at<cv::Vec2f>(y + 1, x);
  const auto d = flow.at<cv::Vec2f>(y + 1, x + 1);
  const auto du = (b - a) * (1.f - dy) + (d - c) * dy;
  const auto dv = (c - a) * (1.f - dx) + (d - b) * dx;
  Eigen::Matrix2d result;
  result << 1.0 + du[0], dv[0], du[1], 1.0 + dv[1];
  return result;
}

}  // namespace

cv::Mat structureTensor(const cv::Mat& gray, int window) {
  // Same derivative scaling as cv::cornerMinEigenVal for 8-bit input, so min_texture applies.
  const double scale = 1.0 / (4.0 * 3.0 * 255.0);
  cv::Mat dx, dy, tensor;
  cv::Sobel(gray, dx, CV_32F, 1, 0, 3, scale);
  cv::Sobel(gray, dy, CV_32F, 0, 1, 3, scale);
  cv::merge(std::vector<cv::Mat>{dx.mul(dx), dx.mul(dy), dy.mul(dy)}, tensor);
  cv::boxFilter(tensor, tensor, CV_32F, cv::Size(window, window));
  return tensor * 9.0;  // Window mean expressed as an unnormalized 3x3 block sum.
}

void declare_config(OpticalFlowDynamics::Config& config) {
  using namespace config;
  name("OpticalFlowDynamics");
  field(config.scoring_mode, "scoring_mode");
  field(config.debug_output, "debug_output");
  field(config.position_sigma_floor, "position_sigma_floor");
  field(config.depth_jump_relative, "depth_jump_relative");
  field(config.depth_neighborhood_radius, "depth_neighborhood_radius");
  field(config.texture_variance_scale, "texture_variance_scale");
  field(config.fb_variance_scale, "fb_variance_scale");
  field(config.aperture_sigma, "aperture_sigma", "px");
  field(config.ambiguity_check, "ambiguity_check");
  field(config.ambiguity_min_separation, "ambiguity_min_separation", "px");
  field(config.photometric_tolerance, "photometric_tolerance", "gray levels");
  field(config.min_fit_observations, "min_fit_observations");
  field(config.min_temporal_span, "min_temporal_span");
  field(config.min_reliability, "min_reliability");
  field(config.max_fit_cdf, "max_fit_cdf");
  field(config.motion_penalty, "motion_penalty");
  field(config.motion_score_scale, "motion_score_scale");
  field(config.min_displacement, "min_displacement");
  field(config.min_displacement_sigma, "min_displacement_sigma");
  field(config.spatial_enabled, "spatial_enabled");
  field(config.spatial_radius, "spatial_radius");
  field(config.spatial_min_neighbors, "spatial_min_neighbors");
  field(config.spatial_min_reliability, "spatial_min_reliability");
  field(config.spatial_color_sigma, "spatial_color_sigma");
  field(config.spatial_velocity_tolerance, "spatial_velocity_tolerance");
  field(config.spatial_motion_fraction, "spatial_motion_fraction");

  field(config.window_size, "window_size");
  field(config.min_observations, "min_observations");
  field(config.max_frame_gap, "max_frame_gap", "s");
  field(config.max_window_duration, "max_window_duration", "s");
  field(config.reference_pixel_sigma, "reference_pixel_sigma", "px");
  field(config.flow_sigma, "flow_sigma", "px");
  field(config.depth_sigma, "depth_sigma", "m");
  field(config.depth_sigma_scale, "depth_sigma_scale", "1/m");
  field(config.max_forward_backward_error, "max_forward_backward_error", "px");
  field(config.max_depth_jump, "max_depth_jump", "m");
  field(config.min_texture, "min_texture");
  field(config.flow_levels, "flow_levels");
  field(config.flow_window_size, "flow_window_size", "px");
  field(config.flow_iterations, "flow_iterations");

  check(config.window_size, GE, 2, "window_size");
  check(config.min_observations, GE, 1, "min_observations");
  checkCondition(config.min_observations < config.window_size,
                 "min_observations must be smaller than window_size");
  check(config.max_frame_gap, GT, 0.0, "max_frame_gap");
  check(config.max_window_duration, GT, 0.0, "max_window_duration");
  check(config.reference_pixel_sigma, GE, 0.0, "reference_pixel_sigma");
  check(config.flow_sigma, GT, 0.0, "flow_sigma");
  check(config.depth_sigma, GT, 0.0, "depth_sigma");
  check(config.depth_sigma_scale, GE, 0.0, "depth_sigma_scale");
  check(config.max_forward_backward_error, GT, 0.0, "max_forward_backward_error");
  check(config.max_depth_jump, GT, 0.0, "max_depth_jump");
  check(config.min_texture, GE, 0.0, "min_texture");
  checkCondition(std::isfinite(config.max_frame_gap) &&
                     std::isfinite(config.max_window_duration) &&
                     std::isfinite(config.reference_pixel_sigma) && std::isfinite(config.flow_sigma) &&
                     std::isfinite(config.depth_sigma) && std::isfinite(config.depth_sigma_scale) &&
                     std::isfinite(config.max_forward_backward_error) &&
                     std::isfinite(config.max_depth_jump) && std::isfinite(config.min_texture),
                 "floating-point parameters must be finite");
  check(config.flow_levels, GE, 1, "flow_levels");
  check(config.flow_window_size, GE, 3, "flow_window_size");
  checkCondition(config.flow_window_size % 2 == 1, "flow_window_size must be odd");
  check(config.flow_iterations, GE, 1, "flow_iterations");
  checkCondition(config.scoring_mode == "legacy" || config.scoring_mode == "temporal",
                 "scoring_mode must be legacy or temporal");
  check(config.position_sigma_floor, GT, 0.0, "position_sigma_floor");
  check(config.depth_jump_relative, GE, 0.0, "depth_jump_relative");
  checkInRange(config.depth_neighborhood_radius, 0, 3, "depth_neighborhood_radius");
  check(config.texture_variance_scale, GE, 0.0, "texture_variance_scale");
  check(config.fb_variance_scale, GE, 0.0, "fb_variance_scale");
  check(config.aperture_sigma, GE, 0.0, "aperture_sigma");
  check(config.ambiguity_min_separation, GE, 0.0, "ambiguity_min_separation");
  check(config.photometric_tolerance, GE, 0.0, "photometric_tolerance");
  check(config.min_fit_observations, GE, 3, "min_fit_observations");
  checkCondition(config.scoring_mode != "temporal" || config.min_fit_observations <= config.window_size,
                 "temporal min_fit_observations must not exceed window_size");
  check(config.min_temporal_span, GT, 0.0, "min_temporal_span");
  checkInRange(config.min_reliability, 0.0, 1.0, "min_reliability");
  checkInRange(config.max_fit_cdf, 0.0, 1.0, "max_fit_cdf", false, false);
  check(config.motion_penalty, GE, 0.0, "motion_penalty");
  check(config.motion_score_scale, GT, 0.0, "motion_score_scale");
  check(config.min_displacement, GE, 0.0, "min_displacement");
  check(config.min_displacement_sigma, GT, 0.0, "min_displacement_sigma");
  checkInRange(config.spatial_radius, 1, 3, "spatial_radius");
  check(config.spatial_min_neighbors, GE, 2, "spatial_min_neighbors");
  checkCondition(config.spatial_min_neighbors <= (2 * config.spatial_radius + 1) *
                     (2 * config.spatial_radius + 1) - 1, "spatial_min_neighbors exceeds footprint");
  checkInRange(config.spatial_min_reliability, 0.0, 1.0, "spatial_min_reliability", false, true);
  check(config.spatial_color_sigma, GT, 0.0, "spatial_color_sigma");
  check(config.spatial_velocity_tolerance, GT, 0.0, "spatial_velocity_tolerance");
  checkInRange(config.spatial_motion_fraction, 0.5, 1.0, "spatial_motion_fraction");
  checkCondition(std::isfinite(config.position_sigma_floor) && std::isfinite(config.depth_jump_relative) &&
                     std::isfinite(config.texture_variance_scale) && std::isfinite(config.fb_variance_scale) && std::isfinite(config.aperture_sigma) &&
                     std::isfinite(config.ambiguity_min_separation) &&
                     std::isfinite(config.photometric_tolerance) &&
                     std::isfinite(config.min_temporal_span) && std::isfinite(config.min_reliability) &&
                     std::isfinite(config.max_fit_cdf) && std::isfinite(config.motion_penalty) &&
                     std::isfinite(config.motion_score_scale) && std::isfinite(config.min_displacement) &&
                     std::isfinite(config.min_displacement_sigma) && std::isfinite(config.spatial_min_reliability) &&
                     std::isfinite(config.spatial_color_sigma) && std::isfinite(config.spatial_velocity_tolerance) &&
                     std::isfinite(config.spatial_motion_fraction), "temporal parameters must be finite");

}

OpticalFlowDynamics::OpticalFlowDynamics(const Config& config)
    : config(config::checkValid(config)) {
  // cdf(cost) <= max_fit_cdf  <=>  cost <= quantile; avoids two incomplete gammas per pixel.
  fit_thresholds_.assign(3 * this->config.window_size + 1, 0.0);
  for (size_t dof = 1; dof < fit_thresholds_.size(); ++dof) {
    fit_thresholds_[dof] = boost::math::quantile(
        boost::math::chi_squared_distribution<double>(dof), this->config.max_fit_cdf);
  }
}

void OpticalFlowDynamics::reset() {
  estimate_ = DynamicsEstimate();
  previous_gray_.release();
  tracked_pixels_.release();
  previous_texture_.release();
  previous_tensor_.release();
  reference_gray_.release();
  uncertainties_.clear();
  temporal_models_.clear();
  pixel_covariances_.clear();
  reference_color_.release();
  reference_depth_.release();
  sensor_name_.clear();
}

DynamicsEstimate::ConstPtr OpticalFlowDynamics::processInput(const hydra::InputData& input) {
  const auto* camera = dynamic_cast<const hydra::Camera*>(&input.getSensor());
  const auto size = input.color_image.size();
  if (!camera || input.color_image.type() != CV_8UC3 || size.width < 3 || size.height < 3 ||
      input.depth_image.type() != CV_32FC1 || input.depth_image.size() != size ||
      input.vertex_map.type() != CV_32FC3 || input.vertex_map.size() != size ||
      size.width != camera->getConfig().width || size.height != camera->getConfig().height ||
      !input.getSensorPose().matrix().allFinite()) {
    LOG(WARNING) << "[OpticalFlowDynamics] Expected registered RGB-D camera input, vertex map "
                    "and finite pose; resetting window.";
    reset();
    return nullptr;
  }

  cv::Mat gray;
  cv::cvtColor(input.color_image, gray, cv::COLOR_RGB2GRAY);
  const auto& c = camera->getConfig();
  const bool new_camera = sensor_name_ != camera->name || previous_gray_.size() != size ||
                          c.fx != camera_config_.fx || c.fy != camera_config_.fy ||
                          c.cx != camera_config_.cx || c.cy != camera_config_.cy;
  const bool restart = previous_gray_.empty() || new_camera ||
                       estimate_.num_frames >= config.window_size ||
                       input.timestamp_ns <= estimate_.latest_timestamp_ns ||
                       hydra::toSeconds(input.timestamp_ns - estimate_.latest_timestamp_ns) >
                           config.max_frame_gap ||
                       hydra::toSeconds(input.timestamp_ns - estimate_.reference_timestamp_ns) >
                           config.max_window_duration;
  if (restart) {
    camera_config_ = c;
    sensor_name_ = camera->name;
    startWindow(input, gray);
  } else {
    track(input, gray);
    ++estimate_.num_frames;
  }
  estimate_.latest_timestamp_ns = input.timestamp_ns;
  previous_gray_ = gray;
  if (!estimate_.legacy_mode) regularizeSpatial();
  return snapshot();
}

void OpticalFlowDynamics::startWindow(const hydra::InputData& input, const cv::Mat& gray) {
  estimate_ = DynamicsEstimate();
  estimate_.reference_timestamp_ns = input.timestamp_ns;
  estimate_.num_frames = 1;
  estimate_.legacy_mode = config.scoring_mode == "legacy";
  estimate_.debug_output = config.debug_output;
  estimate_.reference_points = cv::Mat(gray.size(), CV_32FC3, cv::Scalar::all(kNaN));
  estimate_.probability = cv::Mat(gray.size(), CV_32FC1, cv::Scalar(kNaN));
  estimate_.squared_error = cv::Mat::zeros(gray.size(), CV_64FC1);
  estimate_.observation_count = cv::Mat::zeros(gray.size(), CV_32SC1);
  estimate_.failure_flags = cv::Mat::zeros(gray.size(), CV_16UC1);
  estimate_.failure_count = cv::Mat::zeros(gray.size(), CV_32SC1);
  estimate_.track_alive = cv::Mat::zeros(gray.size(), CV_8UC1);
  estimate_.forward_backward_error = cv::Mat(gray.size(), CV_32FC1, cv::Scalar(kNaN));
  estimate_.residual_norm = cv::Mat(gray.size(), CV_32FC1, cv::Scalar(kNaN));
  estimate_.innovation_sigma = cv::Mat(gray.size(), CV_32FC1, cv::Scalar(kNaN));
  estimate_.motion_evidence = cv::Mat(gray.size(), CV_32FC1, cv::Scalar(kNaN));
  estimate_.regularized_evidence = cv::Mat(gray.size(), CV_32FC1, cv::Scalar(kNaN));
  estimate_.reliability = cv::Mat::zeros(gray.size(), CV_32FC1);
  estimate_.validity = cv::Mat::zeros(gray.size(), CV_8UC1);
  estimate_.spatial_support = cv::Mat::zeros(gray.size(), CV_8UC1);
  estimate_.accepted_count = cv::Mat::zeros(gray.size(), CV_32SC1);
  estimate_.temporal_span = cv::Mat::zeros(gray.size(), CV_32FC1);
  estimate_.velocity = cv::Mat(gray.size(), CV_32FC3, cv::Scalar::all(kNaN));
  if (config.debug_output) {
    estimate_.measurement_sigma = cv::Mat(gray.size(), CV_32FC1, cv::Scalar(kNaN));
    estimate_.fitted_point = cv::Mat(gray.size(), CV_32FC3, cv::Scalar::all(kNaN));
    estimate_.velocity_covariance.create(gray.size(), CV_32FC(6));
    std::fill_n(estimate_.velocity_covariance.ptr<float>(), gray.total() * 6, kNaN);
    for (auto* image : {&estimate_.static_cost, &estimate_.dynamic_cost, &estimate_.delta_cost,
                       &estimate_.static_consistency, &estimate_.dynamic_consistency}) {
      *image = cv::Mat(gray.size(), CV_32FC1, cv::Scalar(kNaN));
    }
  }
  tracked_pixels_.create(gray.size(), CV_32FC2);
  uncertainties_.clear();
  temporal_models_.clear();
  pixel_covariances_.clear();
  if (estimate_.legacy_mode) {
    uncertainties_.resize(gray.total());
  } else {
    temporal_models_.resize(gray.total());
    pixel_covariances_.resize(gray.total());
    reference_color_ = input.color_image.clone();
    reference_depth_ = input.depth_image.clone();
    reference_gray_ = gray;  // A fresh buffer per input; never written afterwards.
  }
  cv::Mat texture;
  cv::cornerMinEigenVal(gray, texture, 3);
  const Eigen::Isometry3d world_T_sensor = input.getSensorPose();
  const Eigen::Isometry3d sensor_T_world = world_T_sensor.inverse();

  for (int v = 0; v < gray.rows; ++v) {
    for (int u = 0; u < gray.cols; ++u) {
      const cv::Vec2f pixel(u, v);
      tracked_pixels_.at<cv::Vec2f>(v, u) = pixel;
      auto& flags = estimate_.failure_flags.at<uint16_t>(v, u);
      const auto raw = input.vertex_map.at<cv::Vec3f>(v, u);
      Eigen::Vector3d point(raw[0], raw[1], raw[2]);
      const Eigen::Vector3d sensor_point =
          input.points_in_world_frame ? sensor_T_world * point : point;
      const float depth = input.depth_image.at<float>(v, u);
      if (!sensor_point.allFinite() || sensor_point.z() <= 0.0 ||
          !std::isfinite(depth) || depth <= 0.f || !input.inRange(sensor_point.norm())) {
        flags = kInvalidReference;
        continue;
      }
      if (!input.points_in_world_frame) {
        point = world_T_sensor * point;
      }
      estimate_.reference_points.at<cv::Vec3f>(v, u) = cv::Vec3f(point.x(), point.y(), point.z());
      PointSample sample;
      flags = samplePoint(input, camera_config_, pixel, config, sample);
      if (texture.at<float>(v, u) < config.min_texture) {
        flags |= kLowTexture;
      }
      if (flags & ~kLowTexture || (estimate_.legacy_mode && flags)) {
        continue;
      }
      const int index = v * gray.cols + u;
      if (!estimate_.legacy_mode) {
        const double texture_ratio = config.min_texture > 0.0
            ? std::max(0.0, 1.0 - texture.at<float>(v, u) / config.min_texture) : 0.0;
        const double inflation = 1.0 + config.texture_variance_scale * texture_ratio;
        pixel_covariances_[index] = config.reference_pixel_sigma * config.reference_pixel_sigma *
                                    inflation * Eigen::Matrix2d::Identity();
        const Eigen::Matrix3d jacobian = world_T_sensor.rotation() * sample.jacobian;
        Eigen::Matrix3d covariance = jacobian.leftCols<2>() * pixel_covariances_[index] *
                                     jacobian.leftCols<2>().transpose() +
            sample.depth_variance * jacobian.col(2) * jacobian.col(2).transpose();
        covariance.diagonal().array() += config.position_sigma_floor * config.position_sigma_floor;
        if (!temporal_models_[index].add(point, covariance, 0.0, 1.0 / std::sqrt(inflation))) {
          flags |= kNumericalFailure;
          continue;
        }
        scoreTemporal(v, u);
      } else {
        uncertainties_[index].initialize(
          world_T_sensor.rotation() * sample.jacobian,
          config.reference_pixel_sigma * config.reference_pixel_sigma, sample.depth_variance);
      }
      if (estimate_.legacy_mode) {
        estimate_.accepted_count.at<int>(v, u) = 1;
        estimate_.reliability.at<float>(v, u) = 1.f;
        estimate_.validity.at<uint8_t>(v, u) = static_cast<uint8_t>(DynamicsValidity::kInsufficient);
      }
      estimate_.track_alive.at<uint8_t>(v, u) = 255;
    }
  }
  previous_texture_ = texture;
  if (!estimate_.legacy_mode) previous_tensor_ = structureTensor(gray, config.flow_window_size);
}

void OpticalFlowDynamics::track(const hydra::InputData& input, const cv::Mat& gray) {
  cv::Mat forward, backward, texture;
  cv::calcOpticalFlowFarneback(previous_gray_, gray, forward, 0.5, config.flow_levels,
                             config.flow_window_size, config.flow_iterations, 5, 1.2, 0);
  cv::calcOpticalFlowFarneback(gray, previous_gray_, backward, 0.5, config.flow_levels,
                             config.flow_window_size, config.flow_iterations, 5, 1.2, 0);
  cv::cornerMinEigenVal(gray, texture, 3);
  const Eigen::Isometry3d world_T_sensor = input.getSensorPose();
  const Eigen::Isometry3d sensor_T_world = world_T_sensor.inverse();

  for (int v = 0; v < gray.rows; ++v) {
    for (int u = 0; u < gray.cols; ++u) {
      auto& alive = estimate_.track_alive.at<uint8_t>(v, u);
      if (!alive) {
        continue;
      }
      auto& flags = estimate_.failure_flags.at<uint16_t>(v, u);
      auto& failures = estimate_.failure_count.at<int>(v, u);
      auto& cycle_error = estimate_.forward_backward_error.at<float>(v, u);
      cycle_error = kNaN;
      auto& pixel = tracked_pixels_.at<cv::Vec2f>(v, u);
      const auto next = pixel + interpolate<cv::Vec2f>(forward, pixel);
      if (!inside(gray, next)) {
        flags |= std::isfinite(next[0]) && std::isfinite(next[1]) ? kOutsideImage : kNonFiniteFlow;
        ++failures;
        // Lost correspondence ends the track; accepted observations keep their fit.
        alive = 0;
        continue;
      }
      const auto cycle = next + interpolate<cv::Vec2f>(backward, next) - pixel;
      const Eigen::Matrix2d transition = flowJacobian(forward, pixel);
      uint16_t flow_failure = 0;
      if (!std::isfinite(cycle[0]) || !std::isfinite(cycle[1]) || !transition.allFinite()) {
        flow_failure |= kNonFiniteFlow;
      } else {
        cycle_error = cv::norm(cycle);
        if (cycle_error > config.max_forward_backward_error) {
          flow_failure |= kFlowInconsistent;
        }
      }
      if (interpolate<float>(previous_texture_, pixel) < config.min_texture ||
          interpolate<float>(texture, next) < config.min_texture) {
        flow_failure |= kLowTexture;
      }
      PointSample sample;
      int footprint = 0;
      const int index = v * gray.cols + u;
      if (!estimate_.legacy_mode) {
        const double variance = pixel_covariances_[index].trace() * 0.5;
        footprint = std::isfinite(variance) ? static_cast<int>(std::min(3.0, std::ceil(std::sqrt(std::max(0.0, variance))))) : 3;
      }
      const uint16_t depth_failure = samplePoint(input, camera_config_, next, config, sample, footprint);
      flags |= flow_failure | depth_failure;
      if (flow_failure & ~kLowTexture || (estimate_.legacy_mode && flow_failure)) {
        ++failures;
        alive = 0;
        continue;
      }
      double quality = 1.0;
      if (estimate_.legacy_mode) {
        uncertainties_[index].predict(transition, config.flow_sigma * config.flow_sigma);
      } else {
        const double local_texture = std::min(interpolate<float>(previous_texture_, pixel),
                                              interpolate<float>(texture, next));
        const double texture_ratio = config.min_texture > 0.0
            ? std::max(0.0, 1.0 - local_texture / config.min_texture) : 0.0;
        const double inflation = 1.0 + config.texture_variance_scale * texture_ratio +
            config.fb_variance_scale * std::pow(cycle_error / config.max_forward_backward_error, 2.0);
        quality = 1.0 / std::sqrt(inflation);
        auto& covariance = pixel_covariances_[index];
        covariance = (transition * covariance * transition.transpose()).eval();
        covariance.diagonal().array() += config.flow_sigma * config.flow_sigma * inflation;
        if (config.aperture_sigma > 0.0 && config.min_texture > 0.0) {
          // Lucas-Kanade-style aperture model: weakly textured directions are poorly observed.
          const auto g = interpolate<cv::Vec3f>(previous_tensor_, pixel);
          Eigen::Matrix2d information;
          information << g[0], g[1], g[1], g[2];
          information.diagonal().array() += 0.01 * config.min_texture;
          const Eigen::LLT<Eigen::Matrix2d> factor(information);
          if (factor.info() == Eigen::Success) {
            covariance += config.aperture_sigma * config.aperture_sigma * config.min_texture *
                          factor.solve(Eigen::Matrix2d::Identity());
          }
        }
      }
      pixel = next;
      if (depth_failure) {
        // Flow still supplies an identity: skip this depth observation as missing evidence.
        ++failures;
        continue;
      }
      const auto reference = estimate_.reference_points.at<cv::Vec3f>(v, u);
      const Eigen::Vector3d reference_point(reference[0], reference[1], reference[2]);
      if (!estimate_.legacy_mode && config.ambiguity_check) {
        // Exact poses predict where a static point must appear. If the image matches that
        // prediction about as well as the flow endpoint, it cannot distinguish the two.
        // Disagreements that cannot be tested (static prediction behind the camera or a patch
        // leaving the image) are ambiguous too: unverified flow is missing evidence.
        const Eigen::Vector3d c = sensor_T_world * reference_point;
        bool ambiguous = c.z() <= 0.0;
        if (!ambiguous) {
          const cv::Vec2f predicted(camera_config_.fx * c.x() / c.z() + camera_config_.cx,
                                    camera_config_.fy * c.y() / c.z() + camera_config_.cy);
          if (!(cv::norm(predicted - next) <= config.ambiguity_min_separation)) {
            const cv::Vec2f origin(u, v);
            const double flow_difference = patchDifference(reference_gray_, origin, gray, next);
            const double static_difference = patchDifference(reference_gray_, origin, gray, predicted);
            ambiguous = !std::isfinite(flow_difference) || !std::isfinite(static_difference) ||
                        static_difference <= flow_difference + config.photometric_tolerance;
          }
        }
        if (ambiguous) {
          flags |= kAmbiguousCorrespondence;
          ++failures;
          continue;
        }
      }
      const Eigen::Vector3d residual = world_T_sensor * sample.point - reference_point;
      estimate_.residual_norm.at<float>(v, u) = residual.norm();
      if (!estimate_.legacy_mode) {
        const Eigen::Matrix3d jacobian = world_T_sensor.rotation() * sample.jacobian;
        Eigen::Matrix3d covariance = jacobian.leftCols<2>() * pixel_covariances_[index] *
                                     jacobian.leftCols<2>().transpose() +
            sample.depth_variance * jacobian.col(2) * jacobian.col(2).transpose();
        covariance.diagonal().array() += config.position_sigma_floor * config.position_sigma_floor;
        covariance = (0.5 * (covariance + covariance.transpose())).eval();
        const double time = hydra::toSeconds(input.timestamp_ns - estimate_.reference_timestamp_ns);
        if (!temporal_models_[index].add(world_T_sensor * sample.point, covariance, time, quality)) {
          flags |= kNumericalFailure;
          ++failures;
          alive = 0;
          continue;
        }
        ++estimate_.observation_count.at<int>(v, u);
        if (config.debug_output) estimate_.measurement_sigma.at<float>(v, u) = std::sqrt(covariance.trace() / 3.0);
        scoreTemporal(v, u);
        continue;
      }
      double innovation_squared;
      Eigen::Matrix3d covariance;
      if (!uncertainties_[index].observe(residual, world_T_sensor.rotation() * sample.jacobian,
                               sample.depth_variance, innovation_squared, covariance)) {
        flags |= kNumericalFailure;
        ++failures;
        alive = 0;
        continue;
      }
      estimate_.residual_norm.at<float>(v, u) = residual.norm();
      estimate_.innovation_sigma.at<float>(v, u) = std::sqrt(covariance.trace() / 3.0);
      auto& error = estimate_.squared_error.at<double>(v, u);
      error += innovation_squared;
      const int count = ++estimate_.observation_count.at<int>(v, u);
      estimate_.accepted_count.at<int>(v, u) = count + 1;
      estimate_.temporal_span.at<float>(v, u) = hydra::toSeconds(input.timestamp_ns - estimate_.reference_timestamp_ns);
      if (count >= config.min_observations) {
        estimate_.probability.at<float>(v, u) =
            std::isinf(error)
                ? 1.f
                : boost::math::cdf(boost::math::chi_squared_distribution<double>(3.0 * count),
                                   error);
        estimate_.validity.at<uint8_t>(v, u) = static_cast<uint8_t>(
            estimate_.probability.at<float>(v, u) >= config.max_fit_cdf ? DynamicsValidity::kDynamic : DynamicsValidity::kStatic);
      }
    }
  }
  previous_texture_ = texture;
  if (!estimate_.legacy_mode) previous_tensor_ = structureTensor(gray, config.flow_window_size);
}

void OpticalFlowDynamics::scoreTemporal(int v, int u) {
  const auto fit = temporal_models_[v * estimate_.probability.cols + u].fit();
  estimate_.accepted_count.at<int>(v, u) = fit.observations;
  estimate_.temporal_span.at<float>(v, u) = fit.span;
  estimate_.reliability.at<float>(v, u) = fit.reliability;
  auto& state = estimate_.validity.at<uint8_t>(v, u);
  auto& score = estimate_.motion_evidence.at<float>(v, u);
  score = kNaN;
  state = static_cast<uint8_t>(DynamicsValidity::kInsufficient);
  if (config.debug_output) {
    estimate_.fitted_point.at<cv::Vec3f>(v, u) = cv::Vec3f(
        fit.static_point.x(), fit.static_point.y(), fit.static_point.z());
    estimate_.static_cost.at<float>(v, u) = fit.static_cost;
  }
  if (!fit.valid || fit.observations < config.min_fit_observations ||
      fit.span < config.min_temporal_span) return;
  estimate_.velocity.at<cv::Vec3f>(v, u) = cv::Vec3f(fit.velocity.x(), fit.velocity.y(), fit.velocity.z());
  const bool static_plausible = fit.static_cost <= fit_thresholds_[3 * fit.observations - 3];
  const bool dynamic_plausible = fit.dynamic_cost <= fit_thresholds_[3 * fit.observations - 6];
  const double improvement = std::max(0.0, fit.static_cost - fit.dynamic_cost);
  if (config.debug_output) {
    estimate_.dynamic_cost.at<float>(v, u) = fit.dynamic_cost;
    estimate_.delta_cost.at<float>(v, u) = improvement;
    estimate_.static_consistency.at<float>(v, u) = boost::math::cdf(
        boost::math::chi_squared_distribution<double>(3 * fit.observations - 3), fit.static_cost);
    estimate_.dynamic_consistency.at<float>(v, u) = boost::math::cdf(
        boost::math::chi_squared_distribution<double>(3 * fit.observations - 6), fit.dynamic_cost);
    const auto& c = fit.velocity_covariance;
    estimate_.velocity_covariance.at<cv::Vec<float, 6>>(v, u) =
        cv::Vec<float, 6>(c(0,0), c(0,1), c(0,2), c(1,1), c(1,2), c(2,2));
  }
  state = static_cast<uint8_t>(DynamicsValidity::kUnreliable);
  if (fit.reliability < config.min_reliability) return;
  const double speed = fit.velocity.norm();
  const Eigen::Vector3d direction = speed > 1.e-12 ? Eigen::Vector3d(fit.velocity / speed)
                                                 : Eigen::Vector3d::UnitX();
  const double velocity_sigma = std::sqrt(std::max(0.0, direction.dot(fit.velocity_covariance * direction)));
  const bool meaningful = speed * fit.span >= config.min_displacement &&
                          speed >= config.min_displacement_sigma * velocity_sigma;
  if (dynamic_plausible && meaningful && improvement > config.motion_penalty) {
    score = -std::expm1(-(improvement - config.motion_penalty) / config.motion_score_scale);
    state = static_cast<uint8_t>(DynamicsValidity::kDynamic);
  } else if (static_plausible) {
    score = 0.f;
    state = static_cast<uint8_t>(DynamicsValidity::kStatic);
  } else {
    estimate_.failure_flags.at<uint16_t>(v, u) |= kPoorModelFit;
  }
}

void OpticalFlowDynamics::regularizeSpatial() {
  applySpatialSupport(estimate_, reference_color_, reference_depth_, config);
}

void applySpatialSupport(DynamicsEstimate& estimate, const cv::Mat& reference_color,
                         const cv::Mat& reference_depth, const OpticalFlowDynamics::Config& config) {
  estimate.regularized_evidence = estimate.motion_evidence.clone();
  estimate.spatial_support.setTo(0);
  for (int v = 0; v < estimate.motion_evidence.rows; ++v) {
    for (int u = 0; u < estimate.motion_evidence.cols; ++u) {
      if (std::isfinite(estimate.motion_evidence.at<float>(v, u))) {
        estimate.spatial_support.at<uint8_t>(v, u) = 1;
      }
    }
  }
  if (!config.spatial_enabled) return;
  for (int v = 0; v < estimate.motion_evidence.rows; ++v) {
    for (int u = 0; u < estimate.motion_evidence.cols; ++u) {
      if (std::isfinite(estimate.motion_evidence.at<float>(v, u)) &&
          estimate.reliability.at<float>(v, u) >= config.spatial_min_reliability) continue;
      const auto point = estimate.reference_points.at<cv::Vec3f>(v, u);
      if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) continue;
      const float depth = reference_depth.at<float>(v, u);
      const auto color = reference_color.at<cv::Vec3b>(v, u);
      std::array<Eigen::Vector3d, 49> velocities;
      std::array<double, 49> scores, weights, values;
      int count = 0;
      for (int y = std::max(0, v - config.spatial_radius);
           y <= std::min(reference_depth.rows - 1, v + config.spatial_radius); ++y) {
        for (int x = std::max(0, u - config.spatial_radius);
             x <= std::min(reference_depth.cols - 1, u + config.spatial_radius); ++x) {
          if (x == u && y == v) continue;
          const float s = estimate.motion_evidence.at<float>(y, x);
          const float reliability = estimate.reliability.at<float>(y, x);
          if (!std::isfinite(s) || reliability < config.spatial_min_reliability) continue;
          const float neighbor_depth = reference_depth.at<float>(y, x);
          const double tolerance = config.max_depth_jump + config.depth_jump_relative * std::min(depth, neighbor_depth);
          if (!std::isfinite(neighbor_depth) || std::abs(depth - neighbor_depth) > tolerance) continue;
          const auto neighbor_color = reference_color.at<cv::Vec3b>(y, x);
          const cv::Vec3d difference = cv::Vec3d(color) - cv::Vec3d(neighbor_color);
          if (cv::norm(difference) > config.spatial_color_sigma) continue;
          // Check a connecting path as well: a narrow depth boundary can lie between endpoints.
          bool same_surface = true;
          const int steps = std::max(std::abs(x - u), std::abs(y - v));
          for (int i = 1; i < steps; ++i) {
            const int px = std::lround(u + (x - u) * static_cast<double>(i) / steps);
            const int py = std::lround(v + (y - v) * static_cast<double>(i) / steps);
            const float z = reference_depth.at<float>(py, px);
            if (!std::isfinite(z) || z <= 0.f || std::abs(z - depth) > tolerance) same_surface = false;
          }
          if (!same_surface) continue;
          const auto velocity = estimate.velocity.at<cv::Vec3f>(y, x);
          velocities[count] = Eigen::Vector3d(velocity[0], velocity[1], velocity[2]);
          if (!velocities[count].allFinite()) continue;
          scores[count] = s;
          weights[count] = reliability * std::exp(-difference.dot(difference) /
              (2.0 * config.spatial_color_sigma * config.spatial_color_sigma));
          ++count;
        }
      }
      if (count < config.spatial_min_neighbors) continue;
      Eigen::Vector3d median;
      for (int axis = 0; axis < 3; ++axis) {
        for (int i = 0; i < count; ++i) values[i] = velocities[i][axis];
        std::nth_element(values.begin(), values.begin() + count / 2, values.begin() + count);
        median[axis] = values[count / 2];
      }
      double weight = 0.0, weighted_score = 0.0, moving_weight = 0.0;
      int compatible = 0;
      for (int i = 0; i < count; ++i) {
        if ((velocities[i] - median).norm() > config.spatial_velocity_tolerance) continue;
        weight += weights[i];
        weighted_score += weights[i] * scores[i];
        if (scores[i] >= 0.5) moving_weight += weights[i];
        ++compatible;
      }
      if (compatible < config.spatial_min_neighbors || weight <= 0.0) continue;
      const double fraction = moving_weight / weight;
      if (fraction < config.spatial_motion_fraction && 1.0 - fraction < config.spatial_motion_fraction) continue;
      estimate.regularized_evidence.at<float>(v, u) = weighted_score / weight;
      estimate.spatial_support.at<uint8_t>(v, u) = 2;
    }
  }
}

DynamicsEstimate::ConstPtr OpticalFlowDynamics::snapshot() const {
  auto result = std::make_shared<DynamicsEstimate>(estimate_);
  // Frame buffers and sinks may retain snapshots while the next input is processed.
  result->reference_points = estimate_.reference_points.clone();
  result->probability = estimate_.probability.clone();
  result->squared_error = estimate_.squared_error.clone();
  result->observation_count = estimate_.observation_count.clone();
  result->failure_flags = estimate_.failure_flags.clone();
  result->failure_count = estimate_.failure_count.clone();
  result->track_alive = estimate_.track_alive.clone();
  result->forward_backward_error = estimate_.forward_backward_error.clone();
  result->residual_norm = estimate_.residual_norm.clone();
  result->innovation_sigma = estimate_.innovation_sigma.clone();
  result->motion_evidence = estimate_.motion_evidence.clone();
  result->regularized_evidence = estimate_.regularized_evidence.clone();
  result->reliability = estimate_.reliability.clone();
  result->validity = estimate_.validity.clone();
  result->spatial_support = estimate_.spatial_support.clone();
  result->accepted_count = estimate_.accepted_count.clone();
  result->temporal_span = estimate_.temporal_span.clone();
  result->velocity = estimate_.velocity.clone();
  result->fitted_point = estimate_.fitted_point.clone();
  result->static_cost = estimate_.static_cost.clone();
  result->dynamic_cost = estimate_.dynamic_cost.clone();
  result->delta_cost = estimate_.delta_cost.clone();
  result->static_consistency = estimate_.static_consistency.clone();
  result->dynamic_consistency = estimate_.dynamic_consistency.clone();
  result->velocity_covariance = estimate_.velocity_covariance.clone();
  result->measurement_sigma = estimate_.measurement_sigma.clone();
  return result;
}

}  // namespace khronos
