#include "khronos/active_window/motion_detection/optical_flow_dynamics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <glog/logging.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace khronos {
namespace {

const auto registration =
    config::RegistrationWithConfig<DynamicsEstimator,
                                   OpticalFlowDynamics,
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

// Per-pixel depth-layer state, computed once per frame for the temporal neighborhood test.
enum DepthState : uint8_t { kValidDepth = 1, kRightBreak = 2, kDownBreak = 4 };

// A break marks a valid neighbor on another layer: |z - n| > max_depth_jump + relative*min(z, n).
cv::Mat depthLayerState(const cv::Mat& depth, const OpticalFlowDynamics::Config& config) {
  cv::Mat state = cv::Mat::zeros(depth.size(), CV_8UC1);
  const auto valid = [](double z) { return std::isfinite(z) && z > 0.0; };
  cv::parallel_for_(cv::Range(0, depth.rows), [&](const cv::Range& rows) {
    for (int v = rows.start; v < rows.end; ++v) {
      for (int u = 0; u < depth.cols; ++u) {
        const double z = depth.at<float>(v, u);
        if (!valid(z)) continue;
        uint8_t s = kValidDepth;
        const auto breaks = [&](double neighbor) {
          return valid(neighbor) &&
                 std::abs(z - neighbor) >
                     config.max_depth_jump + config.depth_jump_relative * std::min(z, neighbor);
        };
        if (u + 1 < depth.cols && breaks(depth.at<float>(v, u + 1))) s |= kRightBreak;
        if (v + 1 < depth.rows && breaks(depth.at<float>(v + 1, u))) s |= kDownBreak;
        state.at<uint8_t>(v, u) = s;
      }
    }
  });
  return state;
}

struct PointSample {
  Eigen::Vector3d point;
  Eigen::Matrix3d jacobian;
  double depth_variance = 0.0;
};

// depth_state (temporal mode only) enables the depth-layer neighborhood test.
uint16_t samplePoint(const cv::Mat& depth_image,
                     const cv::Mat& depth_state,
                     float min_range,
                     float max_range,
                     const hydra::Camera::Config& camera,
                     const cv::Vec2f& p,
                     const OpticalFlowDynamics::Config& config,
                     PointSample& sample,
                     int footprint_radius = 0) {
  if (!inside(depth_image, p)) {
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
      const double z = depth_image.at<float>(y + iy, x + ix);
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
  if (!depth_state.empty()) {
    // Scan order decides between invalid depth and a layer break: right neighbor, then down.
    const int radius = std::max(config.depth_neighborhood_radius, std::min(footprint_radius, 3));
    const int y0 = std::max(0, y - radius), y1 = std::min(depth_image.rows - 1, y + radius + 1);
    const int x0 = std::max(0, x - radius), x1 = std::min(depth_image.cols - 1, x + radius + 1);
    for (int iy = y0; iy <= y1; ++iy) {
      const uint8_t* row = depth_state.ptr<uint8_t>(iy);
      const uint8_t* below = iy < y1 ? depth_state.ptr<uint8_t>(iy + 1) : nullptr;
      for (int ix = x0; ix <= x1; ++ix) {
        const uint8_t s = row[ix];
        if (!(s & kValidDepth)) return kInvalidDepth;
        if (ix < x1) {
          if (!(row[ix + 1] & kValidDepth)) return kInvalidDepth;
          if (s & kRightBreak) return kDepthDiscontinuity;
        }
        if (below) {
          if (!(below[ix] & kValidDepth)) return kInvalidDepth;
          if (s & kDownBreak) return kDepthDiscontinuity;
        }
      }
    }
  } else if (max_depth - min_depth > config.max_depth_jump) {
    return kDepthDiscontinuity;
  }
  const double z = interpolate<float>(depth_image, p);
  sample.point =
      Eigen::Vector3d((p[0] - camera.cx) * z / camera.fx, (p[1] - camera.cy) * z / camera.fy, z);
  if (!sample.point.allFinite() ||
      !(sample.point.norm() >= min_range && sample.point.norm() <= max_range)) {
    return kOutOfRange;
  }
  const Eigen::Vector2d gradient(
      (1.0 - dy) * (depths[0][1] - depths[0][0]) + dy * (depths[1][1] - depths[1][0]),
      (1.0 - dx) * (depths[1][0] - depths[0][0]) + dx * (depths[1][1] - depths[0][1]));
  sample.jacobian =
      rgbdPointJacobian(p[0], p[1], z, camera.fx, camera.fy, camera.cx, camera.cy, gradient);
  return 0;
}

// 5x5 gray patch around an integer reference pixel, row-major; false if it leaves the image.
using Patch = std::array<float, 25>;
bool referencePatch(const cv::Mat& image, int u, int v, Patch& patch) {
  if (u < 2 || v < 2 || u + 2 >= image.cols - 1 || v + 2 >= image.rows - 1) return false;
  int i = 0;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) patch[i++] = image.at<uint8_t>(v + dy, u + dx);
  }
  return true;
}

// Mean absolute gray difference to a reference patch, sampling the current image bilinearly
// around p; NaN if that patch leaves the image.
double patchDifference(const Patch& reference, const cv::Mat& current, const cv::Vec2f& p) {
  const cv::Vec2f margin(2.f, 2.f);
  if (!inside(current, p - margin) || !inside(current, p + margin)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const auto sample = [&](float x, float y) {
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const float dx = x - x0, dy = y - y0;
    return (current.at<uint8_t>(y0, x0) * (1.f - dx) + current.at<uint8_t>(y0, x0 + 1) * dx) *
               (1.f - dy) +
           (current.at<uint8_t>(y0 + 1, x0) * (1.f - dx) +
            current.at<uint8_t>(y0 + 1, x0 + 1) * dx) *
               dy;
  };
  double sum = 0.0;
  int i = 0;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) sum += std::abs(reference[i++] - sample(p[0] + dx, p[1] + dy));
  }
  return sum / 25.0;
}

enum class Visibility { kUntestable, kPresent, kOccluded, kFreeSpace };

// Compare predicted optical-axis depth with the observed 3x3 depth around a pixel.
Visibility visibilityAt(const cv::Mat& depth,
                        const cv::Vec2f& p,
                        double predicted,
                        const OpticalFlowDynamics::Config& config) {
  if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || p[0] < 0.5f || p[1] < 0.5f ||
      p[0] >= depth.cols - 1.5f || p[1] >= depth.rows - 1.5f) {
    return Visibility::kUntestable;
  }
  const int x = std::lround(p[0]), y = std::lround(p[1]);
  const double tolerance = config.max_depth_jump + config.depth_jump_relative * predicted;
  int valid = 0, nearer = 0, farther = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const float z = depth.at<float>(y + dy, x + dx);
      if (!std::isfinite(z) || z <= 0.f) continue;
      ++valid;
      if (std::abs(z - predicted) <= tolerance) return Visibility::kPresent;
      (z < predicted ? nearer : farther)++;
    }
  }
  if (valid < 5) return Visibility::kUntestable;
  if (farther == valid) return Visibility::kFreeSpace;
  if (nearer == valid) return Visibility::kOccluded;
  return Visibility::kUntestable;
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

Eigen::Matrix3d pointCovariance(const PointSample& sample,
                                const Eigen::Matrix3d& rotation,
                                const Eigen::Matrix2d& pixel_covariance,
                                double position_sigma_floor) {
  const Eigen::Matrix3d jacobian = rotation * sample.jacobian;
  Eigen::Matrix3d covariance =
      jacobian.leftCols<2>() * pixel_covariance * jacobian.leftCols<2>().transpose() +
      sample.depth_variance * jacobian.col(2) * jacobian.col(2).transpose();
  covariance.diagonal().array() += position_sigma_floor * position_sigma_floor;
  return covariance;
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

OpticalFlowDynamics::OpticalFlowDynamics(const Config& config)
    : config(config::checkValid(config)) {
  // cdf(cost) <= max_fit_cdf  <=>  cost <= quantile; avoids two incomplete gammas per pixel.
  fit_thresholds_.assign(3 * this->config.window_size + 1, 0.0);
  for (size_t dof = 1; dof < fit_thresholds_.size(); ++dof) {
    fit_thresholds_[dof] = boost::math::quantile(boost::math::chi_squared_distribution<double>(dof),
                                                 this->config.max_fit_cdf);
  }
  if (this->config.flow_method != "farneback") {
    dis_ = cv::DISOpticalFlow::create(this->config.flow_method == "dis_fast"
                                          ? cv::DISOpticalFlow::PRESET_FAST
                                          : cv::DISOpticalFlow::PRESET_MEDIUM);
  }
}

void OpticalFlowDynamics::reset() {
  estimate_ = DynamicsEstimate();
  history_.clear();
  inputs_since_keyframe_ = 0;
  tracked_pixels_.release();
  reference_gray_.release();
  crf_unary_.release();
  crf_weights_.clear();
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

  const auto& c = camera->getConfig();
  const bool changed_camera =
      sensor_name_ != camera->name || (!history_.empty() && history_.back().gray.size() != size) ||
      c.fx != camera_config_.fx || c.fy != camera_config_.fy || c.cx != camera_config_.cx ||
      c.cy != camera_config_.cy || c.min_range != camera_config_.min_range ||
      c.max_range != camera_config_.max_range;
  const bool discontinuity =
      !history_.empty() &&
      (input.timestamp_ns <= history_.back().timestamp_ns ||
       hydra::toSeconds(input.timestamp_ns - history_.back().timestamp_ns) > config.max_frame_gap);
  const bool legacy_restart =
      config.scoring_mode == "legacy" &&
      (estimate_.num_frames >= config.window_size ||
       (input.timestamp_ns > estimate_.reference_timestamp_ns &&
        hydra::toSeconds(input.timestamp_ns - estimate_.reference_timestamp_ns) >
            config.max_window_duration));
  if (changed_camera || discontinuity || legacy_restart) reset();
  camera_config_ = c;
  sensor_name_ = camera->name;

  // Expire history before flow computation: never bridge a discarded temporal interval.
  if (config.scoring_mode == "temporal") {
    while (!history_.empty() &&
           (history_.size() >= static_cast<size_t>(config.window_size) ||
            hydra::toSeconds(input.timestamp_ns - history_.front().timestamp_ns) >
                config.max_window_duration)) {
      history_.pop_front();
    }
  }
  auto current = prepareFrame(input);
  if (config.scoring_mode == "legacy") {
    if (history_.empty()) {
      startWindow(input, current);
    } else {
      track(current,
            history_.back(),
            current.from_previous,
            current.to_previous,
            hydra::toSeconds(input.timestamp_ns - estimate_.reference_timestamp_ns));
      ++estimate_.num_frames;
    }
    history_.clear();
    history_.push_back(std::move(current));
  } else {
    history_.push_back(std::move(current));
    // The oldest cached frame has no usable predecessor inside the retained window.
    history_.front().to_previous.release();
    history_.front().from_previous.release();
    // Keyframes: every input extends the flow history; motion is detected at keyframes only.
    if (++inputs_since_keyframe_ < config.keyframe_interval) return nullptr;
    inputs_since_keyframe_ = 0;
    startWindow(input, history_.back());
    for (size_t i = history_.size() - 1; i > 0; --i) {
      const auto& source = history_[i];
      const auto& target = history_[i - 1];
      // Fit against increasing look-back time; negate velocity when exposing physical time.
      track(target,
            source,
            source.to_previous,
            source.from_previous,
            hydra::toSeconds(input.timestamp_ns - target.timestamp_ns));
    }
    estimate_.num_frames = history_.size();
    cv::parallel_for_(cv::Range(0, size.height), [&](const cv::Range& rows) {
      for (int v = rows.start; v < rows.end; ++v) {
        for (int u = 0; u < size.width; ++u) {
          if (temporal_models_[v * size.width + u].observations()) scoreTemporal(v, u);
        }
      }
    });
    if (config.visibility_check) applyFreeSpace();
    regularizeSpatial();
  }
  estimate_.latest_timestamp_ns = input.timestamp_ns;
  return snapshot();
}

OpticalFlowDynamics::Frame OpticalFlowDynamics::prepareFrame(const hydra::InputData& input) const {
  Frame frame;
  frame.timestamp_ns = input.timestamp_ns;
  frame.world_T_sensor = input.getSensorPose();
  frame.min_range = input.getSensor().min_range();
  frame.max_range = std::min(input.getSensor().max_range(), input.max_range);
  frame.depth = input.depth_image.clone();  // Callers may reuse their input buffers.
  cv::cvtColor(input.color_image, frame.gray, cv::COLOR_RGB2GRAY);
  cv::cornerMinEigenVal(frame.gray, frame.texture, 3);
  if (config.scoring_mode == "temporal") frame.depth_state = depthLayerState(frame.depth, config);
  if (config.scoring_mode == "temporal" && config.aperture_sigma > 0.0 &&
      config.min_texture > 0.0) {
    frame.tensor = structureTensor(frame.gray, config.flow_window_size);
  }
  if (!history_.empty()) {
    const cv::Mat& previous = history_.back().gray;
    if (dis_) {
      // DIS parallelizes internally, so the directions run one after the other. The output
      // flows are empty here, so DIS does not use them as initial estimates.
      dis_->calc(frame.gray, previous, frame.to_previous);
      dis_->calc(previous, frame.gray, frame.from_previous);
    } else {
      // Farneback runs on one thread; the two directions are independent.
      cv::parallel_for_(cv::Range(0, 2), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
          cv::calcOpticalFlowFarneback(i ? previous : frame.gray,
                                       i ? frame.gray : previous,
                                       i ? frame.from_previous : frame.to_previous,
                                       0.5,
                                       config.flow_levels,
                                       config.flow_window_size,
                                       config.flow_iterations,
                                       5,
                                       1.2,
                                       0);
        }
      });
    }
  }
  return frame;
}

void OpticalFlowDynamics::startWindow(const hydra::InputData& input, const Frame& frame) {
  const auto& gray = frame.gray;
  const auto& texture = frame.texture;
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
    for (auto* image : {&estimate_.static_cost,
                        &estimate_.dynamic_cost,
                        &estimate_.delta_cost,
                        &estimate_.static_consistency,
                        &estimate_.dynamic_consistency}) {
      *image = cv::Mat(gray.size(), CV_32FC1, cv::Scalar(kNaN));
    }
  }
  tracked_pixels_.create(gray.size(), CV_32FC2);
  uncertainties_.clear();
  if (estimate_.legacy_mode) {
    uncertainties_.resize(gray.total());
  } else {
    // Models are cleared per pixel below; covariances are only read for accepted references.
    temporal_models_.resize(gray.total());
    pixel_covariances_.resize(gray.total());
    if (config.spatial_enabled) {
      reference_color_ = input.color_image.clone();
      reference_depth_ = input.depth_image.clone();
    }
    reference_gray_ = gray;  // A fresh buffer per input; never written afterwards.
    crf_unary_ = cv::Mat::zeros(gray.size(), CV_32FC1);
    crf_weights_.clear();
    estimate_.free_space_count = cv::Mat::zeros(gray.size(), CV_32SC1);
    estimate_.visibility_tests = cv::Mat::zeros(gray.size(), CV_32SC1);
    // A contrario: smallest k whose expected number of chance detections per image is <= eps.
    const int tests = config.window_size - 1;
    const boost::math::binomial_distribution<double> chance(tests, config.free_space_false_rate);
    free_space_min_count_ = tests + 1;  // Unreachable unless some k qualifies.
    for (int k = 1; k <= tests; ++k) {
      if (gray.total() * boost::math::cdf(boost::math::complement(chance, k - 1)) <=
          config.free_space_max_false_alarms) {
        free_space_min_count_ = k;
        break;
      }
    }
  }
  const Eigen::Isometry3d world_T_sensor = input.getSensorPose();
  const Eigen::Isometry3d sensor_T_world = world_T_sensor.inverse();

  cv::parallel_for_(cv::Range(0, gray.rows), [&](const cv::Range& rows) {
    for (int v = rows.start; v < rows.end; ++v) {
      for (int u = 0; u < gray.cols; ++u) {
        const cv::Vec2f pixel(u, v);
        tracked_pixels_.at<cv::Vec2f>(v, u) = pixel;
        if (!estimate_.legacy_mode) temporal_models_[v * gray.cols + u].clear();
        auto& flags = estimate_.failure_flags.at<uint16_t>(v, u);
        const auto raw = input.vertex_map.at<cv::Vec3f>(v, u);
        Eigen::Vector3d point(raw[0], raw[1], raw[2]);
        const Eigen::Vector3d sensor_point =
            input.points_in_world_frame ? sensor_T_world * point : point;
        const float depth = input.depth_image.at<float>(v, u);
        if (!sensor_point.allFinite() || sensor_point.z() <= 0.0 || !std::isfinite(depth) ||
            depth <= 0.f || !input.inRange(sensor_point.norm())) {
          flags = kInvalidReference;
          continue;
        }
        if (!input.points_in_world_frame) {
          point = world_T_sensor * point;
        }
        estimate_.reference_points.at<cv::Vec3f>(v, u) = cv::Vec3f(point.x(), point.y(), point.z());
        PointSample sample;
        flags = samplePoint(frame.depth,
                            frame.depth_state,
                            frame.min_range,
                            frame.max_range,
                            camera_config_,
                            pixel,
                            config,
                            sample);
        if (texture.at<float>(v, u) < config.min_texture) {
          flags |= kLowTexture;
        }
        if (flags & ~kLowTexture || (estimate_.legacy_mode && flags)) {
          continue;
        }
        const int index = v * gray.cols + u;
        if (!estimate_.legacy_mode) {
          const double texture_ratio =
              config.min_texture > 0.0
                  ? std::max(0.0, 1.0 - texture.at<float>(v, u) / config.min_texture)
                  : 0.0;
          const double inflation = 1.0 + config.texture_variance_scale * texture_ratio;
          pixel_covariances_[index] = config.reference_pixel_sigma * config.reference_pixel_sigma *
                                      inflation * Eigen::Matrix2d::Identity();
          Eigen::Matrix3d covariance = pointCovariance(sample,
                                                       world_T_sensor.rotation(),
                                                       pixel_covariances_[index],
                                                       config.position_sigma_floor);
          if (!temporal_models_[index].add(point, covariance, 0.0, 1.0 / std::sqrt(inflation))) {
            flags |= kNumericalFailure;
            continue;
          }
        } else {
          uncertainties_[index].initialize(
              world_T_sensor.rotation() * sample.jacobian,
              config.reference_pixel_sigma * config.reference_pixel_sigma,
              sample.depth_variance);
        }
        if (estimate_.legacy_mode) {
          estimate_.accepted_count.at<int>(v, u) = 1;
          estimate_.reliability.at<float>(v, u) = 1.f;
          estimate_.validity.at<uint8_t>(v, u) =
              static_cast<uint8_t>(DynamicsValidity::kInsufficient);
        }
        estimate_.track_alive.at<uint8_t>(v, u) = 255;
      }
    }
  });
}

void OpticalFlowDynamics::track(const Frame& target,
                                const Frame& source,
                                const cv::Mat& forward,
                                const cv::Mat& backward,
                                double time) {
  const auto& gray = target.gray;
  const auto& texture = target.texture;
  const Eigen::Isometry3d& world_T_sensor = target.world_T_sensor;
  const Eigen::Isometry3d sensor_T_world = world_T_sensor.inverse();

  cv::parallel_for_(cv::Range(0, gray.rows), [&](const cv::Range& rows) {
    for (int v = rows.start; v < rows.end; ++v) {
      for (int u = 0; u < gray.cols; ++u) {
        // Track-free visibility test of the static hypothesis, for every valid reference pixel.
        Visibility visibility = Visibility::kUntestable;
        const auto reference_raw = estimate_.reference_points.at<cv::Vec3f>(v, u);
        const Eigen::Vector3d reference_point(reference_raw[0], reference_raw[1], reference_raw[2]);
        Eigen::Vector3d static_point = Eigen::Vector3d::Zero();
        cv::Vec2f predicted(kNaN, kNaN);
        if (!estimate_.legacy_mode && (config.visibility_check || config.ambiguity_check) &&
            std::isfinite(reference_raw[0])) {
          static_point = sensor_T_world * reference_point;
          if (static_point.z() > 0.0) {
            predicted = cv::Vec2f(
                camera_config_.fx * static_point.x() / static_point.z() + camera_config_.cx,
                camera_config_.fy * static_point.y() / static_point.z() + camera_config_.cy);
          }
        }
        if (!estimate_.legacy_mode && config.visibility_check && static_point.z() > 0.0) {
          visibility = visibilityAt(target.depth, predicted, static_point.z(), config);
          if (visibility != Visibility::kUntestable) ++estimate_.visibility_tests.at<int>(v, u);
          if (visibility == Visibility::kFreeSpace) ++estimate_.free_space_count.at<int>(v, u);
        }
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
          flags |=
              std::isfinite(next[0]) && std::isfinite(next[1]) ? kOutsideImage : kNonFiniteFlow;
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
        const double local_texture =
            std::min(interpolate<float>(source.texture, pixel), interpolate<float>(texture, next));
        if (local_texture < config.min_texture) {
          flow_failure |= kLowTexture;
        }
        PointSample sample;
        int footprint = 0;
        const int index = v * gray.cols + u;
        if (!estimate_.legacy_mode) {
          const double variance = pixel_covariances_[index].trace() * 0.5;
          footprint =
              std::isfinite(variance)
                  ? static_cast<int>(std::min(3.0, std::ceil(std::sqrt(std::max(0.0, variance)))))
                  : 3;
        }
        const uint16_t depth_failure = samplePoint(target.depth,
                                                   target.depth_state,
                                                   target.min_range,
                                                   target.max_range,
                                                   camera_config_,
                                                   next,
                                                   config,
                                                   sample,
                                                   footprint);
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
          const double texture_ratio = config.min_texture > 0.0
                                           ? std::max(0.0, 1.0 - local_texture / config.min_texture)
                                           : 0.0;
          const double inflation =
              1.0 + config.texture_variance_scale * texture_ratio +
              config.fb_variance_scale *
                  std::pow(cycle_error / config.max_forward_backward_error, 2.0);
          quality = 1.0 / std::sqrt(inflation);
          auto& covariance = pixel_covariances_[index];
          covariance = (transition * covariance * transition.transpose()).eval();
          covariance.diagonal().array() += config.flow_sigma * config.flow_sigma * inflation;
          if (config.aperture_sigma > 0.0 && config.min_texture > 0.0) {
            // Lucas-Kanade-style aperture model: weakly textured directions are poorly observed.
            const auto g = interpolate<cv::Vec3f>(source.tensor, pixel);
            Eigen::Matrix2d information;
            information << g[0], g[1], g[1], g[2];
            information.diagonal().array() += 0.01 * config.min_texture;
            const Eigen::LLT<Eigen::Matrix2d> factor(information);
            if (factor.info() == Eigen::Success) {
              covariance += config.aperture_sigma * config.aperture_sigma * config.min_texture *
                            choleskyInverse(factor);
            }
          }
        }
        pixel = next;
        if (depth_failure) {
          // Flow still supplies an identity: skip this depth observation as missing evidence.
          ++failures;
          continue;
        }
        if (visibility == Visibility::kOccluded) {
          // The static point is hidden: a flow match cannot observe it, so this is missing
          // evidence.
          flags |= kOccluded;
          ++failures;
          continue;
        }
        if (!estimate_.legacy_mode && config.ambiguity_check) {
          // Exact poses predict where a static point must appear. If the image matches that
          // prediction about as well as the flow endpoint, it cannot distinguish the two.
          // Disagreements that cannot be tested (static prediction behind the camera or a patch
          // leaving the image) are ambiguous too: unverified flow is missing evidence.
          bool ambiguous = static_point.z() <= 0.0;
          if (!ambiguous) {
            if (!(cv::norm(predicted - next) <= config.ambiguity_min_separation)) {
              Patch patch;
              ambiguous = !referencePatch(reference_gray_, u, v, patch);
              if (!ambiguous) {
                const double flow_difference = patchDifference(patch, gray, next);
                const double static_difference = patchDifference(patch, gray, predicted);
                ambiguous = !std::isfinite(flow_difference) ||
                            !std::isfinite(static_difference) ||
                            static_difference <= flow_difference + config.photometric_tolerance;
              }
            }
          }
          if (ambiguous) {
            flags |= kAmbiguousCorrespondence;
            ++failures;
            continue;
          }
        }
        const Eigen::Vector3d observed_point = world_T_sensor * sample.point;
        const Eigen::Vector3d residual = observed_point - reference_point;
        estimate_.residual_norm.at<float>(v, u) = residual.norm();
        if (!estimate_.legacy_mode) {
          Eigen::Matrix3d covariance = pointCovariance(sample,
                                                       world_T_sensor.rotation(),
                                                       pixel_covariances_[index],
                                                       config.position_sigma_floor);
          covariance = (0.5 * (covariance + covariance.transpose())).eval();
          if (!temporal_models_[index].add(observed_point, covariance, time, quality)) {
            flags |= kNumericalFailure;
            ++failures;
            alive = 0;
            continue;
          }
          ++estimate_.observation_count.at<int>(v, u);
          if (config.debug_output)
            estimate_.measurement_sigma.at<float>(v, u) = std::sqrt(covariance.trace() / 3.0);
          continue;
        }
        double innovation_squared;
        Eigen::Matrix3d covariance;
        if (!uncertainties_[index].observe(residual,
                                           world_T_sensor.rotation() * sample.jacobian,
                                           sample.depth_variance,
                                           innovation_squared,
                                           covariance)) {
          flags |= kNumericalFailure;
          ++failures;
          alive = 0;
          continue;
        }
        estimate_.innovation_sigma.at<float>(v, u) = std::sqrt(covariance.trace() / 3.0);
        auto& error = estimate_.squared_error.at<double>(v, u);
        error += innovation_squared;
        const int count = ++estimate_.observation_count.at<int>(v, u);
        estimate_.accepted_count.at<int>(v, u) = count + 1;
        estimate_.temporal_span.at<float>(v, u) = time;
        if (count >= config.min_observations) {
          estimate_.probability.at<float>(v, u) =
              std::isinf(error)
                  ? 1.f
                  : boost::math::cdf(boost::math::chi_squared_distribution<double>(3.0 * count),
                                     error);
          estimate_.validity.at<uint8_t>(v, u) =
              static_cast<uint8_t>(estimate_.probability.at<float>(v, u) >= config.max_fit_cdf
                                       ? DynamicsValidity::kDynamic
                                       : DynamicsValidity::kStatic);
        }
      }
    }
  });
}

void OpticalFlowDynamics::applyFreeSpace() {
  // Apply historical visibility evidence after the complete current-reference flow fit.
  for (int v = 0; v < estimate_.free_space_count.rows; ++v) {
    for (int u = 0; u < estimate_.free_space_count.cols; ++u) {
      const int violations = estimate_.free_space_count.at<int>(v, u);
      if (violations < free_space_min_count_) continue;
      // Flow data that neither model explains stays unknown: free space overrides static or
      // missing flow evidence, not contradictory evidence.
      if (estimate_.validity.at<uint8_t>(v, u) ==
              static_cast<uint8_t>(DynamicsValidity::kUnreliable) &&
          (estimate_.failure_flags.at<uint16_t>(v, u) & kPoorModelFit)) {
        continue;
      }
      auto& score = estimate_.motion_evidence.at<float>(v, u);
      const float visible = -std::expm1(-static_cast<float>(violations));  // Uncalibrated.
      score = std::isfinite(score) ? std::max(score, visible) : visible;
      estimate_.validity.at<uint8_t>(v, u) = static_cast<uint8_t>(DynamicsValidity::kDynamic);
      auto& unary = crf_unary_.at<float>(v, u);
      unary = std::max<float>(
          unary, std::min(config.crf_max_unary, violations * config.free_space_log_odds));
    }
  }
}

void OpticalFlowDynamics::scoreTemporal(int v, int u) {
  const auto& model = temporal_models_[v * estimate_.motion_evidence.cols + u];
  estimate_.accepted_count.at<int>(v, u) = model.observations();
  estimate_.temporal_span.at<float>(v, u) = model.span();
  estimate_.reliability.at<float>(v, u) = model.reliability();
  auto& state = estimate_.validity.at<uint8_t>(v, u);
  auto& score = estimate_.motion_evidence.at<float>(v, u);
  auto& unary = crf_unary_.at<float>(v, u);
  unary = 0.f;
  score = kNaN;
  state = static_cast<uint8_t>(DynamicsValidity::kInsufficient);
  // Partial windows only need counters unless static-fit diagnostics were requested.
  if (!config.debug_output && (model.observations() < config.min_fit_observations ||
                               model.span() < config.min_temporal_span))
    return;
  const auto fit = model.fit();
  if (config.debug_output) {
    estimate_.fitted_point.at<cv::Vec3f>(v, u) =
        cv::Vec3f(fit.static_point.x(), fit.static_point.y(), fit.static_point.z());
    estimate_.static_cost.at<float>(v, u) = fit.static_cost;
  }
  if (!fit.valid || fit.observations < config.min_fit_observations ||
      fit.span < config.min_temporal_span)
    return;
  estimate_.velocity.at<cv::Vec3f>(v, u) =
      cv::Vec3f(-fit.velocity.x(), -fit.velocity.y(), -fit.velocity.z());
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
        cv::Vec<float, 6>(c(0, 0), c(0, 1), c(0, 2), c(1, 1), c(1, 2), c(2, 2));
  }
  state = static_cast<uint8_t>(DynamicsValidity::kUnreliable);
  if (fit.reliability < config.min_reliability) return;
  const double speed = fit.velocity.norm();
  const Eigen::Vector3d direction =
      speed > 1.e-12 ? Eigen::Vector3d(fit.velocity / speed) : Eigen::Vector3d::UnitX();
  const double velocity_sigma =
      std::sqrt(std::max(0.0, direction.dot(fit.velocity_covariance * direction)));
  const bool meaningful = speed * fit.span >= config.min_displacement &&
                          speed >= config.min_displacement_sigma * velocity_sigma;
  if (dynamic_plausible && meaningful && improvement > config.motion_penalty) {
    score = -std::expm1(-(improvement - config.motion_penalty) / config.motion_score_scale);
    state = static_cast<uint8_t>(DynamicsValidity::kDynamic);
    // Half the penalized cost difference is a log-likelihood ratio under ideal Gaussian
    // assumptions; it is capped because correlated real errors make it overconfident.
    unary = fit.reliability *
            std::clamp(0.5 * (improvement - config.motion_penalty), 0.5, config.crf_max_unary);
  } else if (static_plausible) {
    score = 0.f;
    state = static_cast<uint8_t>(DynamicsValidity::kStatic);
    unary = fit.reliability *
            std::clamp(0.5 * (improvement - config.motion_penalty), -config.crf_max_unary, -0.5);
  } else {
    estimate_.failure_flags.at<uint16_t>(v, u) |= kPoorModelFit;
  }
}

void OpticalFlowDynamics::regularizeSpatial() {
  if (!config.spatial_enabled || config.spatial_method == "support") {
    applySpatialSupport(estimate_, reference_color_, reference_depth_, config);
    return;
  }
  // Zero unaries stay at q=0.5: without any confidence, no pixel can receive support.
  if (config.crf_min_support > 0.0 && cv::countNonZero(crf_unary_) == 0) {
    estimate_.regularized_evidence.setTo(kNaN);
    estimate_.spatial_support.setTo(0);
    return;
  }
  if (crf_weights_.empty()) crf_weights_ = crfWeights(reference_color_, reference_depth_, config);
  applyCrfSupport(estimate_, crf_unary_, crf_weights_, config);
}

DynamicsEstimate::ConstPtr OpticalFlowDynamics::snapshot() {
  // Temporal mode allocates every output again at the next keyframe, so the snapshot can own
  // the current buffers.
  if (!estimate_.legacy_mode) return std::make_shared<const DynamicsEstimate>(std::move(estimate_));
  auto result = std::make_shared<DynamicsEstimate>(estimate_);
  // Legacy mode updates its window in place; frame buffers and sinks may retain snapshots.
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
  result->free_space_count = estimate_.free_space_count.clone();
  result->visibility_tests = estimate_.visibility_tests.clone();
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
