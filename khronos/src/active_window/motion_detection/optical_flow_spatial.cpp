#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <opencv2/core/utility.hpp>

#include "khronos/active_window/motion_detection/optical_flow_dynamics.h"

namespace khronos {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// 8 directions at 1, 3 and 6 px: reach without a dense window; iterations extend it.
const std::array<cv::Point, kCrfStencilSize>& crfStencil() {
  static const auto stencil = [] {
    std::array<cv::Point, kCrfStencilSize> offsets;
    int i = 0;
    for (const int d : {1, 3, 6}) {
      for (const auto& dir : {cv::Point(1, 0),
                              cv::Point(-1, 0),
                              cv::Point(0, 1),
                              cv::Point(0, -1),
                              cv::Point(1, 1),
                              cv::Point(-1, -1),
                              cv::Point(1, -1),
                              cv::Point(-1, 1)}) {
        offsets[i++] = dir * d;
      }
    }
    return offsets;
  }();
  return stencil;
}

double crfNormalizer(const OpticalFlowDynamics::Config& config) {
  double total = 0.0;
  for (const auto& o : crfStencil()) {
    total += std::exp(-(o.x * o.x + o.y * o.y) /
                      (2.0 * config.crf_spatial_sigma * config.crf_spatial_sigma));
  }
  return total;
}

}  // namespace

std::vector<float> crfWeights(const cv::Mat& reference_color,
                              const cv::Mat& reference_depth,
                              const OpticalFlowDynamics::Config& config) {
  const int rows = reference_depth.rows, cols = reference_depth.cols;
  std::vector<float> weights(static_cast<size_t>(rows) * cols * kCrfStencilSize, 0.f);
  const auto& stencil = crfStencil();
  std::array<double, kCrfStencilSize> spatial_exponents;
  for (int k = 0; k < kCrfStencilSize; ++k) {
    spatial_exponents[k] = -(stencil[k].x * stencil[k].x + stencil[k].y * stencil[k].y) /
                           (2.0 * config.crf_spatial_sigma * config.crf_spatial_sigma);
  }
  const auto valid = [](float z) { return std::isfinite(z) && z > 0.f; };
  // Even stencil indices step along (1,0), (0,1), (1,1), (1,-1); each odd index is the opposite
  // of the preceding even one. links[j] marks a pixel whose unit step along direction j stays on
  // the same depth layer, so a segment is connected iff all its unit steps are.
  std::array<cv::Mat, 4> links;
  for (int j = 0; j < 4; ++j) {
    const cv::Point step = stencil[2 * j];
    links[j] = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
      for (int v = range.start; v < range.end; ++v) {
        for (int u = 0; u < cols; ++u) {
          const int x = u + step.x, y = v + step.y;
          if (x < 0 || y < 0 || x >= cols || y >= rows) continue;
          const float a = reference_depth.at<float>(y, x), b = reference_depth.at<float>(v, u);
          links[j].at<uint8_t>(v, u) =
              valid(a) && valid(b) &&
              std::abs(a - b) <= config.max_depth_jump + config.depth_jump_relative * std::min(a, b);
        }
      }
    });
  }
  // Weights are symmetric, so each pair is computed once from its even-index end. Every slot has a
  // single writer, so rows can write into neighboring rows' slots in parallel.
  cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
    for (int v = range.start; v < range.end; ++v) {
      for (int u = 0; u < cols; ++u) {
        if (!valid(reference_depth.at<float>(v, u))) continue;
        const auto color = reference_color.at<cv::Vec3b>(v, u);
        for (int k = 0; k < kCrfStencilSize; k += 2) {
          const int x = u + stencil[k].x, y = v + stencil[k].y;
          if (x < 0 || y < 0 || x >= cols || y >= rows) continue;
          const int steps = std::max(std::abs(stencil[k].x), std::abs(stencil[k].y));
          const cv::Point step = stencil[k] / steps;
          const cv::Mat& link = links[(k % 8) / 2];
          bool connected = true;
          for (int i = 0; i < steps && connected; ++i) {
            connected = link.at<uint8_t>(v + step.y * i, u + step.x * i);
          }
          if (!connected) continue;
          const cv::Vec3d difference =
              cv::Vec3d(color) - cv::Vec3d(reference_color.at<cv::Vec3b>(y, x));
          const float weight = std::exp(spatial_exponents[k] -
                                        difference.dot(difference) /
                                            (2.0 * config.spatial_color_sigma * config.spatial_color_sigma));
          weights[(static_cast<size_t>(v) * cols + u) * kCrfStencilSize + k] = weight;
          weights[(static_cast<size_t>(y) * cols + x) * kCrfStencilSize + (k ^ 1)] = weight;
        }
      }
    }
  });
  return weights;
}

void applyCrfSupport(DynamicsEstimate& estimate,
                     const cv::Mat& unary,
                     const std::vector<float>& weights,
                     const OpticalFlowDynamics::Config& config) {
  const int rows = unary.rows, cols = unary.cols;
  const auto& stencil = crfStencil();
  const double normalizer = crfNormalizer(config);
  const double scale = config.crf_pairwise_weight / normalizer;
  const auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
  cv::Mat q(unary.size(), CV_32FC1), next(unary.size(), CV_32FC1);
  cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
    for (int v = range.start; v < range.end; ++v) {
      for (int u = 0; u < cols; ++u) q.at<float>(v, u) = sigmoid(unary.at<float>(v, u));
    }
  });
  // Synchronous mean-field updates of a binary Potts model with log-odds unaries.
  const auto message = [&](const cv::Mat& marginal, int v, int u, bool support_only) {
    const float* w = &weights[(static_cast<size_t>(v) * cols + u) * kCrfStencilSize];
    double sum = 0.0;
    for (int k = 0; k < kCrfStencilSize; ++k) {
      if (w[k] <= 0.f) continue;
      const double vote = 2.0 * marginal.at<float>(v + stencil[k].y, u + stencil[k].x) - 1.0;
      sum += w[k] * (support_only ? std::abs(vote) : vote);
    }
    return sum;
  };
  for (int iteration = 0; iteration < config.crf_iterations; ++iteration) {
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
      for (int v = range.start; v < range.end; ++v) {
        for (int u = 0; u < cols; ++u) {
          next.at<float>(v, u) = sigmoid(unary.at<float>(v, u) + scale * message(q, v, u, false));
        }
      }
    });
    std::swap(q, next);
  }
  cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
    for (int v = range.start; v < range.end; ++v) {
      for (int u = 0; u < cols; ++u) {
        auto& output = estimate.regularized_evidence.at<float>(v, u);
        auto& source = estimate.spatial_support.at<uint8_t>(v, u);
        output = kNaN;
        source = 0;
        const auto point = estimate.reference_points.at<cv::Vec3f>(v, u);
        if (!std::isfinite(point[0])) continue;
        if (unary.at<float>(v, u) != 0.f) {
          source = 1;
        } else if (message(q, v, u, true) / normalizer >= config.crf_min_support) {
          source = 2;  // Inferred from connected neighbors; reliability is not upgraded.
        } else {
          continue;
        }
        output = q.at<float>(v, u);
      }
    }
  });
}

void applySpatialSupport(DynamicsEstimate& estimate,
                         const cv::Mat& reference_color,
                         const cv::Mat& reference_depth,
                         const OpticalFlowDynamics::Config& config) {
  estimate.motion_evidence.copyTo(estimate.regularized_evidence);
  estimate.spatial_support.setTo(0);
  cv::parallel_for_(cv::Range(0, estimate.motion_evidence.rows), [&](const cv::Range& range) {
    for (int v = range.start; v < range.end; ++v) {
      for (int u = 0; u < estimate.motion_evidence.cols; ++u) {
        if (std::isfinite(estimate.motion_evidence.at<float>(v, u))) {
          estimate.spatial_support.at<uint8_t>(v, u) = 1;
        }
      }
    }
  });
  if (!config.spatial_enabled) return;
  cv::parallel_for_(cv::Range(0, estimate.motion_evidence.rows), [&](const cv::Range& range) {
    for (int v = range.start; v < range.end; ++v) {
      for (int u = 0; u < estimate.motion_evidence.cols; ++u) {
        if (std::isfinite(estimate.motion_evidence.at<float>(v, u)) &&
            estimate.reliability.at<float>(v, u) >= config.spatial_min_reliability)
          continue;
        const auto point = estimate.reference_points.at<cv::Vec3f>(v, u);
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]))
          continue;
        const float depth = reference_depth.at<float>(v, u);
        const auto color = reference_color.at<cv::Vec3b>(v, u);
        std::array<Eigen::Vector3d, 49> velocities;
        std::array<double, 49> scores, weights, values;
        int count = 0;
        for (int y = std::max(0, v - config.spatial_radius);
             y <= std::min(reference_depth.rows - 1, v + config.spatial_radius);
             ++y) {
          for (int x = std::max(0, u - config.spatial_radius);
               x <= std::min(reference_depth.cols - 1, u + config.spatial_radius);
               ++x) {
            if (x == u && y == v) continue;
            const float s = estimate.motion_evidence.at<float>(y, x);
            const float reliability = estimate.reliability.at<float>(y, x);
            if (!std::isfinite(s) || reliability < config.spatial_min_reliability) continue;
            const float neighbor_depth = reference_depth.at<float>(y, x);
            const double tolerance = config.max_depth_jump +
                                     config.depth_jump_relative * std::min(depth, neighbor_depth);
            if (!std::isfinite(neighbor_depth) || std::abs(depth - neighbor_depth) > tolerance)
              continue;
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
              if (!std::isfinite(z) || z <= 0.f || std::abs(z - depth) > tolerance)
                same_surface = false;
            }
            if (!same_surface) continue;
            const auto velocity = estimate.velocity.at<cv::Vec3f>(y, x);
            velocities[count] = Eigen::Vector3d(velocity[0], velocity[1], velocity[2]);
            if (!velocities[count].allFinite()) continue;
            scores[count] = s;
            weights[count] =
                reliability *
                std::exp(-difference.dot(difference) /
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
        if (fraction < config.spatial_motion_fraction &&
            1.0 - fraction < config.spatial_motion_fraction)
          continue;
        estimate.regularized_evidence.at<float>(v, u) = weighted_score / weight;
        estimate.spatial_support.at<uint8_t>(v, u) = 2;
      }
    }
  });
}

}  // namespace khronos
