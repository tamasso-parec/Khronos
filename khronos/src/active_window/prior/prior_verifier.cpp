#include "khronos/active_window/prior/prior_verifier.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace khronos {

void declare_config(PriorVerifier::Config& config) {
  using namespace config;
  name("PriorVerifier");
  field(config.verbosity, "verbosity");
  field(config.max_range, "max_range", "m");
  field(config.point_radius, "point_radius", "m");
  field(config.max_splat_radius, "max_splat_radius", "px");
  field(config.min_tolerance, "min_tolerance", "m");
  field(config.tolerance_factor, "tolerance_factor");
  field(config.silhouette_erosion, "silhouette_erosion", "px");
  field(config.range_edge_threshold, "range_edge_threshold", "m");
  check(config.max_range, GT, 0.f, "max_range");
  check(config.point_radius, GE, 0.f, "point_radius");
  check(config.max_splat_radius, GE, 0, "max_splat_radius");
  check(config.min_tolerance, GT, 0.f, "min_tolerance");
  check(config.tolerance_factor, GE, 0.f, "tolerance_factor");
  check(config.silhouette_erosion, GE, 0, "silhouette_erosion");
}

PriorVerifier::PriorVerifier(const Config& config) : config(config::checkValid(config)) {}

void PriorVerifier::processInput(const ObjectPriorPtrs& priors, FrameData& data) const {
  Timer timer("active_window/prior_verification", data.input.timestamp_ns);
  const cv::Mat& range_image = data.input.range_image;
  if (range_image.empty() || range_image.type() != CV_32FC1) {
    LOG(WARNING) << "[Prior Verifier] Range image is missing or not 32FC1, skipping frame.";
    return;
  }

  data.prior_observations.clear();
  data.prior_id_image = cv::Mat::zeros(range_image.size(), CV_32SC1);
  data.prior_range_image = cv::Mat(range_image.size(),
                                   CV_32FC1,
                                   cv::Scalar(std::numeric_limits<float>::infinity()));
  data.prior_state_image = cv::Mat::zeros(range_image.size(), CV_8UC1);

  const cv::Mat splat_radii = renderPriors(priors, data);
  classifyPixels(computeExclusionMask(data, splat_radii), data);
  computeAbsentComponents(data);

  if (config.verbosity >= 3) {
    int num_present = 0;
    int num_absent = 0;
    for (const auto& obs : data.prior_observations) {
      const float absent_fraction = obs.absentFraction();
      num_present += absent_fraction >= 0.f && absent_fraction < 0.5f;
      num_absent += absent_fraction >= 0.5f;
    }
    LOG(INFO) << "[Prior Verifier] " << data.prior_observations.size()
              << " priors rendered: " << num_present << " mostly present, " << num_absent
              << " mostly absent.";
  }
}

cv::Mat PriorVerifier::renderPriors(const ObjectPriorPtrs& priors, FrameData& data) const {
  const auto& sensor = data.input.getSensor();
  const Eigen::Isometry3f sensor_T_world = data.input.getSensorPose().inverse().cast<float>();
  const Point sensor_position = data.input.getSensorPose().translation().cast<float>();
  const int rows = data.prior_id_image.rows;
  const int cols = data.prior_id_image.cols;
  cv::Mat splat_radii = cv::Mat::zeros(rows, cols, CV_8UC1);

  for (const ObjectPrior* prior_ptr : priors) {
    const ObjectPrior& prior = *prior_ptr;
    // Skip objects that are entirely out of range.
    const float half_diagonal = 0.5f * prior.bounding_box.dimensions.norm();
    if ((prior.bounding_box.world_P_center - sensor_position).norm() - half_diagonal >
        config.max_range) {
      continue;
    }

    int prior_index = -1;  // Index + 1 into prior_observations, assigned on first draw.
    for (const Point& vertex_W : prior.vertices) {
      const Point vertex_C = sensor_T_world * vertex_W;
      const float range = vertex_C.norm();
      if (range <= 0.f || range > config.max_range) {
        continue;
      }
      float u, v;
      if (!sensor.projectPointToImagePlane(vertex_C, u, v)) {
        continue;
      }

      // Estimate the splat radius in pixels by projecting a laterally offset point.
      int radius = 0;
      if (config.point_radius > 0.f) {
        Point lateral = vertex_C.cross(Point::UnitX());
        if (lateral.squaredNorm() < 1e-6f) {
          lateral = vertex_C.cross(Point::UnitY());
        }
        float u_offset, v_offset;
        const Point offset_C = vertex_C + config.point_radius * lateral.normalized();
        if (sensor.projectPointToImagePlane(offset_C, u_offset, v_offset)) {
          radius = static_cast<int>(std::ceil(std::hypot(u_offset - u, v_offset - v)));
        } else {
          radius = config.max_splat_radius;
        }
        radius = std::min(radius, config.max_splat_radius);
      }

      // Draw the splat with a depth test against all previously drawn priors.
      const int u_center = static_cast<int>(std::round(u));
      const int v_center = static_cast<int>(std::round(v));
      for (int y = std::max(0, v_center - radius); y <= std::min(rows - 1, v_center + radius); ++y) {
        for (int x = std::max(0, u_center - radius); x <= std::min(cols - 1, u_center + radius);
             ++x) {
          auto& expected = data.prior_range_image.at<FrameData::PriorRangeImageType>(y, x);
          if (range >= expected) {
            continue;
          }
          if (prior_index < 0) {
            auto& obs = data.prior_observations.emplace_back();
            obs.source = prior.source;
            obs.id = prior.id;
            obs.semantic_label = prior.semantic_label;
            prior_index = static_cast<int>(data.prior_observations.size());
          }
          expected = range;
          data.prior_id_image.at<FrameData::PriorIdImageType>(y, x) = prior_index;
          splat_radii.at<uint8_t>(y, x) = static_cast<uint8_t>(radius);
        }
      }
    }
  }
  return splat_radii;
}

cv::Mat PriorVerifier::computeExclusionMask(const FrameData& data,
                                            const cv::Mat& splat_radii) const {
  const cv::Mat& ids = data.prior_id_image;
  const cv::Mat& ranges = data.input.range_image;
  cv::Mat range_edges = cv::Mat::zeros(ids.size(), CV_8UC1);
  const bool check_edges = config.range_edge_threshold > 0.f;

  cv::Mat exclusion = cv::Mat::zeros(ids.size(), CV_8UC1);
  constexpr int kNeighbors[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  for (int v = 0; v < ids.rows; ++v) {
    for (int u = 0; u < ids.cols; ++u) {
      const int id = ids.at<FrameData::PriorIdImageType>(v, u);
      if (id == 0) {
        continue;
      }
      const float range = ranges.at<InputData::RangeType>(v, u);
      bool is_silhouette = false;
      for (const auto& [du, dv] : kNeighbors) {
        const int nu = u + du;
        const int nv = v + dv;
        if (nu < 0 || nv < 0 || nu >= ids.cols || nv >= ids.rows) {
          continue;
        }
        if (ids.at<FrameData::PriorIdImageType>(nv, nu) != id) {
          is_silhouette = true;
        }
        const float neighbor_range = ranges.at<InputData::RangeType>(nv, nu);
        if (check_edges && range > 0.f && neighbor_range > 0.f &&
            std::abs(range - neighbor_range) > config.range_edge_threshold) {
          range_edges.at<uint8_t>(v, u) = 1;
        }
      }

      // Splats extend the rendered silhouette beyond the object by their radius, so erode by the
      // splat radius plus the configured margin. The silhouette pixel itself counts as one pixel.
      if (!is_silhouette) {
        continue;
      }
      const int erosion = splat_radii.at<uint8_t>(v, u) + config.silhouette_erosion - 1;
      if (erosion < 0) {
        continue;
      }
      cv::rectangle(exclusion,
                    cv::Point(u - erosion, v - erosion),
                    cv::Point(u + erosion, v + erosion),
                    cv::Scalar(1),
                    cv::FILLED);
    }
  }

  if (check_edges) {
    cv::dilate(range_edges, range_edges, cv::Mat::ones(3, 3, CV_8UC1));
    exclusion |= range_edges;
  }
  return exclusion;
}

void PriorVerifier::classifyPixels(const cv::Mat& exclusion_mask, FrameData& data) const {
  const cv::Mat& ranges = data.input.range_image;
  const cv::Mat& labels = data.input.label_image;
  const bool use_labels =
      !labels.empty() && labels.size() == ranges.size() && labels.type() == CV_32SC1;

  for (int v = 0; v < ranges.rows; ++v) {
    for (int u = 0; u < ranges.cols; ++u) {
      const int id = data.prior_id_image.at<FrameData::PriorIdImageType>(v, u);
      if (id == 0) {
        continue;
      }
      PriorObservation& obs = data.prior_observations.at(id - 1);
      auto& state = data.prior_state_image.at<FrameData::PriorStateImageType>(v, u);
      ++obs.rendered;

      if (exclusion_mask.at<uint8_t>(v, u)) {
        state = static_cast<uint8_t>(PriorPixelState::kExcluded);
        ++obs.excluded;
        continue;
      }

      // Invalid returns carry no information, in particular not evidence of absence.
      const float measured = ranges.at<InputData::RangeType>(v, u);
      if (!(measured > 0.f) || !std::isfinite(measured) || !data.input.inRange(measured)) {
        state = static_cast<uint8_t>(PriorPixelState::kUnobserved);
        ++obs.unobserved;
        continue;
      }

      const float expected = data.prior_range_image.at<FrameData::PriorRangeImageType>(v, u);
      const float difference = measured - expected;
      const float tau = tolerance(expected);
      if (difference > tau) {
        state = static_cast<uint8_t>(PriorPixelState::kAbsent);
        ++obs.absent;
      } else if (difference < -tau) {
        state = static_cast<uint8_t>(PriorPixelState::kOccluded);
        ++obs.occluded;
      } else {
        state = static_cast<uint8_t>(PriorPixelState::kPresent);
        ++obs.present;
        if (use_labels) {
          ++obs.labels[labels.at<InputData::LabelType>(v, u)];
        }
      }
    }
  }
}

void PriorVerifier::computeAbsentComponents(FrameData& data) const {
  // Flood fill 4-connected patches of absent pixels that belong to the same prior.
  const cv::Mat& ids = data.prior_id_image;
  const cv::Mat& states = data.prior_state_image;
  const auto kAbsent = static_cast<FrameData::PriorStateImageType>(PriorPixelState::kAbsent);
  cv::Mat visited = cv::Mat::zeros(ids.size(), CV_8UC1);
  std::vector<Pixel> stack;
  for (int v = 0; v < ids.rows; ++v) {
    for (int u = 0; u < ids.cols; ++u) {
      if (visited.at<uint8_t>(v, u) || states.at<FrameData::PriorStateImageType>(v, u) != kAbsent) {
        continue;
      }
      const int id = ids.at<FrameData::PriorIdImageType>(v, u);
      int size = 0;
      stack.assign(1, Pixel(u, v));
      visited.at<uint8_t>(v, u) = 1;
      while (!stack.empty()) {
        const Pixel pixel = stack.back();
        stack.pop_back();
        ++size;
        for (const Pixel& neighbor : pixel.neighbors4()) {
          if (!neighbor.isInImage(ids) || visited.at<uint8_t>(neighbor.v, neighbor.u) ||
              states.at<FrameData::PriorStateImageType>(neighbor.v, neighbor.u) != kAbsent ||
              ids.at<FrameData::PriorIdImageType>(neighbor.v, neighbor.u) != id) {
            continue;
          }
          visited.at<uint8_t>(neighbor.v, neighbor.u) = 1;
          stack.push_back(neighbor);
        }
      }
      auto& obs = data.prior_observations.at(id - 1);
      obs.largest_absent_component = std::max(obs.largest_absent_component, size);
    }
  }
}

float PriorVerifier::tolerance(float range) const {
  return std::max(config.min_tolerance, config.tolerance_factor * range * range);
}

}  // namespace khronos
