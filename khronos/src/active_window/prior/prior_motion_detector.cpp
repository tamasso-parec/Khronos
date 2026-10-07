#include "khronos/active_window/prior/prior_motion_detector.h"

namespace khronos {

void declare_config(PriorMotionDetector::Config& config) {
  using namespace config;
  name("PriorMotionDetector");
  field(config.verbosity, "verbosity");
  field(config.min_informative_pixels, "min_informative_pixels", "px");
  field(config.min_baseline_frames, "min_baseline_frames", "frames");
  field(config.baseline_rate, "baseline_rate");
  field(config.min_absent_fraction, "min_absent_fraction");
  field(config.min_absent_increase, "min_absent_increase");
  field(config.max_baseline, "max_baseline");
  field(config.min_absent_component, "min_absent_component", "px");
  field(config.min_onset_frames, "min_onset_frames", "frames");
  field(config.enable_seeds, "enable_seeds");
  field(config.search_margin, "search_margin", "m");
  field(config.min_displaced_pixels, "min_displaced_pixels", "px");
  field(config.displacement_requires_free_space, "displacement_requires_free_space");
  field(config.require_label_match, "require_label_match");
  field(config.enable_suppression, "enable_suppression");
  field(config.max_suppression_absent_fraction, "max_suppression_absent_fraction");
  field(config.state_timeout, "state_timeout", "s");
  check(config.min_baseline_frames, GE, 0, "min_baseline_frames");
  checkInRange(config.baseline_rate, 0.f, 1.f, "baseline_rate");
  check(config.min_onset_frames, GT, 0, "min_onset_frames");
  check(config.search_margin, GE, 0.f, "search_margin");
  check(config.state_timeout, GT, 0.f, "state_timeout");
}

PriorMotionDetector::PriorMotionDetector(const Config& config)
    : config(config::checkValid(config)) {}

void PriorMotionDetector::processInput(const ObjectPriorPtrs& priors,
                                       FrameData& data,
                                       const TrackingLayer* tracking_layer) {
  Timer timer("active_window/prior_motion_detection", data.input.timestamp_ns);
  const TimeStamp stamp = data.input.timestamp_ns;
  if (data.prior_state_image.empty()) {
    return;
  }

  std::unordered_map<uint64_t, const ObjectPrior*> priors_by_key;
  for (const ObjectPrior* prior : priors) {
    priors_by_key.emplace(prior->key(), prior);
  }

  data.prior_seed_mask = cv::Mat::zeros(data.prior_state_image.size(), CV_8UC1);
  data.prior_suppress_mask = cv::Mat::zeros(data.prior_state_image.size(), CV_8UC1);

  int num_moving = 0;
  for (size_t i = 0; i < data.prior_observations.size(); ++i) {
    PriorObservation& obs = data.prior_observations[i];
    const uint64_t key = ObjectPrior::key(obs.source, obs.id);
    State& state = states_[key];
    state.last_seen = stamp;
    updateState(obs, state);
    if (!obs.moving) {
      continue;
    }
    ++num_moving;
    const auto prior = priors_by_key.find(key);
    if (config.enable_seeds && prior != priors_by_key.end()) {
      obs.seed_pixels = seedPixels(*prior->second, static_cast<int>(i) + 1, tracking_layer, data);
    }
  }

  if (config.enable_suppression) {
    suppressPixels(data);
  }

  // Forget priors that have not been seen for a while.
  const TimeStamp min_stamp = stamp - std::min(stamp, fromSeconds(config.state_timeout));
  for (auto it = states_.begin(); it != states_.end();) {
    it = it->second.last_seen < min_stamp ? states_.erase(it) : std::next(it);
  }

  CLOG(3) << "[Prior Motion Detector] " << data.prior_observations.size() << " priors, "
          << num_moving << " moving, " << cv::countNonZero(data.prior_seed_mask)
          << " seed pixels, " << cv::countNonZero(data.prior_suppress_mask)
          << " suppressing pixels.";
}

void PriorMotionDetector::updateState(PriorObservation& obs, State& state) const {
  if (obs.present + obs.absent < config.min_informative_pixels) {
    // Not enough evidence: keep the current state.
    obs.baseline = state.baseline_frames > 0 ? state.baseline : -1.f;
    obs.moving = state.onset_frames >= config.min_onset_frames;
    return;
  }

  const float absent_fraction = obs.absentFraction();
  const bool onset = state.baseline_frames >= config.min_baseline_frames &&
                     state.baseline <= config.max_baseline &&
                     absent_fraction >= config.min_absent_fraction &&
                     absent_fraction - state.baseline >= config.min_absent_increase &&
                     obs.largest_absent_component >= config.min_absent_component;
  if (onset) {
    // Freeze the baseline while the object is displaced.
    ++state.onset_frames;
  } else {
    state.onset_frames = 0;
    state.baseline = state.baseline_frames == 0
                         ? absent_fraction
                         : (1.f - config.baseline_rate) * state.baseline +
                               config.baseline_rate * absent_fraction;
    ++state.baseline_frames;
  }
  obs.baseline = state.baseline;
  obs.moving = state.onset_frames >= config.min_onset_frames;
}

int PriorMotionDetector::seedPixels(const ObjectPrior& prior,
                                    int prior_index,
                                    const TrackingLayer* tracking_layer,
                                    FrameData& data) const {
  const cv::Mat& labels = data.input.label_image;
  const bool has_labels = !labels.empty() && labels.size() == data.prior_seed_mask.size() &&
                          labels.type() == CV_32SC1;
  if (config.require_label_match && !has_labels) {
    LOG_FIRST_N(WARNING, 1) << "[Prior Motion Detector] No label image available, not seeding.";
    return 0;
  }

  const Eigen::Vector3f margin = Eigen::Vector3f::Constant(config.search_margin);
  const Eigen::Vector3f min = prior.bounding_box.world_P_center -
                              0.5f * prior.bounding_box.dimensions - margin;
  const Eigen::Vector3f max = prior.bounding_box.world_P_center +
                              0.5f * prior.bounding_box.dimensions + margin;
  const auto kPresent = static_cast<FrameData::PriorStateImageType>(PriorPixelState::kPresent);

  // Candidates are current pixels of the object's label near its prior. Displaced candidates are
  // those not explained by the prior itself.
  Pixels candidates;
  int num_displaced = 0;
  for (int v = 0; v < data.prior_seed_mask.rows; ++v) {
    for (int u = 0; u < data.prior_seed_mask.cols; ++u) {
      const float range = data.input.range_image.at<InputData::RangeType>(v, u);
      if (!(range > 0.f) || !data.input.inRange(range)) {
        continue;
      }
      if (config.require_label_match &&
          labels.at<InputData::LabelType>(v, u) != static_cast<int>(prior.semantic_label)) {
        continue;
      }
      // Pixels that match another static prior belong to that object.
      const int id = data.prior_id_image.at<FrameData::PriorIdImageType>(v, u);
      const bool present =
          data.prior_state_image.at<FrameData::PriorStateImageType>(v, u) == kPresent;
      if (id != 0 && id != prior_index && present && !data.prior_observations.at(id - 1).moving) {
        continue;
      }
      const auto& vertex = data.input.vertex_map.at<InputData::VertexType>(v, u);
      const Eigen::Vector3f point(vertex[0], vertex[1], vertex[2]);
      if ((point.array() < min.array()).any() || (point.array() > max.array()).any()) {
        continue;
      }
      candidates.emplace_back(u, v);
      if (id == prior_index && present) {
        continue;
      }
      if (config.displacement_requires_free_space && tracking_layer) {
        const Point p_W(vertex[0], vertex[1], vertex[2]);
        const auto block = tracking_layer->getBlockPtr(p_W);
        if (!block) {
          continue;
        }
        const VoxelIndex voxel_index = block->getVoxelIndex(p_W);
        if (!block->isValidVoxelIndex(voxel_index) || !block->getVoxel(voxel_index).ever_free) {
          continue;
        }
      }
      ++num_displaced;
    }
  }

  if (num_displaced < config.min_displaced_pixels) {
    return 0;
  }
  for (const Pixel& pixel : candidates) {
    data.prior_seed_mask.at<uint8_t>(pixel.v, pixel.u) = 1;
  }
  return static_cast<int>(candidates.size());
}

void PriorMotionDetector::suppressPixels(FrameData& data) const {
  const auto kPresent = static_cast<FrameData::PriorStateImageType>(PriorPixelState::kPresent);
  for (int v = 0; v < data.prior_suppress_mask.rows; ++v) {
    for (int u = 0; u < data.prior_suppress_mask.cols; ++u) {
      if (data.prior_state_image.at<FrameData::PriorStateImageType>(v, u) != kPresent ||
          data.prior_seed_mask.at<uint8_t>(v, u)) {
        continue;
      }
      const auto& obs =
          data.prior_observations.at(data.prior_id_image.at<FrameData::PriorIdImageType>(v, u) - 1);
      if (obs.moving || obs.baseline < 0.f ||
          obs.present + obs.absent < config.min_informative_pixels ||
          obs.absentFraction() > config.max_suppression_absent_fraction) {
        continue;
      }
      data.prior_suppress_mask.at<uint8_t>(v, u) = 1;
    }
  }
}

}  // namespace khronos
