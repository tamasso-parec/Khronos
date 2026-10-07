#include "khronos/active_window/prior/motion_evaluation_logger.h"

#include <filesystem>

#include <hydra/common/global_info.h>

namespace khronos {

void declare_config(MotionEvaluationLogger::Config& config) {
  using namespace config;
  name("MotionEvaluationLogger");
  field(config.output_file, "output_file");
  field(config.max_range, "max_range", "m");
  field(config.min_cluster_precision, "min_cluster_precision");
  checkCondition(!config.output_file.empty(), "param 'output_file' must not be empty");
  checkInRange(config.min_cluster_precision, 0.f, 1.f, "min_cluster_precision");
}

MotionEvaluationLogger::MotionEvaluationLogger(const Config& config)
    : config(config::checkValid(config)) {
  const std::filesystem::path path(config.output_file);
  std::error_code error;
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path(), error);
  }
  file_.open(path);
  if (!file_.is_open()) {
    LOG(ERROR) << "[Motion Evaluation Logger] Could not open '" << config.output_file << "'.";
    return;
  }
  file_ << "timestamp_ns,gt_pixels,detected_pixels,tp_pixels,clusters,tp_clusters,"
           "prior_seed_pixels,prior_seed_gt_pixels,suppress_pixels,suppress_gt_pixels,"
           "moving_priors\n";
}

void MotionEvaluationLogger::call(const FrameData& data,
                                  const VolumetricMap& /* map */,
                                  const Tracks& /* tracks */) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const cv::Mat& labels = data.input.label_image;
  const cv::Mat& ranges = data.input.range_image;
  if (!file_.is_open() || labels.empty() || labels.type() != CV_32SC1 ||
      labels.size() != data.dynamic_image.size()) {
    return;
  }
  const auto& label_space = hydra::GlobalInfo::instance().getLabelSpaceConfig();
  const bool has_seeds = !data.prior_seed_mask.empty();
  const bool has_suppression = !data.prior_suppress_mask.empty();

  // Per-pixel ground truth: pixels of dynamic labels within range.
  cv::Mat gt = cv::Mat::zeros(labels.size(), CV_8UC1);
  int gt_pixels = 0, detected_pixels = 0, tp_pixels = 0;
  int seed_pixels = 0, seed_gt_pixels = 0, suppress_pixels = 0, suppress_gt_pixels = 0;
  for (int v = 0; v < labels.rows; ++v) {
    for (int u = 0; u < labels.cols; ++u) {
      const float range = ranges.at<InputData::RangeType>(v, u);
      if (!(range > 0.f) || range > config.max_range) {
        continue;
      }
      const int label = labels.at<InputData::LabelType>(v, u);
      const bool is_gt = label >= 0 && label_space.isDynamic(label);
      const bool detected = data.dynamic_image.at<FrameData::DynamicImageType>(v, u) != 0;
      gt.at<uint8_t>(v, u) = is_gt;
      gt_pixels += is_gt;
      detected_pixels += detected;
      tp_pixels += is_gt && detected;
      if (has_seeds && data.prior_seed_mask.at<uint8_t>(v, u)) {
        ++seed_pixels;
        seed_gt_pixels += is_gt;
      }
      if (has_suppression && data.prior_suppress_mask.at<uint8_t>(v, u)) {
        ++suppress_pixels;
        suppress_gt_pixels += is_gt;
      }
    }
  }

  // Per-cluster precision.
  int tp_clusters = 0;
  for (const auto& cluster : data.dynamic_clusters) {
    int cluster_gt = 0;
    for (const Pixel& pixel : cluster.pixels) {
      cluster_gt += gt.at<uint8_t>(pixel.v, pixel.u);
    }
    tp_clusters += !cluster.pixels.empty() &&
                   cluster_gt >= config.min_cluster_precision * cluster.pixels.size();
  }

  int moving_priors = 0;
  for (const auto& obs : data.prior_observations) {
    moving_priors += obs.moving;
  }

  file_ << data.input.timestamp_ns << "," << gt_pixels << "," << detected_pixels << ","
        << tp_pixels << "," << data.dynamic_clusters.size() << "," << tp_clusters << ","
        << seed_pixels << "," << seed_gt_pixels << "," << suppress_pixels << ","
        << suppress_gt_pixels << "," << moving_priors << "\n";
  file_.flush();
}

std::string MotionEvaluationLogger::printInfo() const { return config::toString(config); }

}  // namespace khronos
