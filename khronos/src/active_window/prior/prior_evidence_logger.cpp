#include "khronos/active_window/prior/prior_evidence_logger.h"

#include <filesystem>

namespace khronos {

void declare_config(PriorEvidenceLogger::Config& config) {
  using namespace config;
  name("PriorEvidenceLogger");
  field(config.output_file, "output_file");
  checkCondition(!config.output_file.empty(), "param 'output_file' must not be empty");
}

PriorEvidenceLogger::PriorEvidenceLogger(const Config& config)
    : config(config::checkValid(config)) {
  const std::filesystem::path path(config.output_file);
  std::error_code error;
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path(), error);
  }
  file_.open(path);
  if (!file_.is_open()) {
    LOG(ERROR) << "[Prior Evidence Logger] Could not open '" << config.output_file << "'.";
    return;
  }
  file_ << "timestamp_ns,source,node_id,semantic_label,rendered,excluded,present,absent,occluded,"
           "unobserved,absent_fraction,largest_absent_component,baseline,moving,seed_pixels,"
           "dominant_label,dominant_label_count\n";
}

void PriorEvidenceLogger::call(const FrameData& data,
                               const VolumetricMap& /* map */,
                               const Tracks& /* tracks */) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!file_.is_open()) {
    return;
  }

  for (const PriorObservation& obs : data.prior_observations) {
    int dominant_label = -1;
    int dominant_count = 0;
    for (const auto& [label, count] : obs.labels) {
      if (count > dominant_count) {
        dominant_label = label;
        dominant_count = count;
      }
    }
    const bool is_track = obs.source == PriorSource::kTrack;
    file_ << data.input.timestamp_ns << "," << (is_track ? "track" : "scene_graph") << ","
          << (is_track ? "T(" + std::to_string(obs.id) + ")" : NodeSymbol(obs.id).str()) << ","
          << obs.semantic_label << "," << obs.rendered << "," << obs.excluded << ","
          << obs.present << "," << obs.absent << "," << obs.occluded << "," << obs.unobserved
          << "," << obs.absentFraction() << "," << obs.largest_absent_component << ","
          << obs.baseline << "," << obs.moving << "," << obs.seed_pixels << "," << dominant_label
          << "," << dominant_count << "\n";
  }
  file_.flush();
}

std::string PriorEvidenceLogger::printInfo() const { return config::toString(config); }

}  // namespace khronos
