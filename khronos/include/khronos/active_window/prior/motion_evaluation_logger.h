#pragma once

#include <fstream>
#include <mutex>
#include <string>

#include <config_utilities/config_utilities.h>

#include "khronos/active_window/active_window.h"

namespace khronos {

/**
 * @brief Active window sink that evaluates the motion detection of every frame against the
 * semantic label image, counting pixels of dynamic labels (e.g. humans) as ground truth moving.
 * Writes one CSV row per frame. Independent of whether object priors are used, so runs with and
 * without priors can be compared.
 */
class MotionEvaluationLogger : public ActiveWindow::KhronosSink {
 public:
  // Config.
  struct Config {
    // File to write the evaluation to. Will be overwritten.
    std::string output_file;

    // Only evaluate pixels up to this range in meters, should match the motion detector.
    float max_range = 5.f;

    // A detected cluster counts as true positive if at least this fraction of its pixels is
    // ground truth dynamic.
    float min_cluster_precision = 0.5f;
  } const config;

  // Construction.
  explicit MotionEvaluationLogger(const Config& config);
  virtual ~MotionEvaluationLogger() = default;

  void call(const FrameData& data, const VolumetricMap& map, const Tracks& tracks) const override;
  std::string printInfo() const override;

 private:
  mutable std::mutex mutex_;
  mutable std::ofstream file_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<ActiveWindow::KhronosSink,
                                     MotionEvaluationLogger,
                                     MotionEvaluationLogger::Config>("MotionEvaluationLogger");
};

void declare_config(MotionEvaluationLogger::Config& config);

}  // namespace khronos
