#pragma once

#include <fstream>
#include <mutex>
#include <string>

#include <config_utilities/config_utilities.h>

#include "khronos/active_window/active_window.h"

namespace khronos {

/**
 * @brief Active window sink that appends the per-frame evidence of every rendered object prior
 * to a CSV file for offline analysis.
 */
class PriorEvidenceLogger : public ActiveWindow::KhronosSink {
 public:
  // Config.
  struct Config {
    // File to write the evidence to. Will be overwritten.
    std::string output_file;
  } const config;

  // Construction.
  explicit PriorEvidenceLogger(const Config& config);
  virtual ~PriorEvidenceLogger() = default;

  void call(const FrameData& data, const VolumetricMap& map, const Tracks& tracks) const override;
  std::string printInfo() const override;

 private:
  mutable std::mutex mutex_;
  mutable std::ofstream file_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<ActiveWindow::KhronosSink,
                                     PriorEvidenceLogger,
                                     PriorEvidenceLogger::Config>("PriorEvidenceLogger");
};

void declare_config(PriorEvidenceLogger::Config& config);

}  // namespace khronos
