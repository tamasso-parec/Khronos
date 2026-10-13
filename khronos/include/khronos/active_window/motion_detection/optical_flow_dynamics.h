#pragma once

#include <dynamic_psg/optical_flow_dynamics.h>

#include "khronos/active_window/motion_detection/dynamics_estimator.h"

namespace khronos {

// Runs dynamic_psg::OpticalFlowDynamics on Hydra camera inputs. Registered as
// "OpticalFlowDynamics", so mapper configs select it with `type: OpticalFlowDynamics` and
// configure it with the fields of dynamic_psg::OpticalFlowDynamics::Config.
class OpticalFlowDynamics : public DynamicsEstimator {
 public:
  using Config = dynamic_psg::OpticalFlowDynamics::Config;

  explicit OpticalFlowDynamics(const Config& config);

  // Non-camera sensors reset the estimator and return nullptr.
  DynamicsEstimate::ConstPtr processInput(const hydra::InputData& input) override;
  void reset() override;

  const Config& config() const { return estimator_.config; }

 private:
  dynamic_psg::OpticalFlowDynamics estimator_;
};

// Converts a Hydra camera input, using the sensor pose and Hydra's vertex map; false for other
// sensors. Image buffers are shared, not copied.
bool toRgbdFrame(const hydra::InputData& input, dynamic_psg::RgbdFrame& frame);

}  // namespace khronos
