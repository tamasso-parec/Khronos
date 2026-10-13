#pragma once

#include <dynamic_psg/dynamics_estimate.h>
#include <hydra/input/input_data.h>

namespace khronos {

// Per-pixel dynamics evidence, implemented by the dynamic_psg package.
using dynamic_psg::DynamicsEstimate;
using dynamic_psg::DynamicsFailure;
using dynamic_psg::DynamicsValidity;

// Independent of binary motion clustering; reference geometry is never fitted or extrapolated.
class DynamicsEstimator {
 public:
  virtual ~DynamicsEstimator() = default;
  virtual DynamicsEstimate::ConstPtr processInput(const hydra::InputData& input) = 0;
  virtual void reset() = 0;
};

}  // namespace khronos
