#include "khronos/active_window/motion_detection/optical_flow_dynamics.h"

#include <config_utilities/config_utilities.h>
#include <glog/logging.h>
#include <hydra/input/camera.h>

namespace khronos {
namespace {

const auto registration =
    config::RegistrationWithConfig<DynamicsEstimator,
                                   OpticalFlowDynamics,
                                   OpticalFlowDynamics::Config>("OpticalFlowDynamics");

}  // namespace

bool toRgbdFrame(const hydra::InputData& input, dynamic_psg::RgbdFrame& frame) {
  const auto* camera = dynamic_cast<const hydra::Camera*>(&input.getSensor());
  if (!camera) {
    return false;
  }
  const auto& intrinsics = camera->getConfig();
  frame.timestamp_ns = input.timestamp_ns;
  frame.sensor_name = camera->name;
  frame.camera.fx = intrinsics.fx;
  frame.camera.fy = intrinsics.fy;
  frame.camera.cx = intrinsics.cx;
  frame.camera.cy = intrinsics.cy;
  frame.camera.width = intrinsics.width;
  frame.camera.height = intrinsics.height;
  frame.camera.min_range = camera->min_range();
  frame.camera.max_range = camera->max_range();
  frame.world_T_camera = input.getSensorPose();
  frame.color = input.color_image;
  frame.depth = input.depth_image;
  frame.max_range = input.max_range;
  frame.vertex_map = input.vertex_map;
  frame.points_in_world_frame = input.points_in_world_frame;
  return true;
}

OpticalFlowDynamics::OpticalFlowDynamics(const Config& config) : estimator_(config) {}

DynamicsEstimate::ConstPtr OpticalFlowDynamics::processInput(const hydra::InputData& input) {
  dynamic_psg::RgbdFrame frame;
  if (!toRgbdFrame(input, frame)) {
    LOG(WARNING) << "[OpticalFlowDynamics] Expected a camera input; resetting window.";
    estimator_.reset();
    return nullptr;
  }
  return estimator_.process(frame);
}

void OpticalFlowDynamics::reset() { estimator_.reset(); }

}  // namespace khronos
