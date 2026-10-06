#pragma once

#include <ianvs/node_handle.h>
#include <khronos/active_window/active_window.h>
#include <khronos_msgs/msg/measurement_frame.hpp>

namespace khronos {

// Numeric export only: does not modify the active window or retain its frames.
class MeasurementSink : public ActiveWindow::KhronosSink {
 public:
  explicit MeasurementSink(ianvs::NodeHandle nh);
  void call(const FrameData& data, const VolumetricMap& map,
            const Tracks& tracks) const override;

 private:
  rclcpp::Publisher<khronos_msgs::msg::MeasurementFrame>::SharedPtr publisher_;
  mutable uint64_t sequence_ = 0;
};

}  // namespace khronos
