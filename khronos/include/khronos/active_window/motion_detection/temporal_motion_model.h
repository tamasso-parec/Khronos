#pragma once

#include <Eigen/Core>

namespace khronos {

struct TemporalMotionFit {
  bool valid = false;
  int observations = 0;
  double span = 0.0;
  double reliability = 0.0;
  double static_cost = 0.0;
  double dynamic_cost = 0.0;
  Eigen::Vector3d static_point = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Matrix3d velocity_covariance = Eigen::Matrix3d::Zero();
};

// Fixed-size sufficient statistics, independent-observation approximation.
// Coordinates are translated by the first accepted point solely for numerical conditioning.
class TemporalMotionModel {
 public:
  bool add(const Eigen::Vector3d& point, const Eigen::Matrix3d& covariance,
           double time_seconds, double reliability);
  TemporalMotionFit fit() const;

 private:
  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d precision_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d time_precision_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d time2_precision_ = Eigen::Matrix3d::Zero();
  Eigen::Vector3d weighted_point_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d time_weighted_point_ = Eigen::Vector3d::Zero();
  double weighted_norm_ = 0.0;
  double time_sum_ = 0.0;
  double first_time_ = 0.0;
  double last_time_ = 0.0;
  double reliability_sum_ = 0.0;
  int count_ = 0;
};

}  // namespace khronos
