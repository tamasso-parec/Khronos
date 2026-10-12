#include "khronos/active_window/motion_detection/temporal_motion_model.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>

namespace khronos {

bool TemporalMotionModel::add(const Eigen::Vector3d& point, const Eigen::Matrix3d& covariance,
                              double time, double reliability) {
  if (!point.allFinite() || !covariance.allFinite() || !std::isfinite(time) ||
      !std::isfinite(reliability) || reliability < 0.0 || reliability > 1.0 ||
      !covariance.isApprox(covariance.transpose(), 1.e-10) || (count_ && time <= last_time_)) {
    return false;
  }
  const Eigen::LLT<Eigen::Matrix3d> factor(covariance);
  if (factor.info() != Eigen::Success) return false;
  const Eigen::Matrix3d weight = factor.solve(Eigen::Matrix3d::Identity());
  if (!weight.allFinite()) return false;
  if (!count_) {
    *this = TemporalMotionModel();
    origin_ = point;
    first_time_ = time;
  }
  const Eigen::Vector3d centered = point - origin_;
  precision_ += weight;
  time_precision_ += time * weight;
  time2_precision_ += time * time * weight;
  weighted_point_ += weight * centered;
  time_weighted_point_ += time * weight * centered;
  weighted_norm_ += centered.dot(weight * centered);
  time_sum_ += time;
  last_time_ = time;
  reliability_sum_ += reliability;
  ++count_;
  return true;
}

TemporalMotionFit TemporalMotionModel::fit() const {
  TemporalMotionFit result;
  result.observations = count_;
  if (!count_) return result;
  result.span = last_time_ - first_time_;
  result.reliability = reliability_sum_ / count_;
  const Eigen::LLT<Eigen::Matrix3d> static_factor(precision_);
  if (static_factor.info() != Eigen::Success) return result;
  const Eigen::Vector3d static_offset = static_factor.solve(weighted_point_);
  result.static_point = origin_ + static_offset;
  result.static_cost = std::max(0.0, weighted_norm_ - weighted_point_.dot(static_offset));
  if (count_ < 3 || result.span <= 1.e-9) return result;
  const double center = time_sum_ / count_;
  const Eigen::Matrix3d cross = time_precision_ - center * precision_;
  const Eigen::Matrix3d time2 = time2_precision_ - 2.0 * center * time_precision_ +
                               center * center * precision_;
  Eigen::Matrix<double, 6, 6> normal;
  normal << precision_, cross, cross.transpose(), time2;
  Eigen::Matrix<double, 6, 1> rhs;
  rhs << weighted_point_, time_weighted_point_ - center * weighted_point_;
  const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> factor(normal);
  if (factor.info() != Eigen::Success || !factor.isPositive() ||
      factor.vectorD().minCoeff() <= 1.e-12 * factor.vectorD().maxCoeff()) return result;
  const Eigen::Matrix<double, 6, 1> parameters = factor.solve(rhs);
  // Only the velocity columns of the inverse normal matrix are needed.
  const Eigen::Matrix<double, 6, 3> velocity_columns =
      factor.solve(Eigen::Matrix<double, 6, 6>::Identity().rightCols<3>());
  result.velocity = parameters.tail<3>();
  result.velocity_covariance = velocity_columns.bottomRows<3>();
  result.dynamic_cost = std::max(0.0, weighted_norm_ - rhs.dot(parameters));
  result.valid = parameters.allFinite() && velocity_columns.allFinite() &&
                 std::isfinite(result.static_cost) && std::isfinite(result.dynamic_cost);
  return result;
}

}  // namespace khronos
