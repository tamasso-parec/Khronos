#include "khronos/active_window/motion_detection/motion_uncertainty.h"

#include <cmath>

#include <Eigen/Cholesky>

namespace khronos {

Eigen::Matrix3d rgbdPointJacobian(double u, double v, double z,
                                double fx, double fy, double cx, double cy,
                                const Eigen::Vector2d& depth_gradient) {
  Eigen::Matrix3d jacobian = Eigen::Matrix3d::Zero();
  jacobian(0, 0) = z / fx;
  jacobian(1, 1) = z / fy;
  jacobian.col(2) = Eigen::Vector3d((u - cx) / fx, (v - cy) / fy, 1.0);
  jacobian.leftCols<2>() += jacobian.col(2) * depth_gradient.transpose();
  return jacobian;
}

void MotionUncertainty::initialize(const Eigen::Matrix3d& jacobian,
                                   double pixel_variance, double depth_variance) {
  mean_.setZero();
  covariance_.setZero();
  covariance_.topLeftCorner<2, 2>() = pixel_variance * Eigen::Matrix2d::Identity();
  covariance_.topRightCorner<2, 3>() = pixel_variance * jacobian.leftCols<2>().transpose();
  covariance_.bottomLeftCorner<3, 2>() = covariance_.topRightCorner<2, 3>().transpose();
  covariance_.bottomRightCorner<3, 3>() =
      jacobian * Eigen::Vector3d(pixel_variance, pixel_variance, depth_variance).asDiagonal() *
      jacobian.transpose();
}

void MotionUncertainty::predict(const Eigen::Matrix2d& flow_jacobian, double flow_variance) {
  Eigen::Matrix<double, 5, 5> transition = Eigen::Matrix<double, 5, 5>::Identity();
  transition.topLeftCorner<2, 2>() = flow_jacobian;
  mean_ = (transition * mean_).eval();
  covariance_ = (transition * covariance_ * transition.transpose()).eval();
  // ponytail: independent new flow errors; add image-reuse correlations if calibration needs them.
  covariance_.topLeftCorner<2, 2>().diagonal().array() += flow_variance;
}

bool MotionUncertainty::observe(const Eigen::Vector3d& residual,
                                const Eigen::Matrix3d& jacobian,
                                double depth_variance,
                                double& squared_innovation,
                                Eigen::Matrix3d& innovation_covariance) {
  Eigen::Matrix<double, 3, 5> observation;
  observation.leftCols<2>() = jacobian.leftCols<2>();
  observation.rightCols<3>() = -Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d noise = depth_variance * jacobian.col(2) * jacobian.col(2).transpose();
  const Eigen::Matrix<double, 5, 3> cross = covariance_ * observation.transpose();
  innovation_covariance = observation * cross + noise;
  innovation_covariance = (0.5 * (innovation_covariance + innovation_covariance.transpose())).eval();
  if (!innovation_covariance.allFinite() || !mean_.allFinite() || !residual.allFinite()) {
    return false;
  }
  const Eigen::LLT<Eigen::Matrix3d> factor(innovation_covariance);
  if (factor.info() != Eigen::Success) {
    return false;
  }
  const Eigen::Vector3d innovation = residual - observation * mean_;
  squared_innovation = innovation.dot(factor.solve(innovation));
  if (!std::isfinite(squared_innovation) || squared_innovation < 0.0) {
    return false;
  }
  const Eigen::Matrix<double, 5, 3> gain = factor.solve(cross.transpose()).transpose();
  const Eigen::Matrix<double, 5, 1> mean = mean_ + gain * innovation;
  const Eigen::Matrix<double, 5, 5> update =
      Eigen::Matrix<double, 5, 5>::Identity() - gain * observation;
  // Joseph form preserves positive semidefiniteness when depth noise has rank one.
  const Eigen::Matrix<double, 5, 5> covariance =
      update * covariance_ * update.transpose() + gain * noise * gain.transpose();
  if (!mean.allFinite() || !covariance.allFinite()) {
    return false;
  }
  mean_ = mean;
  covariance_ = 0.5 * (covariance + covariance.transpose());
  return true;
}

}  // namespace khronos
