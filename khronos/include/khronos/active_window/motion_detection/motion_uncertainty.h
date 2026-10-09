#pragma once

#include <Eigen/Core>

namespace khronos {

// Derivative of unproject(u, v, z(u,v)) with respect to (u, v, independent depth noise).
Eigen::Matrix3d rgbdPointJacobian(double u, double v, double z,
                                double fx, double fy, double cx, double cy,
                                const Eigen::Vector2d& depth_gradient);

// Online whitening of correlated first-frame-relative residuals. The latent state is
// [current pixel error (2), fixed reference-point error in world coordinates (3)].
// Conditioning this noise state never updates the actual reference point or tracked pixels.
class MotionUncertainty {
 public:
  void initialize(const Eigen::Matrix3d& reference_jacobian,
                  double pixel_variance, double depth_variance);
  void predict(const Eigen::Matrix2d& flow_jacobian, double flow_variance);
  bool observe(const Eigen::Vector3d& residual,
               const Eigen::Matrix3d& world_jacobian,
               double depth_variance,
               double& squared_innovation,
               Eigen::Matrix3d& innovation_covariance);

 private:
  Eigen::Matrix<double, 5, 1> mean_ = Eigen::Matrix<double, 5, 1>::Zero();
  Eigen::Matrix<double, 5, 5> covariance_ = Eigen::Matrix<double, 5, 5>::Zero();
};

}  // namespace khronos
