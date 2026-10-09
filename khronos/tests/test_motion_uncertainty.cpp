#include "khronos/active_window/motion_detection/motion_uncertainty.h"

#include <limits>
#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace khronos {
namespace {

TEST(MotionUncertainty, UnprojectionJacobianMatchesFiniteDifferences) {
  const Eigen::Vector3d value(42.3, 38.7, 2.4);
  const Eigen::Vector2d slope(0.003, -0.005);
  const auto unproject = [&](const Eigen::Vector3d& x) -> Eigen::Vector3d {
    const double z = x.z() + slope.dot(x.head<2>() - value.head<2>());
    return Eigen::Vector3d((x.x() - 64.0) * z / 110.0, (x.y() - 48.0) * z / 90.0, z);
  };
  const auto analytic = rgbdPointJacobian(value.x(), value.y(), value.z(),
                                         110.0, 90.0, 64.0, 48.0, slope);
  Eigen::Matrix3d numeric;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d step = Eigen::Vector3d::Zero();
    step[i] = 1.e-5;
    numeric.col(i) = (unproject(value + step) - unproject(value - step)) / 2.e-5;
  }
  EXPECT_LT((analytic - numeric).norm(), 1.e-9);
}

TEST(MotionUncertainty, InnovationsMatchFullJointCovarianceWithSkippedObservation) {
  constexpr int steps = 4;
  constexpr int sources = 3 + 3 * steps;
  const double pixel_sigma = 0.4, flow_sigma = 0.7, depth_sigma = 0.02;
  const auto reference = rgbdPointJacobian(45, 36, 2.0, 110, 90, 64, 48, {0.002, -0.001});
  Eigen::Matrix<double, 2, sources> pixel_design = Eigen::Matrix<double, 2, sources>::Zero();
  pixel_design.leftCols<2>() = pixel_sigma * Eigen::Matrix2d::Identity();
  Eigen::Matrix<double, 3, sources> reference_design = Eigen::Matrix<double, 3, sources>::Zero();
  reference_design.leftCols<3>() =
      reference * Eigen::Vector3d(pixel_sigma, pixel_sigma, depth_sigma).asDiagonal();
  MotionUncertainty state;
  state.initialize(reference, pixel_sigma * pixel_sigma, depth_sigma * depth_sigma);
  Eigen::Matrix<double, 9, sources> design;
  Eigen::Matrix<double, 9, 1> residuals;
  double online = 0.0;
  int accepted = 0;
  for (int i = 0; i < steps; ++i) {
    Eigen::Matrix2d transition;
    transition << 1.01, 0.03, -0.02, 0.99;
    state.predict(transition, flow_sigma * flow_sigma);
    pixel_design = (transition * pixel_design).eval();
    pixel_design.middleCols<2>(3 + 3 * i) += flow_sigma * Eigen::Matrix2d::Identity();
    if (i == 1) {
      continue;
    }
    const Eigen::Matrix3d jacobian =
        Eigen::AngleAxisd(0.1 * i, Eigen::Vector3d::UnitY()).toRotationMatrix() *
        rgbdPointJacobian(45 + i, 36 - i, 2.0 + 0.1 * i, 110, 90, 64, 48, {0.002, -0.001});
    const Eigen::Vector3d residual(0.015 * (i + 1), -0.01 * i, 0.025);
    auto block = design.middleRows<3>(3 * accepted);
    block = jacobian.leftCols<2>() * pixel_design - reference_design;
    block.col(5 + 3 * i) += depth_sigma * jacobian.col(2);
    residuals.segment<3>(3 * accepted) = residual;
    double increment;
    Eigen::Matrix3d covariance;
    ASSERT_TRUE(state.observe(residual, jacobian, depth_sigma * depth_sigma, increment, covariance));
    online += increment;
    ++accepted;
  }
  const Eigen::Matrix<double, 9, 9> covariance = design * design.transpose();
  const double batch = residuals.dot(covariance.llt().solve(residuals));
  EXPECT_NEAR(online, batch, 1.e-9);
}

TEST(MotionUncertainty, StaticGaussianModelHasExpectedChiSquareMoments) {
  constexpr int trials = 3000;
  const Eigen::Matrix3d jacobian = rgbdPointJacobian(64, 48, 2, 100, 100, 64, 48, {0, 0});
  std::mt19937 rng(37);
  std::normal_distribution<double> normal;
  double sum = 0.0, sum_squared = 0.0;
  for (int trial = 0; trial < trials; ++trial) {
    MotionUncertainty state;
    state.initialize(jacobian, 0.25, 0.0004);
    Eigen::Vector2d pixel(0.5 * normal(rng), 0.5 * normal(rng));
    const Eigen::Vector3d reference = jacobian * Eigen::Vector3d(pixel.x(), pixel.y(), 0.02 * normal(rng));
    double statistic = 0.0;
    for (int i = 0; i < 3; ++i) {
      pixel += Eigen::Vector2d(0.5 * normal(rng), 0.5 * normal(rng));
      state.predict(Eigen::Matrix2d::Identity(), 0.25);
      const Eigen::Vector3d residual =
          jacobian * Eigen::Vector3d(pixel.x(), pixel.y(), 0.02 * normal(rng)) - reference;
      double increment;
      Eigen::Matrix3d covariance;
      ASSERT_TRUE(state.observe(residual, jacobian, 0.0004, increment, covariance));
      statistic += increment;
    }
    sum += statistic;
    sum_squared += statistic * statistic;
  }
  const double mean = sum / trials;
  EXPECT_NEAR(mean, 9.0, 0.35);
  EXPECT_NEAR(sum_squared / trials - mean * mean, 18.0, 2.0);
}

TEST(MotionUncertainty, InvalidCovarianceAndResidualAreRejected) {
  MotionUncertainty state;
  state.initialize(Eigen::Matrix3d::Identity(), 0.0, 0.0);
  double increment;
  Eigen::Matrix3d covariance;
  EXPECT_FALSE(state.observe(Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(),
                             0.0, increment, covariance));
  state.initialize(Eigen::Matrix3d::Identity(), 1.0, 1.0);
  state.predict(Eigen::Matrix2d::Identity(), 1.0);
  EXPECT_FALSE(state.observe(Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
                             Eigen::Matrix3d::Identity(), 1.0, increment, covariance));
}

}  // namespace
}  // namespace khronos
