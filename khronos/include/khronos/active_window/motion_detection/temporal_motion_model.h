#pragma once

#include <Eigen/Cholesky>
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

// Inverse of a small SPD matrix (N <= 4) from its Cholesky factor. It performs the operations of
// Eigen 3.4's LLT::solve(Identity) in the same order, so results are bit-identical, without the
// blocked-kernel overhead that dominates that call at this size.
template <int N>
Eigen::Matrix<double, N, N> choleskyInverse(const Eigen::LLT<Eigen::Matrix<double, N, N>>& factor) {
  static_assert(N <= 4, "larger systems use Eigen's blocked update path");
  const auto& l = factor.matrixLLT();  // The lower triangle holds L.
  Eigen::Matrix<double, N, N> x = Eigen::Matrix<double, N, N>::Identity();
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {  // L y = e_j, column-oriented.
      x(i, j) *= 1.0 / l(i, i);
      const double b = x(i, j);
      for (int r = i + 1; r < N; ++r) x(r, j) -= b * l(r, i);
    }
    for (int i = N - 1; i >= 0; --i) {  // L^T x = y, row-oriented.
      double b = 0.0;
      for (int r = i + 1; r < N; ++r) b += l(r, i) * x(r, j);
      x(i, j) = (x(i, j) - b) * (1.0 / l(i, i));
    }
  }
  return x;
}

// Fixed-size sufficient statistics, independent-observation approximation.
// Coordinates are translated by the first accepted point solely for numerical conditioning.
class TemporalMotionModel {
 public:
  bool add(const Eigen::Vector3d& point,
           const Eigen::Matrix3d& covariance,
           double time_seconds,
           double reliability);
  TemporalMotionFit fit() const;
  // Forgets all observations in O(1); the next add() resets the accumulators.
  void clear() { count_ = 0; }
  int observations() const { return count_; }
  double span() const { return count_ ? last_time_ - first_time_ : 0.0; }
  double reliability() const { return count_ ? reliability_sum_ / count_ : 0.0; }

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
