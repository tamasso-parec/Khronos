#include "khronos/active_window/motion_detection/temporal_motion_model.h"

#include <random>

#include <Eigen/LU>
#include <gtest/gtest.h>

namespace khronos {

TEST(TemporalMotionModel, StaticNoiseHasSensibleFalsePositiveRate) {
  std::mt19937 generator(73);
  std::normal_distribution<double> noise(0.0, 0.01);
  int significant = 0;
  for (int trial = 0; trial < 1000; ++trial) {
    TemporalMotionModel model;
    for (int i = 0; i < 6; ++i) {
      ASSERT_TRUE(model.add(Eigen::Vector3d(100 + noise(generator), 20 + noise(generator), noise(generator)),
                            0.0001 * Eigen::Matrix3d::Identity(), i * 0.1, 1.0));
    }
    const auto fit = model.fit();
    ASSERT_TRUE(fit.valid);
    significant += fit.static_cost - fit.dynamic_cost > 11.345;
  }
  EXPECT_GT(significant, 0);
  EXPECT_LT(significant, 35);
}

TEST(TemporalMotionModel, WeightedStaticPointDoesNotRemainAtBiasedReference) {
  TemporalMotionModel model;
  ASSERT_TRUE(model.add({0.2, 0, 2}, 0.04 * Eigen::Matrix3d::Identity(), 0.0, 1.0));
  for (int i = 1; i < 5; ++i) {
    ASSERT_TRUE(model.add({0, 0, 2}, 0.0001 * Eigen::Matrix3d::Identity(), i * 0.1, 1.0));
  }
  const auto fit = model.fit();
  ASSERT_TRUE(fit.valid);
  EXPECT_LT(fit.static_point.x(), 0.001);
  EXPECT_LT(fit.static_cost, 1.1);
}

TEST(TemporalMotionModel, VelocityUsesActualCenteredTimestampsAndKnownCovariance) {
  TemporalMotionModel model;
  const Eigen::Vector3d velocity(0.6, -0.2, 0.1);
  for (double time : {0.0, 0.04, 0.11, 0.24, 0.4}) {
    ASSERT_TRUE(model.add(Eigen::Vector3d(1000, -200, 3) + time * velocity,
                          0.0001 * Eigen::Matrix3d::Identity(), time, 0.8));
  }
  const auto fit = model.fit();
  ASSERT_TRUE(fit.valid);
  EXPECT_LT((fit.velocity - velocity).norm(), 1.e-8);
  EXPECT_NEAR(fit.span, 0.4, 1.e-12);
  EXPECT_NEAR(fit.reliability, 0.8, 1.e-12);
  EXPECT_LT(fit.dynamic_cost, 1.e-6);
  EXPECT_GT(fit.static_cost, 300.0);
  EXPECT_GT(fit.velocity_covariance.determinant(), 0.0);
}

TEST(TemporalMotionModel, TwoSamplesRepeatedTimesAndDegenerateSpanAreInsufficient) {
  TemporalMotionModel model;
  const auto covariance = Eigen::Matrix3d::Identity();
  ASSERT_TRUE(model.add({0, 0, 2}, covariance, 0.0, 1));
  EXPECT_FALSE(model.add({0, 0, 2}, covariance, 0.0, 1));
  ASSERT_TRUE(model.add({1, 0, 2}, covariance, 0.1, 1));
  EXPECT_FALSE(model.fit().valid);
  TemporalMotionModel short_span;
  for (int i = 0; i < 4; ++i) ASSERT_TRUE(short_span.add({0, 0, 2}, covariance, i * 1.e-11, 1));
  EXPECT_FALSE(short_span.fit().valid);
  EXPECT_FALSE(model.add({0, 0, 2}, -covariance, 0.2, 1));
}

TEST(TemporalMotionModel, HigherUncertaintyWeakensModelImprovement) {
  TemporalMotionModel low, high;
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(low.add({i * 0.03, 0, 2}, 0.0001 * Eigen::Matrix3d::Identity(), i * 0.05, 1));
    ASSERT_TRUE(high.add({i * 0.03, 0, 2}, 0.01 * Eigen::Matrix3d::Identity(), i * 0.05, 1));
  }
  EXPECT_GT(low.fit().static_cost - low.fit().dynamic_cost,
            90 * (high.fit().static_cost - high.fit().dynamic_cost));
}

TEST(TemporalMotionModel, LongerTemporalBaselineStrengthensEvidence) {
  TemporalMotionModel short_track, long_track;
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(short_track.add({i * 0.02, 0, 2}, 0.0001 * Eigen::Matrix3d::Identity(), i * 0.02, 1));
    ASSERT_TRUE(long_track.add({i * 0.1, 0, 2}, 0.0001 * Eigen::Matrix3d::Identity(), i * 0.1, 1));
  }
  EXPECT_GT(long_track.fit().static_cost - long_track.fit().dynamic_cost,
            20 * (short_track.fit().static_cost - short_track.fit().dynamic_cost));
}

}  // namespace khronos
