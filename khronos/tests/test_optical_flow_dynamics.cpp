#include "khronos/active_window/motion_detection/optical_flow_dynamics.h"

#include <cmath>
#include <limits>

#include <config_utilities/parsing/yaml.h>
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

namespace khronos {
namespace {

class OpticalFlowDynamicsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    hydra::Camera::Config camera_config;
    camera_config.width = 128;
    camera_config.height = 96;
    camera_config.fx = camera_config.fy = 100.f;
    camera_config.cx = 64.f;
    camera_config.cy = 48.f;
    camera_config.min_range = 0.1;
    camera_config.max_range = 10.0;
    hydra::ParamSensorExtrinsics::Config extrinsics;
    extrinsics.body_R_sensor = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ());
    extrinsics.body_p_sensor = Eigen::Vector3d(0.2, -0.1, 0.3);
    camera_config.extrinsics = extrinsics;
    camera = std::make_shared<hydra::Camera>(camera_config, "rgbd");
    cv::Mat noise(96, 128, CV_8UC1);
    cv::RNG rng(42);
    rng.fill(noise, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(noise, noise, cv::Size(3, 3), 0.6);
    cv::cvtColor(noise, color, cv::COLOR_GRAY2RGB);
    config.scoring_mode = "legacy";
    config.min_observations = 1;
    config.flow_sigma = 0.5;
    config.depth_sigma = 0.01;
    config.depth_sigma_scale = 0.0;
  }

  hydra::InputData frame(uint64_t stamp,
                         float image_shift = 0.f,
                         double camera_x = 0.0,
                         float depth = 2.f,
                         bool world_points = false) const {
    hydra::InputData input(camera);
    input.timestamp_ns = stamp;
    Eigen::Isometry3d world_T_sensor = Eigen::Isometry3d::Identity();
    world_T_sensor.translation().x() = camera_x;
    input.world_T_body = world_T_sensor * camera->body_T_sensor().inverse();
    const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, image_shift, 0, 1, 0);
    cv::warpAffine(color, input.color_image, shift, color.size(), cv::INTER_LINEAR,
                   cv::BORDER_REFLECT);
    input.depth_image = cv::Mat(color.size(), CV_32FC1, cv::Scalar(depth));
    const Eigen::Isometry3f pose = world_T_sensor.cast<float>();
    input.vertex_map = camera->computeVertexMap(input.depth_image, world_points ? &pose : nullptr);
    input.points_in_world_frame = world_points;
    return input;
  }

  std::shared_ptr<hydra::Camera> camera;
  cv::Mat color;
  OpticalFlowDynamics::Config config;
};

TEST_F(OpticalFlowDynamicsTest, StaticSceneAndSnapshotOwnership) {
  config.min_observations = 2;
  OpticalFlowDynamics estimator(config);
  const auto first = estimator.processInput(frame(1000000000));
  ASSERT_TRUE(first);
  EXPECT_TRUE(std::isnan(first->probability.at<float>(48, 64)));
  const auto second = estimator.processInput(frame(1050000000));
  EXPECT_TRUE(std::isnan(second->probability.at<float>(48, 64)));
  const auto third = estimator.processInput(frame(1100000000));
  EXPECT_EQ(third->observation_count.at<int>(48, 64), 2);
  EXPECT_LT(third->probability.at<float>(48, 64), 1.e-5);
  EXPECT_EQ(first->observation_count.at<int>(48, 64), 0);
  EXPECT_EQ(second->observation_count.at<int>(48, 64), 1);
  EXPECT_EQ(third->reference_timestamp_ns, 1000000000u);
  EXPECT_EQ(third->latest_timestamp_ns, 1100000000u);
  EXPECT_EQ(third->num_frames, 3);
  EXPECT_EQ(third->reference_points.at<cv::Vec3f>(48, 64), cv::Vec3f(0.f, 0.f, 2.f));
}

TEST_F(OpticalFlowDynamicsTest, CameraMotionIsRemovedWithExtrinsicsAndBothPointFrames) {
  for (bool world_points : {false, true}) {
    OpticalFlowDynamics estimator(config);
    estimator.processInput(frame(1000000000, 0.f, 0.2, 2.f, world_points));
    // Camera moves +4 cm: a static plane at 2 m shifts -2 pixels at fx=100.
    const auto result = estimator.processInput(frame(1050000000, -2.f, 0.24, 2.f, world_points));
    EXPECT_EQ(result->observation_count.at<int>(48, 64), 1);
    EXPECT_LT(result->probability.at<float>(48, 64), 0.01);
    EXPECT_NEAR(result->reference_points.at<cv::Vec3f>(48, 64)[0], 0.2f, 1.e-6);
  }
}

TEST_F(OpticalFlowDynamicsTest, MovingSurfaceAndChainedSubpixelTracks) {
  OpticalFlowDynamics estimator(config);
  estimator.processInput(frame(1000000000));
  const auto second = estimator.processInput(frame(1050000000, 2.5f));
  EXPECT_GT(second->probability.at<float>(48, 64), 0.99);
  const auto third = estimator.processInput(frame(1100000000, 5.f));
  EXPECT_EQ(third->observation_count.at<int>(48, 64), 2);
  EXPECT_GT(third->probability.at<float>(48, 64), 0.999);
  EXPECT_GT(third->squared_error.at<double>(48, 64), 20.0);
  EXPECT_GT(third->squared_error.at<double>(48, 64), second->squared_error.at<double>(48, 64));
  EXPECT_EQ(third->failure_flags.at<uint16_t>(48, 64), 0);
  EXPECT_NEAR(third->residual_norm.at<float>(48, 64), 0.1, 0.005);
}

TEST_F(OpticalFlowDynamicsTest, CameraRotationIsRemoved) {
  OpticalFlowDynamics estimator(config);
  estimator.processInput(frame(1000000000));
  auto next = frame(1050000000);
  constexpr double angle = 0.03;
  const cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f(64.f, 48.f),
                                                 angle * 180.0 / CV_PI, 1.0);
  cv::warpAffine(color, next.color_image, rotation, color.size(), cv::INTER_LINEAR,
                 cv::BORDER_REFLECT);
  Eigen::Isometry3d world_T_sensor = Eigen::Isometry3d::Identity();
  world_T_sensor.linear() = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  next.world_T_body = world_T_sensor * camera->body_T_sensor().inverse();
  const auto result = estimator.processInput(next);
  EXPECT_EQ(result->observation_count.at<int>(30, 40), 1);
  EXPECT_LT(result->probability.at<float>(30, 40), 0.01);
}

TEST_F(OpticalFlowDynamicsTest, ImageBoundaryAndInconsistentFlowRejectTracks) {
  OpticalFlowDynamics estimator(config);
  estimator.processInput(frame(1000000000));
  const auto result = estimator.processInput(frame(1050000000, 4.f));
  EXPECT_EQ(result->observation_count.at<int>(48, 125), 0);
  EXPECT_TRUE(result->failure_flags.at<uint16_t>(48, 125) & kOutsideImage);
  EXPECT_TRUE(std::isnan(result->probability.at<float>(48, 125)));

  // A near-zero tolerance rejects interpolation/estimation inconsistencies even when
  // both images are textured. It must not turn a rejected correspondence into motion.
  config.max_forward_backward_error = 1.e-10;
  OpticalFlowDynamics strict(config);
  strict.processInput(frame(1000000000));
  const auto rejected = strict.processInput(frame(1050000000, 2.5f));
  EXPECT_EQ(rejected->observation_count.at<int>(48, 64), 0);
  EXPECT_TRUE(rejected->failure_flags.at<uint16_t>(48, 64) & kFlowInconsistent);
  EXPECT_EQ(rejected->failure_count.at<int>(48, 64), 1);
  EXPECT_GT(rejected->forward_backward_error.at<float>(48, 64), 0.f);
  const auto later = strict.processInput(frame(1100000000, 5.f));
  EXPECT_EQ(later->failure_count.at<int>(48, 64), 1);
  EXPECT_EQ(later->observation_count.at<int>(48, 64), 0);
  EXPECT_TRUE(std::isnan(rejected->probability.at<float>(48, 64)));
}

TEST_F(OpticalFlowDynamicsTest, ThreeDimensionalResidualAndChiSquareDegreesOfFreedom) {
  config.depth_sigma = 0.05 / std::sqrt(2.0);
  OpticalFlowDynamics estimator(config);
  estimator.processInput(frame(1000000000));
  const auto second = estimator.processInput(frame(1050000000, 0.f, 0.0, 2.05f));
  EXPECT_NEAR(second->squared_error.at<double>(48, 64), 1.0, 1.e-4);
  // chi-square(3) at 1.0. Depth-only motion is invisible to a 2D reprojection residual.
  EXPECT_NEAR(second->probability.at<float>(48, 64), 0.1987480431, 1.e-4);
  const auto third = estimator.processInput(frame(1100000000, 0.f, 0.0, 2.05f));
  // Shared reference variance couples the two residuals: joint statistic is 4/3, not 2.
  EXPECT_NEAR(third->squared_error.at<double>(48, 64), 4.0 / 3.0, 1.e-3);
  EXPECT_NEAR(third->probability.at<float>(48, 64),
              1.0 - std::exp(-2.0 / 3.0) * (1.0 + 2.0 / 3.0 + 2.0 / 9.0), 1.e-4);
}

TEST_F(OpticalFlowDynamicsTest, DepthFailuresRetainEvidenceAndPermitFlowTrackRecovery) {
  OpticalFlowDynamics estimator(config);
  auto first = frame(1000000000);
  first.depth_image.at<float>(20, 20) = 0.f;
  first.vertex_map.at<cv::Vec3f>(30, 30)[0] = std::numeric_limits<float>::quiet_NaN();
  estimator.processInput(first);
  auto next = frame(1050000000);
  next.depth_image(cv::Rect(62, 46, 5, 5)).setTo(std::numeric_limits<float>::quiet_NaN());
  next.depth_image.at<float>(41, 41) = 3.f;
  auto result = estimator.processInput(next);
  for (const cv::Point& p : {cv::Point(20, 20), cv::Point(30, 30), cv::Point(64, 48),
                           cv::Point(40, 40)}) {
    EXPECT_EQ(result->observation_count.at<int>(p), 0);
    EXPECT_TRUE(std::isnan(result->probability.at<float>(p)));
  }
  EXPECT_TRUE(result->failure_flags.at<uint16_t>(48, 64) & kInvalidDepth);
  EXPECT_TRUE(result->failure_flags.at<uint16_t>(40, 40) & kDepthDiscontinuity);
  EXPECT_EQ(result->failure_count.at<int>(48, 64), 1);
  EXPECT_EQ(result->track_alive.at<uint8_t>(48, 64), 255);
  const auto failed = result;
  result = estimator.processInput(frame(1100000000));
  EXPECT_EQ(result->observation_count.at<int>(48, 64), 1);
  EXPECT_EQ(result->observation_count.at<int>(40, 40), 1);
  EXPECT_EQ(result->observation_count.at<int>(20, 20), 0);
  EXPECT_TRUE(result->failure_flags.at<uint16_t>(48, 64) & kInvalidDepth);
  EXPECT_EQ(result->failure_count.at<int>(48, 64), 1);
  EXPECT_EQ(failed->observation_count.at<int>(48, 64), 0);
}

TEST_F(OpticalFlowDynamicsTest, UntexturedRegionsStayUnknown) {
  color.setTo(cv::Scalar::all(100));
  OpticalFlowDynamics estimator(config);
  estimator.processInput(frame(1000000000));
  const auto result = estimator.processInput(frame(1050000000));
  EXPECT_EQ(cv::countNonZero(result->observation_count), 0);
  EXPECT_TRUE(result->failure_flags.at<uint16_t>(48, 64) & kLowTexture);
  EXPECT_EQ(result->track_alive.at<uint8_t>(48, 64), 0);
  EXPECT_TRUE(std::isnan(result->probability.at<float>(48, 64)));
}

TEST_F(OpticalFlowDynamicsTest, WindowLimitsGapsAndRewindsResetReference) {
  config.window_size = 2;
  OpticalFlowDynamics estimator(config);
  estimator.processInput(frame(1000000000));
  EXPECT_EQ(estimator.processInput(frame(1050000000))->num_frames, 2);
  auto result = estimator.processInput(frame(1100000000));
  EXPECT_EQ(result->reference_timestamp_ns, 1100000000u);
  EXPECT_EQ(result->num_frames, 1);
  result = estimator.processInput(frame(1500000000));
  EXPECT_EQ(result->reference_timestamp_ns, 1500000000u);
  result = estimator.processInput(frame(1400000000));
  EXPECT_EQ(result->reference_timestamp_ns, 1400000000u);
  EXPECT_EQ(result->num_frames, 1);
  result = estimator.processInput(frame(1400000000));
  EXPECT_EQ(result->num_frames, 1);

  config.window_size = 10;
  config.max_window_duration = 0.075;
  OpticalFlowDynamics bounded(config);
  bounded.processInput(frame(1000000000));
  bounded.processInput(frame(1050000000));
  EXPECT_EQ(bounded.processInput(frame(1100000000))->reference_timestamp_ns, 1100000000u);
}

TEST_F(OpticalFlowDynamicsTest, MalformedInputResetsHistory) {
  OpticalFlowDynamics estimator(config);
  estimator.processInput(frame(1000000000));
  auto bad = frame(1050000000);
  bad.depth_image = cv::Mat(5, 5, CV_16UC1);
  EXPECT_FALSE(estimator.processInput(bad));
  EXPECT_EQ(estimator.processInput(frame(1100000000))->num_frames, 1);
  bad = frame(1150000000);
  bad.world_T_body.translation().x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(estimator.processInput(bad));
}

TEST_F(OpticalFlowDynamicsTest, GreaterDepthUncertaintyLowersEvidence) {
  auto low_config = config;
  low_config.depth_sigma = 0.01;
  auto high_config = config;
  high_config.depth_sigma = 0.1;
  OpticalFlowDynamics low(low_config), high(high_config);
  low.processInput(frame(1000000000));
  high.processInput(frame(1000000000));
  const auto a = low.processInput(frame(1050000000, 0.f, 0.0, 2.05f));
  const auto b = high.processInput(frame(1050000000, 0.f, 0.0, 2.05f));
  EXPECT_GT(a->squared_error.at<double>(48, 64), b->squared_error.at<double>(48, 64));
  EXPECT_GT(a->probability.at<float>(48, 64), b->probability.at<float>(48, 64));
  EXPECT_LT(a->innovation_sigma.at<float>(48, 64), b->innovation_sigma.at<float>(48, 64));
  EXPECT_EQ(a->reference_points.at<cv::Vec3f>(48, 64), b->reference_points.at<cv::Vec3f>(48, 64));
}

TEST_F(OpticalFlowDynamicsTest, DepthNoiseScalesWithRange) {
  config.depth_sigma = 0.005;
  config.depth_sigma_scale = 0.01;
  double previous = std::numeric_limits<double>::infinity();
  for (float z : {2.f, 4.f}) {
    OpticalFlowDynamics estimator(config);
    estimator.processInput(frame(1000000000, 0.f, 0.0, z));
    const float observed_z = z + 0.05f;
    const auto result = estimator.processInput(frame(1050000000, 0.f, 0.0, observed_z));
    const double sigma0 = config.depth_sigma + config.depth_sigma_scale * z * z;
    const double sigma1 = config.depth_sigma + config.depth_sigma_scale * observed_z * observed_z;
    const double dz = observed_z - z;
    const double expected = dz * dz / (sigma0 * sigma0 + sigma1 * sigma1);
    const double statistic = result->squared_error.at<double>(48, 64);
    EXPECT_NEAR(statistic, expected, 1.e-3);
    EXPECT_LT(statistic, previous);
    previous = statistic;
  }
}

TEST_F(OpticalFlowDynamicsTest, ConfigurationValidationAndFactory) {
  EXPECT_TRUE(config::isValid(config));
  auto bad = config;
  bad.depth_sigma = 0.0;
  EXPECT_FALSE(config::isValid(bad));
  bad = config;
  bad.max_frame_gap = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(config::isValid(bad));
  bad = config;
  bad.min_observations = bad.window_size;
  EXPECT_FALSE(config::isValid(bad));
  const auto parsed = config::fromYaml<config::VirtualConfig<DynamicsEstimator>>(
      YAML::Load("{type: OpticalFlowDynamics, scoring_mode: legacy, window_size: 3, min_observations: 1}"));
  const auto estimator = parsed.create();
  ASSERT_TRUE(estimator);
  EXPECT_EQ(estimator->processInput(frame(1000000000))->num_frames, 1);
}

TEST_F(OpticalFlowDynamicsTest, TemporalCoherentMotionAndReferenceGeometry) {
  config.scoring_mode = "temporal";
  config.debug_output = true;
  config.min_temporal_span = 0.1;
  OpticalFlowDynamics estimator(config);
  const auto reference = estimator.processInput(frame(1000000000));
  DynamicsEstimate::ConstPtr result;
  for (int i = 1; i < 5; ++i) {
    result = estimator.processInput(frame(1000000000 + i * 50000000, i * 2.5f));
    if (i < 3) {
      EXPECT_TRUE(std::isnan(result->motion_evidence.at<float>(48, 64)));
    }
  }
  ASSERT_TRUE(result);
  EXPECT_EQ(result->accepted_count.at<int>(48, 64), 5);
  EXPECT_NEAR(result->temporal_span.at<float>(48, 64), 0.2, 1.e-6);
  EXPECT_GT(result->motion_evidence.at<float>(48, 64), 0.5);
  EXPECT_NEAR(result->velocity.at<cv::Vec3f>(48, 64)[0], 1.0, 0.1);
  EXPECT_EQ(result->reference_points.at<cv::Vec3f>(48, 64), reference->reference_points.at<cv::Vec3f>(48, 64));
  EXPECT_TRUE(std::isnan(result->probability.at<float>(48, 64)));
  EXPECT_LT(result->dynamic_cost.at<float>(48, 64), result->static_cost.at<float>(48, 64));
  EXPECT_NEAR(result->latest_timestamp_ns, 1200000000u, 1);
}

TEST_F(OpticalFlowDynamicsTest, TemporalMissingFinalObservationKeepsEarlierEvidence) {
  config.scoring_mode = "temporal";
  config.ambiguity_check = false;  // Border patches are untestable; this test isolates retention.
  // A centre pixel loses depth; a pixel near the border leaves the image.
  for (const int u : {64, 118}) {
    OpticalFlowDynamics estimator(config);
    DynamicsEstimate::ConstPtr result;
    for (int i = 0; i < 4; ++i) {
      result = estimator.processInput(frame(1000000000 + i * 50000000, i * 2.5f));
    }
    const float before = result->motion_evidence.at<float>(48, u);
    ASSERT_GT(before, 0.5);
    auto last = frame(1200000000, 10.f);
    last.depth_image(cv::Rect(70, 44, 9, 9)).setTo(std::numeric_limits<float>::quiet_NaN());
    result = estimator.processInput(last);
    // Missing evidence neither erases nor replaces the fit of accepted observations.
    EXPECT_EQ(result->accepted_count.at<int>(48, u), 4);
    EXPECT_EQ(result->motion_evidence.at<float>(48, u), before);
    EXPECT_EQ(result->validity.at<uint8_t>(48, u), static_cast<uint8_t>(DynamicsValidity::kDynamic));
    EXPECT_EQ(result->failure_count.at<int>(48, u), 1);
    EXPECT_EQ(result->track_alive.at<uint8_t>(48, u), u == 64 ? 255 : 0);
    EXPECT_TRUE(result->failure_flags.at<uint16_t>(48, u) & (u == 64 ? kInvalidDepth : kOutsideImage));
  }
}

TEST_F(OpticalFlowDynamicsTest, TemporalJumpAndPoorFitsRemainUnknown) {
  config.scoring_mode = "temporal";
  config.debug_output = true;
  for (const auto& depths : {std::vector<float>{2, 2, 2.2, 2, 2},
                             std::vector<float>{2, 2.2, 1.9, 2.3, 1.8}}) {
    OpticalFlowDynamics estimator(config);
    DynamicsEstimate::ConstPtr result;
    for (int i = 0; i < 5; ++i) result = estimator.processInput(frame(1000000000 + i * 50000000, 0, 0, depths[i]));
    EXPECT_EQ(result->accepted_count.at<int>(48, 64), 5);
    EXPECT_TRUE(std::isnan(result->motion_evidence.at<float>(48, 64)));
    EXPECT_EQ(result->validity.at<uint8_t>(48, 64), static_cast<uint8_t>(DynamicsValidity::kUnreliable));
    EXPECT_TRUE(result->failure_flags.at<uint16_t>(48, 64) & kPoorModelFit);
    EXPECT_FALSE(result->failure_flags.at<uint16_t>(48, 64) & kDepthDiscontinuity);
  }
}

TEST_F(OpticalFlowDynamicsTest, TemporalStaticCameraMotionAndSoftTexture) {
  config.scoring_mode = "temporal";
  OpticalFlowDynamics estimator(config);
  DynamicsEstimate::ConstPtr result;
  for (int i = 0; i < 5; ++i) result = estimator.processInput(frame(1000000000 + i * 50000000, -i * 2.f, i * 0.04));
  EXPECT_LT(result->motion_evidence.at<float>(48, 64), 0.05);
  color.setTo(cv::Scalar::all(100));
  OpticalFlowDynamics weak(config);
  for (int i = 0; i < 5; ++i) result = weak.processInput(frame(1000000000 + i * 50000000));
  EXPECT_EQ(result->accepted_count.at<int>(48, 64), 5);
  EXPECT_TRUE(result->failure_flags.at<uint16_t>(48, 64) & kLowTexture);
  EXPECT_LT(result->reliability.at<float>(48, 64), 0.5);
  EXPECT_GT(result->reliability.at<float>(48, 64), 0.25);
  EXPECT_EQ(result->motion_evidence.at<float>(48, 64), 0.f);
}

TEST_F(OpticalFlowDynamicsTest, TemporalDepthLayersMissingSamplesAndTimestampReset) {
  config.scoring_mode = "temporal";
  OpticalFlowDynamics estimator(config);
  auto reference = frame(1000000000);
  reference.depth_image.at<float>(48, 65) = 4.f;
  const auto first = estimator.processInput(reference);
  EXPECT_TRUE(first->failure_flags.at<uint16_t>(48, 64) & kDepthDiscontinuity);
  EXPECT_EQ(first->accepted_count.at<int>(48, 64), 0);
  estimator.reset();
  estimator.processInput(frame(1000000000));
  auto missing = frame(1050000000);
  missing.depth_image(cv::Rect(60, 44, 9, 9)).setTo(std::numeric_limits<float>::quiet_NaN());
  auto result = estimator.processInput(missing);
  EXPECT_EQ(result->track_alive.at<uint8_t>(48, 64), 255);
  for (int i = 2; i < 5; ++i) result = estimator.processInput(frame(1000000000 + i * 50000000));
  EXPECT_EQ(result->accepted_count.at<int>(48, 64), 4);
  EXPECT_EQ(result->motion_evidence.at<float>(48, 64), 0.f);
  result = estimator.processInput(frame(1200000000));
  EXPECT_EQ(result->accepted_count.at<int>(48, 64), 1);
  EXPECT_TRUE(std::isnan(result->motion_evidence.at<float>(48, 64)));
}


TEST_F(OpticalFlowDynamicsTest, TemporalUncertaintyAndContinuousTexturedSurface) {
  config.scoring_mode = "temporal";
  auto high_config = config;
  high_config.position_sigma_floor = 0.2;
  OpticalFlowDynamics low(config), high(high_config);
  DynamicsEstimate::ConstPtr a, b;
  for (int i = 0; i < 5; ++i) {
    const auto input = frame(1000000000 + i * 50000000, 0, 0, 2 + i * 0.05f);
    a = low.processInput(input);
    b = high.processInput(input);
  }
  EXPECT_GT(a->motion_evidence.at<float>(48, 64), b->motion_evidence.at<float>(48, 64));
  color.colRange(0, 64).setTo(cv::Scalar::all(20));
  color.colRange(64, 128).setTo(cv::Scalar::all(200));
  OpticalFlowDynamics edge(config);
  for (int i = 0; i < 5; ++i) {
    auto input = frame(1000000000 + i * 50000000);
    for (int v = 0; v < input.depth_image.rows; ++v) for (int u = 0; u < input.depth_image.cols; ++u) {
      input.depth_image.at<float>(v, u) = 1.f + 0.04f * u;
    }
    input.vertex_map = camera->computeVertexMap(input.depth_image);
    a = edge.processInput(input);
  }
  EXPECT_FALSE(a->failure_flags.at<uint16_t>(48, 64) & kDepthDiscontinuity);
  EXPECT_EQ(a->accepted_count.at<int>(48, 64), 5);
}

TEST_F(OpticalFlowDynamicsTest, StructureTensorUsesMinTextureUnits) {
  cv::Mat gray, expected;
  cv::cvtColor(color, gray, cv::COLOR_RGB2GRAY);
  cv::cornerMinEigenVal(gray, expected, 3);
  const cv::Mat tensor = structureTensor(gray, 3);
  for (const auto& p : {cv::Point(10, 10), cv::Point(64, 48), cv::Point(100, 70)}) {
    const auto g = tensor.at<cv::Vec3f>(p);
    const double half_trace = 0.5 * (g[0] + g[2]);
    const double min_eigenvalue =
        half_trace - std::sqrt(0.25 * (g[0] - g[2]) * (g[0] - g[2]) + g[1] * g[1]);
    EXPECT_NEAR(min_eigenvalue, expected.at<float>(p), 1.e-6 + 1.e-4 * expected.at<float>(p));
  }
}

TEST_F(OpticalFlowDynamicsTest, ApertureProblemDoesNotBecomeCoherentMotion) {
  // Horizontal stripes: a static wall seen by a sideways-moving camera has no observable
  // horizontal flow, so tracked pixels stay put and appear to move with the camera.
  for (int v = 0; v < color.rows; ++v) {
    color.row(v).setTo(cv::Scalar::all(128 + 60 * std::sin(2 * M_PI * v / 12.0)));
  }
  config.scoring_mode = "temporal";
  auto naive = config, aperture = config, ambiguity = config;
  naive.aperture_sigma = 0.0;
  naive.ambiguity_check = false;
  aperture.ambiguity_check = false;
  ambiguity.aperture_sigma = 0.0;
  std::vector<OpticalFlowDynamics> estimators{OpticalFlowDynamics(naive), OpticalFlowDynamics(aperture),
                                              OpticalFlowDynamics(ambiguity), OpticalFlowDynamics(config)};
  std::vector<DynamicsEstimate::ConstPtr> results(estimators.size());
  for (int i = 0; i < 5; ++i) {
    const auto input = frame(1000000000 + i * 50000000, -i * 2.f, i * 0.04);
    for (size_t k = 0; k < estimators.size(); ++k) results[k] = estimators[k].processInput(input);
  }
  // Without either safeguard the stripe is wrongly reported as coherent motion...
  ASSERT_EQ(results[0]->accepted_count.at<int>(48, 64), 5);
  EXPECT_EQ(results[0]->validity.at<uint8_t>(48, 64), static_cast<uint8_t>(DynamicsValidity::kDynamic));
  // ...each safeguard alone, and both together, prevent confident motion evidence.
  for (size_t k = 1; k < results.size(); ++k) {
    EXPECT_NE(results[k]->validity.at<uint8_t>(48, 64), static_cast<uint8_t>(DynamicsValidity::kDynamic)) << k;
    EXPECT_FALSE(results[k]->motion_evidence.at<float>(48, 64) > 0.f) << k;
  }
  // The static prediction explains the image, so those observations are missing evidence.
  EXPECT_TRUE(results[2]->failure_flags.at<uint16_t>(48, 64) & kAmbiguousCorrespondence);
  EXPECT_EQ(results[2]->accepted_count.at<int>(48, 64), 1);
  EXPECT_EQ(results[2]->track_alive.at<uint8_t>(48, 64), 255);
}

TEST_F(OpticalFlowDynamicsTest, UntestableBorderCorrespondencesAreAmbiguous) {
  config.scoring_mode = "temporal";
  OpticalFlowDynamics estimator(config);
  DynamicsEstimate::ConstPtr result;
  for (int i = 0; i < 5; ++i) result = estimator.processInput(frame(1000000000 + i * 50000000, i * 2.5f));
  // Near the border the 5x5 patch leaves the image, so the flow cannot be verified.
  EXPECT_TRUE(result->failure_flags.at<uint16_t>(48, 120) & kAmbiguousCorrespondence);
  EXPECT_LT(result->accepted_count.at<int>(48, 120), config.min_fit_observations);
  EXPECT_NE(result->validity.at<uint8_t>(48, 120), static_cast<uint8_t>(DynamicsValidity::kDynamic));
  // Away from the border the same motion stays verifiable.
  EXPECT_FALSE(result->failure_flags.at<uint16_t>(48, 64) & kAmbiguousCorrespondence);
}

TEST_F(OpticalFlowDynamicsTest, AmbiguityCheckKeepsDistinguishableMotion) {
  // Textured surface moving against a static camera: the static prediction does not match.
  config.scoring_mode = "temporal";
  OpticalFlowDynamics estimator(config);
  DynamicsEstimate::ConstPtr result;
  for (int i = 0; i < 5; ++i) result = estimator.processInput(frame(1000000000 + i * 50000000, i * 2.5f));
  EXPECT_FALSE(result->failure_flags.at<uint16_t>(48, 64) & kAmbiguousCorrespondence);
  EXPECT_EQ(result->validity.at<uint8_t>(48, 64), static_cast<uint8_t>(DynamicsValidity::kDynamic));
}

TEST(SpatialDynamicsSupport, BoundedSinglePassDepthColorAndDirectEvidence) {
  OpticalFlowDynamics::Config config;
  config.spatial_enabled = true;
  DynamicsEstimate estimate;
  const cv::Size size(15, 9);
  estimate.reference_points = cv::Mat(size, CV_32FC3, cv::Scalar(0, 0, 2));
  estimate.motion_evidence = cv::Mat(size, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
  estimate.reliability = cv::Mat::zeros(size, CV_32FC1);
  estimate.spatial_support = cv::Mat::zeros(size, CV_8UC1);
  estimate.velocity = cv::Mat(size, CV_32FC3, cv::Scalar(1, 0, 0));
  cv::Mat colors(size, CV_8UC3, cv::Scalar(100, 100, 100));
  cv::Mat depths(size, CV_32FC1, cv::Scalar(2));
  for (int v = 1; v < 8; ++v) for (int u = 1; u < 6; ++u) {
    estimate.motion_evidence.at<float>(v, u) = 0.9;
    estimate.reliability.at<float>(v, u) = 1;
  }
  estimate.motion_evidence.at<float>(4, 3) = std::numeric_limits<float>::quiet_NaN();
  estimate.reliability.at<float>(4, 3) = 0.1;
  estimate.motion_evidence.at<float>(2, 3) = 0.f;
  depths.colRange(6, 15).setTo(4);
  applySpatialSupport(estimate, colors, depths, config);
  EXPECT_GT(estimate.regularized_evidence.at<float>(4, 3), 0.5);
  EXPECT_EQ(estimate.spatial_support.at<uint8_t>(4, 3), 2);
  EXPECT_EQ(estimate.reliability.at<float>(4, 3), 0.1f);
  EXPECT_TRUE(std::isnan(estimate.motion_evidence.at<float>(4, 3)));
  EXPECT_EQ(estimate.regularized_evidence.at<float>(2, 3), 0.f);
  EXPECT_TRUE(std::isnan(estimate.regularized_evidence.at<float>(4, 6)));
  EXPECT_TRUE(std::isnan(estimate.regularized_evidence.at<float>(4, 12)));
  colors.at<cv::Vec3b>(4, 3) = cv::Vec3b(250, 250, 250);
  applySpatialSupport(estimate, colors, depths, config);
  EXPECT_TRUE(std::isnan(estimate.regularized_evidence.at<float>(4, 3)));
  config.spatial_enabled = false;
  applySpatialSupport(estimate, colors, depths, config);
  EXPECT_TRUE(std::isnan(estimate.regularized_evidence.at<float>(4, 3)));
}

}  // namespace
}  // namespace khronos
