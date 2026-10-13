#include "khronos/active_window/motion_detection/optical_flow_dynamics.h"

#include <cstring>

#include <config_utilities/parsing/yaml.h>
#include <gtest/gtest.h>
#include <hydra/input/camera.h>
#include <hydra/input/sensor_extrinsics.h>
#include <opencv2/imgproc.hpp>

namespace khronos {
namespace {

std::shared_ptr<hydra::Camera> testCamera() {
  hydra::Camera::Config config;
  config.width = 128;
  config.height = 96;
  config.fx = config.fy = 100.f;
  config.cx = 64.f;
  config.cy = 48.f;
  config.min_range = 0.1;
  config.max_range = 10.0;
  hydra::ParamSensorExtrinsics::Config extrinsics;
  extrinsics.body_R_sensor = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ());
  extrinsics.body_p_sensor = Eigen::Vector3d(0.2, -0.1, 0.3);
  config.extrinsics = extrinsics;
  return std::make_shared<hydra::Camera>(config, "rgbd");
}

hydra::InputData cameraInput(const std::shared_ptr<hydra::Camera>& camera,
                             uint64_t stamp,
                             float image_shift,
                             double camera_x,
                             bool world_points = false) {
  cv::Mat noise(96, 128, CV_8UC1), color;
  cv::RNG rng(42);
  rng.fill(noise, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(noise, noise, cv::Size(3, 3), 0.6);
  cv::cvtColor(noise, color, cv::COLOR_GRAY2RGB);
  hydra::InputData input(camera);
  input.timestamp_ns = stamp;
  Eigen::Isometry3d world_T_sensor = Eigen::Isometry3d::Identity();
  world_T_sensor.translation().x() = camera_x;
  input.world_T_body = world_T_sensor * camera->body_T_sensor().inverse();
  const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, image_shift, 0, 1, 0);
  cv::warpAffine(color, input.color_image, shift, color.size(), cv::INTER_LINEAR,
                 cv::BORDER_REFLECT);
  input.depth_image = cv::Mat(color.size(), CV_32FC1, cv::Scalar(2.f));
  const Eigen::Isometry3f pose = world_T_sensor.cast<float>();
  input.vertex_map = camera->computeVertexMap(input.depth_image, world_points ? &pose : nullptr);
  input.points_in_world_frame = world_points;
  return input;
}

void expectSameImage(const cv::Mat& a, const cv::Mat& b) {
  ASSERT_EQ(a.size(), b.size());
  ASSERT_EQ(a.type(), b.type());
  for (int row = 0; row < a.rows; ++row) {
    EXPECT_EQ(std::memcmp(a.ptr(row), b.ptr(row), a.cols * a.elemSize()), 0);
  }
}

TEST(OpticalFlowDynamicsAdapter, ConvertsHydraCameraInputs) {
  const auto camera = testCamera();
  const auto input = cameraInput(camera, 1000000000, 0.f, 0.3, true);
  dynamic_psg::RgbdFrame frame;
  ASSERT_TRUE(toRgbdFrame(input, frame));
  EXPECT_EQ(frame.timestamp_ns, input.timestamp_ns);
  EXPECT_EQ(frame.sensor_name, "rgbd");
  EXPECT_EQ(frame.camera.width, 128);
  EXPECT_EQ(frame.camera.fx, 100.f);
  EXPECT_EQ(frame.camera.cy, 48.f);
  EXPECT_EQ(frame.camera.min_range, 0.1f);
  EXPECT_EQ(frame.camera.max_range, 10.f);
  // The sensor pose includes the camera extrinsics.
  EXPECT_TRUE(frame.world_T_camera.isApprox(input.getSensorPose()));
  EXPECT_NEAR(frame.world_T_camera.translation().x(), 0.3, 1.e-12);
  EXPECT_EQ(frame.depth.data, input.depth_image.data);  // Shared, not copied.
  EXPECT_EQ(frame.vertex_map.data, input.vertex_map.data);
  EXPECT_TRUE(frame.points_in_world_frame);
}

TEST(OpticalFlowDynamicsAdapter, FactoryMatchesDirectEstimator) {
  const auto adapter = config::fromYaml<config::VirtualConfig<DynamicsEstimator>>(
                           YAML::Load("{type: OpticalFlowDynamics, spatial_enabled: true, "
                                      "spatial_method: crf}"))
                           .create();
  ASSERT_TRUE(adapter);
  dynamic_psg::OpticalFlowDynamics::Config config;
  config.spatial_enabled = true;
  config.spatial_method = "crf";
  dynamic_psg::OpticalFlowDynamics direct(config);
  const auto camera = testCamera();
  for (int i = 0; i < 6; ++i) {
    SCOPED_TRACE(i);
    const auto input = cameraInput(camera, 1000000000 + i * 50000000, i * 2.5f, 0.01 * i);
    const auto a = adapter->processInput(input);
    dynamic_psg::RgbdFrame frame;
    ASSERT_TRUE(toRgbdFrame(input, frame));
    const auto b = direct.process(frame);
    ASSERT_TRUE(a && b);
    EXPECT_EQ(a->reference_timestamp_ns, b->reference_timestamp_ns);
    expectSameImage(a->reference_points, b->reference_points);
    expectSameImage(a->motion_evidence, b->motion_evidence);
    expectSameImage(a->regularized_evidence, b->regularized_evidence);
    expectSameImage(a->validity, b->validity);
    expectSameImage(a->failure_flags, b->failure_flags);
  }
}

}  // namespace
}  // namespace khronos
