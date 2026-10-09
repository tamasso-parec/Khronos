#include "khronos_ros/visualization/dynamics_visualization.h"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace khronos {

TEST(DynamicsVisualization, OrganizedCloudAliasesDebugAndUnknownValues) {
  DynamicsEstimate result;
  const cv::Size size(3, 2);
  result.reference_points = cv::Mat(size, CV_32FC3, cv::Scalar(1, 2, 3));
  result.probability = cv::Mat(size, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
  result.motion_evidence = result.probability.clone();
  result.motion_evidence.at<float>(1, 2) = 0.8;
  result.regularized_evidence = result.motion_evidence.clone();
  result.reliability = cv::Mat(size, CV_32FC1, cv::Scalar(0.9));
  result.observation_count = cv::Mat(size, CV_32SC1, cv::Scalar(3));
  result.accepted_count = cv::Mat(size, CV_32SC1, cv::Scalar(4));
  result.temporal_span = cv::Mat(size, CV_32FC1, cv::Scalar(0.15));
  result.validity = cv::Mat::zeros(size, CV_8UC1);
  result.spatial_support = cv::Mat::zeros(size, CV_8UC1);
  result.track_alive = cv::Mat(size, CV_8UC1, cv::Scalar(255));
  result.failure_flags = cv::Mat::zeros(size, CV_16UC1);
  result.failure_count = cv::Mat::zeros(size, CV_32SC1);
  std_msgs::msg::Header header;
  header.frame_id = "world";
  header.stamp.sec = 7;
  auto cloud = dynamicsCloud(result, header);
  EXPECT_EQ(cloud.height, 2u);
  EXPECT_EQ(cloud.width, 3u);
  EXPECT_EQ(cloud.point_step, 68u);
  EXPECT_EQ(cloud.row_step, 204u);
  EXPECT_EQ(cloud.data.size(), 408u);
  EXPECT_EQ(cloud.header.stamp.sec, 7);
  EXPECT_EQ(cloud.header.frame_id, "world");
  sensor_msgs::PointCloud2ConstIterator<float> legacy(cloud, "probability");
  EXPECT_TRUE(std::isnan(*legacy));
  sensor_msgs::PointCloud2ConstIterator<float> score(cloud, "motion_evidence");
  EXPECT_TRUE(std::isnan(*score));
  score += 5;
  EXPECT_FLOAT_EQ(*score, 0.8f);
  const auto image = dynamicsEvidenceImage(result);
  EXPECT_EQ(image.at<cv::Vec3b>(0, 0), cv::Vec3b(128, 128, 128));
  result.debug_output = true;
  cloud = dynamicsCloud(result, header);
  EXPECT_EQ(cloud.point_step, 152u);
  sensor_msgs::PointCloud2ConstIterator<float> velocity(cloud, "vx");
  EXPECT_TRUE(std::isnan(*velocity));
  result.legacy_mode = true;
  result.probability.setTo(0.99);
  cloud = dynamicsCloud(result, header);
  sensor_msgs::PointCloud2ConstIterator<float> probability(cloud, "probability");
  EXPECT_FLOAT_EQ(*probability, 0.99f);
  sensor_msgs::PointCloud2ConstIterator<uint32_t> mode(cloud, "scoring_mode");
  EXPECT_EQ(*mode, 0u);
}

}  // namespace khronos
