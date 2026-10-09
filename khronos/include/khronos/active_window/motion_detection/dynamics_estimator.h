#pragma once

#include <cstdint>
#include <memory>

#include <hydra/input/input_data.h>

namespace khronos {

// Cumulative evidence flags; a failure is not itself a probability of motion.
enum DynamicsFailure : uint16_t {
  kInvalidReference = 1,
  kLowTexture = 2,
  kFlowInconsistent = 4,
  kOutsideImage = 8,
  kInvalidDepth = 16,
  kDepthDiscontinuity = 32,
  kOutOfRange = 64,
  kNonFiniteFlow = 128,
  kNumericalFailure = 256,
  kPoorModelFit = 512,
  kAmbiguousCorrespondence = 1024,  // Image cannot separate flow from the static prediction.
};

enum class DynamicsValidity : uint8_t {
  kUnavailable = 0, kInsufficient = 1, kUnreliable = 2, kStatic = 3, kDynamic = 4,
};

// All images are indexed by pixels of the reference frame, not the current frame.
struct DynamicsEstimate {
  using ConstPtr = std::shared_ptr<const DynamicsEstimate>;

  hydra::TimeStamp reference_timestamp_ns = 0;
  hydra::TimeStamp latest_timestamp_ns = 0;
  int num_frames = 0;
  bool legacy_mode = false;
  bool debug_output = false;
  cv::Mat reference_points;   // CV_32FC3, world coordinates; invalid points are NaN.
  cv::Mat probability;        // CV_32FC1, chi-square CDF; insufficient evidence is NaN.
  cv::Mat squared_error;      // CV_64FC1, joint Mahalanobis statistic (whitened innovations).
  cv::Mat observation_count;  // CV_32SC1, valid comparisons (excludes reference frame).
  cv::Mat failure_flags;      // CV_16UC1, OR of DynamicsFailure encountered in this window.
  cv::Mat failure_count;      // CV_32SC1, failed attempted comparisons (excludes reference).
  cv::Mat track_alive;        // CV_8UC1, 255 if correspondence can still be followed.
  cv::Mat forward_backward_error;  // CV_32FC1, last attempted flow cycle error [px], NaN if absent.
  cv::Mat residual_norm;      // CV_32FC1, last accepted raw 3D distance [m], NaN if absent.
  cv::Mat innovation_sigma;   // CV_32FC1, sqrt(trace(conditional covariance)/3) [m].
  cv::Mat motion_evidence;    // CV_32FC1, uncalibrated [0,1]; unavailable evidence is NaN.
  cv::Mat regularized_evidence;  // CV_32FC1, optional one-pass spatial inference.
  cv::Mat reliability;        // CV_32FC1, heuristic observation quality [0,1].
  cv::Mat validity;           // CV_8UC1, DynamicsValidity.
  cv::Mat spatial_support;    // CV_8UC1, 0 unavailable, 1 direct, 2 inferred.
  cv::Mat accepted_count;     // CV_32SC1, includes valid reference observation.
  cv::Mat temporal_span;      // CV_32FC1, actual accepted-observation baseline [s].
  cv::Mat velocity;           // CV_32FC3, world coordinates [m/s], NaN until fit available.
  // Allocated only with debug_output. Geometry above remains the initial point cloud.
  cv::Mat fitted_point;       // CV_32FC3, static statistical model parameter.
  cv::Mat static_cost, dynamic_cost, delta_cost;  // CV_32FC1, dimensionless.
  cv::Mat static_consistency, dynamic_consistency;  // CV_32FC1, approximate chi-square CDFs.
  cv::Mat velocity_covariance;  // CV_32FC(6), symmetric xx,xy,xz,yy,yz,zz [(m/s)^2].
  cv::Mat measurement_sigma;    // CV_32FC1, sqrt(trace(world point covariance)/3) [m].
};

// Independent of binary motion clustering: estimates describe an earlier point cloud.
class DynamicsEstimator {
 public:
  virtual ~DynamicsEstimator() = default;
  virtual DynamicsEstimate::ConstPtr processInput(const hydra::InputData& input) = 0;
  virtual void reset() = 0;
};

}  // namespace khronos
