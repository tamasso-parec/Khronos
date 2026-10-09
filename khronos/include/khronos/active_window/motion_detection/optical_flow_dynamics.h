#pragma once

#include <string>
#include <vector>

#include <config_utilities/config_utilities.h>
#include <hydra/input/camera.h>

#include "khronos/active_window/motion_detection/dynamics_estimator.h"
#include "khronos/active_window/motion_detection/motion_uncertainty.h"
#include "khronos/active_window/motion_detection/temporal_motion_model.h"

namespace khronos {

class OpticalFlowDynamics : public DynamicsEstimator {
 public:
  struct Config {
    std::string scoring_mode = "temporal";  // "legacy" preserves the previous reference scorer.
    bool debug_output = false;
    int window_size = 5;  // Includes the reference frame.
    int min_observations = 2;
    double max_frame_gap = 0.2;  // Seconds between successive frames.
    double max_window_duration = 1.0;
    double reference_pixel_sigma = 0.5;  // Per-axis localization uncertainty [px].
    double flow_sigma = 0.5;             // Per-axis uncertainty added by each flow step [px].
    double depth_sigma = 0.01;           // Depth standard-deviation floor [m].
    double depth_sigma_scale = 0.001;    // sigma_z(z) = depth_sigma + depth_sigma_scale*z*z [1/m].
    double position_sigma_floor = 0.001;  // Unmodeled 3D measurement floor [m], temporal only.
    double max_forward_backward_error = 1.0;  // Pixels.
    double max_depth_jump = 0.1;  // Meters across a bilinear depth sampling footprint.
    double min_texture = 1.e-4;   // Minimum image structure tensor eigenvalue.
    double depth_jump_relative = 0.02;  // Allowed local depth spread: max_depth_jump + relative*z.
    int depth_neighborhood_radius = 1;
    double texture_variance_scale = 4.0;  // Heuristic variance inflation on weak texture.
    double fb_variance_scale = 1.0;       // Heuristic inflation from flow cycle error.
    // Aperture heuristic: flow std [px] added along an image direction whose structure-tensor
    // eigenvalue equals min_texture; scales as 1/sqrt(eigenvalue), capped at 10x. 0 disables.
    double aperture_sigma = 1.0;
    // Skip an observation as missing evidence when the static-predicted pixel, more than
    // ambiguity_min_separation px from the flow endpoint, matches the reference 5x5 patch
    // within photometric_tolerance (mean absolute 8-bit gray difference) of the flow endpoint.
    bool ambiguity_check = true;
    double ambiguity_min_separation = 1.0;
    double photometric_tolerance = 3.0;
    // Depth at the static prediction (3x3, depth-layer tolerance): all nearer = occluded
    // (observation skipped), all farther = free space (a static point cannot be seen through).
    // Free space marks a pixel moving with k violations when, a contrario,
    // N_pixels * P(Binomial(window_size - 1, free_space_false_rate) >= k) <= free_space_max_false_alarms.
    bool visibility_check = true;
    double free_space_false_rate = 1.e-3;   // Per test under the static hypothesis.
    double free_space_max_false_alarms = 1.0;  // Expected false detections per image.
    double free_space_log_odds = 1.5;       // CRF unary per violation [nats], capped.
    int min_fit_observations = 4;         // Includes the reference; at least three are required.
    double min_temporal_span = 0.1;       // Seconds.
    double min_reliability = 0.25;
    double max_fit_cdf = 0.99;            // Approximate fit-quality gate.
    double motion_penalty = 11.345;       // LR-inspired complexity penalty for three parameters.
    double motion_score_scale = 10.0;
    double min_displacement = 0.02;       // Meters over the accepted temporal span.
    double min_displacement_sigma = 3.0;
    bool spatial_enabled = false;
    int spatial_radius = 2;               // At most three pixels; one pass, no diffusion.
    int spatial_min_neighbors = 3;
    double spatial_min_reliability = 0.6;
    double spatial_color_sigma = 30.0;    // RGB8 Euclidean difference.
    double spatial_velocity_tolerance = 1.0;  // m/s, local neighbor compatibility.
    double spatial_motion_fraction = 0.7;
    // "support": bounded one-pass borrowing above. "crf": depth-aware mean-field CRF whose
    // per-pixel marginal (0.5 = undecided) replaces regularized_evidence.
    std::string spatial_method = "support";
    int crf_iterations = 5;
    double crf_pairwise_weight = 3.0;   // Max neighborhood log-odds contribution [nats].
    double crf_max_unary = 4.0;         // Cap on a pixel's own log-odds [nats].
    double crf_spatial_sigma = 4.0;     // px, stencil distance falloff.
    double crf_min_support = 0.2;       // Min confident, connected weight for unknown pixels.
    int flow_levels = 3;
    int flow_window_size = 15;
    int flow_iterations = 3;
  };

  explicit OpticalFlowDynamics(const Config& config);

  // Input poses are taken as exact; supply ground-truth body poses and calibrated extrinsics.
  // Returns an independent snapshot, including partial windows. Invalid inputs reset history
  // and return nullptr. Windows restart on size/time limits, sensor changes or timestamp rewind.
  DynamicsEstimate::ConstPtr processInput(const hydra::InputData& input) override;
  void reset() override;

  const Config config;

 private:
  void startWindow(const hydra::InputData& input, const cv::Mat& gray);
  void track(const hydra::InputData& input, const cv::Mat& gray);
  void scoreTemporal(int v, int u);
  void regularizeSpatial();
  void computeCrfWeights();
  void applyFreeSpace();
  DynamicsEstimate::ConstPtr snapshot() const;

  DynamicsEstimate estimate_;
  cv::Mat previous_gray_;
  cv::Mat tracked_pixels_;  // CV_32FC2, subpixel locations in the previous image.
  cv::Mat previous_texture_;
  cv::Mat previous_tensor_;  // CV_32FC3 structure tensor (xx, xy, yy), temporal mode only.
  cv::Mat reference_gray_;
  cv::Mat crf_unary_;              // CV_32FC1 signed log-odds from each pixel's own fit.
  std::vector<float> crf_weights_;  // Per pixel and stencil offset, fixed per window.
  std::vector<double> fit_thresholds_;  // Chi-square max_fit_cdf quantile per DOF.
  int free_space_min_count_ = 0;  // A-contrario threshold for the current image size.
  std::vector<MotionUncertainty> uncertainties_;
  std::vector<TemporalMotionModel> temporal_models_;
  std::vector<Eigen::Matrix2d> pixel_covariances_;
  cv::Mat reference_color_, reference_depth_;
  hydra::Camera::Config camera_config_;
  std::string sensor_name_;
};

void declare_config(OpticalFlowDynamics::Config& config);
// Window-averaged image structure tensor in cornerMinEigenVal units (3x3 Sobel, 3x3 block sum).
cv::Mat structureTensor(const cv::Mat& gray, int window);
// Mean-field CRF on the reference grid; weights are laid out [pixel][kCrfStencilSize].
constexpr int kCrfStencilSize = 24;
std::vector<float> crfWeights(const cv::Mat& reference_color, const cv::Mat& reference_depth,
                              const OpticalFlowDynamics::Config& config);
void applyCrfSupport(DynamicsEstimate& estimate, const cv::Mat& unary,
                     const std::vector<float>& weights, const OpticalFlowDynamics::Config& config);
void applySpatialSupport(DynamicsEstimate& estimate, const cv::Mat& reference_color,
                         const cv::Mat& reference_depth, const OpticalFlowDynamics::Config& config);

}  // namespace khronos
