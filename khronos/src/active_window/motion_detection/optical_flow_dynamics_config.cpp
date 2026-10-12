#include <cmath>

#include "khronos/active_window/motion_detection/optical_flow_dynamics.h"

namespace khronos {

void declare_config(OpticalFlowDynamics::Config& config) {
  using namespace config;
  name("OpticalFlowDynamics");
  field(config.scoring_mode, "scoring_mode");
  field(config.debug_output, "debug_output");

  // Window lifecycle.
  field(config.window_size, "window_size");
  field(config.min_observations, "min_observations");
  field(config.max_frame_gap, "max_frame_gap", "s");
  field(config.max_window_duration, "max_window_duration", "s");
  field(config.keyframe_interval, "keyframe_interval");

  // Measurement uncertainty (standard deviations).
  field(config.reference_pixel_sigma, "reference_pixel_sigma", "px");
  field(config.flow_sigma, "flow_sigma", "px");
  field(config.depth_sigma, "depth_sigma", "m");
  field(config.depth_sigma_scale, "depth_sigma_scale", "1/m");
  field(config.position_sigma_floor, "position_sigma_floor");

  // Correspondence and depth-layer safeguards.
  field(config.max_forward_backward_error, "max_forward_backward_error", "px");
  field(config.max_depth_jump, "max_depth_jump", "m");
  field(config.min_texture, "min_texture");
  field(config.depth_jump_relative, "depth_jump_relative");
  field(config.depth_neighborhood_radius, "depth_neighborhood_radius");
  field(config.texture_variance_scale, "texture_variance_scale");
  field(config.fb_variance_scale, "fb_variance_scale");
  field(config.aperture_sigma, "aperture_sigma", "px");
  field(config.ambiguity_check, "ambiguity_check");
  field(config.ambiguity_min_separation, "ambiguity_min_separation", "px");
  field(config.photometric_tolerance, "photometric_tolerance", "gray levels");
  field(config.visibility_check, "visibility_check");
  field(config.free_space_false_rate, "free_space_false_rate");
  field(config.free_space_max_false_alarms, "free_space_max_false_alarms");
  field(config.free_space_log_odds, "free_space_log_odds");

  // Temporal static/velocity model comparison.
  field(config.min_fit_observations, "min_fit_observations");
  field(config.min_temporal_span, "min_temporal_span");
  field(config.min_reliability, "min_reliability");
  field(config.max_fit_cdf, "max_fit_cdf");
  field(config.motion_penalty, "motion_penalty");
  field(config.motion_score_scale, "motion_score_scale");
  field(config.min_displacement, "min_displacement");
  field(config.min_displacement_sigma, "min_displacement_sigma");

  // Optional reference-grid spatial support.
  field(config.spatial_enabled, "spatial_enabled");
  field(config.spatial_radius, "spatial_radius");
  field(config.spatial_min_neighbors, "spatial_min_neighbors");
  field(config.spatial_min_reliability, "spatial_min_reliability");
  field(config.spatial_color_sigma, "spatial_color_sigma");
  field(config.spatial_velocity_tolerance, "spatial_velocity_tolerance");
  field(config.spatial_motion_fraction, "spatial_motion_fraction");
  field(config.spatial_method, "spatial_method");
  field(config.crf_iterations, "crf_iterations");
  field(config.crf_pairwise_weight, "crf_pairwise_weight");
  field(config.crf_max_unary, "crf_max_unary");
  field(config.crf_spatial_sigma, "crf_spatial_sigma", "px");
  field(config.crf_min_support, "crf_min_support");

  // Farneback dense flow.
  field(config.flow_method, "flow_method");
  field(config.flow_levels, "flow_levels");
  field(config.flow_window_size, "flow_window_size", "px");
  field(config.flow_iterations, "flow_iterations");

  check(config.window_size, GE, 2, "window_size");
  check(config.min_observations, GE, 1, "min_observations");
  checkCondition(config.min_observations < config.window_size,
                 "min_observations must be smaller than window_size");
  check(config.max_frame_gap, GT, 0.0, "max_frame_gap");
  check(config.max_window_duration, GT, 0.0, "max_window_duration");
  check(config.keyframe_interval, GE, 1, "keyframe_interval");
  checkCondition(config.scoring_mode == "temporal" || config.keyframe_interval == 1,
                 "keyframe_interval requires temporal scoring");
  check(config.reference_pixel_sigma, GE, 0.0, "reference_pixel_sigma");
  check(config.flow_sigma, GT, 0.0, "flow_sigma");
  check(config.depth_sigma, GT, 0.0, "depth_sigma");
  check(config.depth_sigma_scale, GE, 0.0, "depth_sigma_scale");
  check(config.max_forward_backward_error, GT, 0.0, "max_forward_backward_error");
  check(config.max_depth_jump, GT, 0.0, "max_depth_jump");
  check(config.min_texture, GE, 0.0, "min_texture");
  checkCondition(std::isfinite(config.max_frame_gap) && std::isfinite(config.max_window_duration) &&
                     std::isfinite(config.reference_pixel_sigma) &&
                     std::isfinite(config.flow_sigma) && std::isfinite(config.depth_sigma) &&
                     std::isfinite(config.depth_sigma_scale) &&
                     std::isfinite(config.max_forward_backward_error) &&
                     std::isfinite(config.max_depth_jump) && std::isfinite(config.min_texture),
                 "floating-point parameters must be finite");
  checkCondition(config.flow_method == "farneback" || config.flow_method == "dis_fast" ||
                     config.flow_method == "dis_medium",
                 "flow_method must be farneback, dis_fast or dis_medium");
  check(config.flow_levels, GE, 1, "flow_levels");
  check(config.flow_window_size, GE, 3, "flow_window_size");
  checkCondition(config.flow_window_size % 2 == 1, "flow_window_size must be odd");
  check(config.flow_iterations, GE, 1, "flow_iterations");
  checkCondition(config.scoring_mode == "legacy" || config.scoring_mode == "temporal",
                 "scoring_mode must be legacy or temporal");
  check(config.position_sigma_floor, GT, 0.0, "position_sigma_floor");
  check(config.depth_jump_relative, GE, 0.0, "depth_jump_relative");
  checkInRange(config.depth_neighborhood_radius, 0, 3, "depth_neighborhood_radius");
  check(config.texture_variance_scale, GE, 0.0, "texture_variance_scale");
  check(config.fb_variance_scale, GE, 0.0, "fb_variance_scale");
  check(config.aperture_sigma, GE, 0.0, "aperture_sigma");
  check(config.ambiguity_min_separation, GE, 0.0, "ambiguity_min_separation");
  check(config.photometric_tolerance, GE, 0.0, "photometric_tolerance");
  checkInRange(config.free_space_false_rate, 0.0, 1.0, "free_space_false_rate", false, false);
  check(config.free_space_max_false_alarms, GT, 0.0, "free_space_max_false_alarms");
  check(config.free_space_log_odds, GT, 0.0, "free_space_log_odds");
  checkCondition(std::isfinite(config.free_space_max_false_alarms) &&
                     std::isfinite(config.free_space_log_odds),
                 "free-space parameters must be finite");
  check(config.min_fit_observations, GE, 3, "min_fit_observations");
  checkCondition(
      config.scoring_mode != "temporal" || config.min_fit_observations <= config.window_size,
      "temporal min_fit_observations must not exceed window_size");
  check(config.min_temporal_span, GT, 0.0, "min_temporal_span");
  checkInRange(config.min_reliability, 0.0, 1.0, "min_reliability");
  checkInRange(config.max_fit_cdf, 0.0, 1.0, "max_fit_cdf", false, false);
  check(config.motion_penalty, GE, 0.0, "motion_penalty");
  check(config.motion_score_scale, GT, 0.0, "motion_score_scale");
  check(config.min_displacement, GE, 0.0, "min_displacement");
  check(config.min_displacement_sigma, GT, 0.0, "min_displacement_sigma");
  checkInRange(config.spatial_radius, 1, 3, "spatial_radius");
  check(config.spatial_min_neighbors, GE, 2, "spatial_min_neighbors");
  checkCondition(config.spatial_min_neighbors <=
                     (2 * config.spatial_radius + 1) * (2 * config.spatial_radius + 1) - 1,
                 "spatial_min_neighbors exceeds footprint");
  checkInRange(config.spatial_min_reliability, 0.0, 1.0, "spatial_min_reliability", false, true);
  check(config.spatial_color_sigma, GT, 0.0, "spatial_color_sigma");
  check(config.spatial_velocity_tolerance, GT, 0.0, "spatial_velocity_tolerance");
  checkInRange(config.spatial_motion_fraction, 0.5, 1.0, "spatial_motion_fraction");
  checkCondition(config.spatial_method == "support" || config.spatial_method == "crf",
                 "spatial_method must be support or crf");
  checkInRange(config.crf_iterations, 1, 20, "crf_iterations");
  check(config.crf_pairwise_weight, GE, 0.0, "crf_pairwise_weight");
  check(config.crf_max_unary, GT, 0.0, "crf_max_unary");
  check(config.crf_spatial_sigma, GT, 0.0, "crf_spatial_sigma");
  checkInRange(config.crf_min_support, 0.0, 1.0, "crf_min_support");
  checkCondition(std::isfinite(config.crf_pairwise_weight) && std::isfinite(config.crf_max_unary) &&
                     std::isfinite(config.crf_spatial_sigma) &&
                     std::isfinite(config.crf_min_support),
                 "crf parameters must be finite");
  checkCondition(
      std::isfinite(config.position_sigma_floor) && std::isfinite(config.depth_jump_relative) &&
          std::isfinite(config.texture_variance_scale) && std::isfinite(config.fb_variance_scale) &&
          std::isfinite(config.aperture_sigma) && std::isfinite(config.ambiguity_min_separation) &&
          std::isfinite(config.photometric_tolerance) && std::isfinite(config.min_temporal_span) &&
          std::isfinite(config.min_reliability) && std::isfinite(config.max_fit_cdf) &&
          std::isfinite(config.motion_penalty) && std::isfinite(config.motion_score_scale) &&
          std::isfinite(config.min_displacement) && std::isfinite(config.min_displacement_sigma) &&
          std::isfinite(config.spatial_min_reliability) &&
          std::isfinite(config.spatial_color_sigma) &&
          std::isfinite(config.spatial_velocity_tolerance) &&
          std::isfinite(config.spatial_motion_fraction),
      "temporal parameters must be finite");
}

}  // namespace khronos
