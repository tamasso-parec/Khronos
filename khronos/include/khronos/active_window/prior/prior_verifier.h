#pragma once

#include <config_utilities/config_utilities.h>
#include <hydra/common/global_info.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/active_window/prior/object_prior_store.h"
#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Renders object priors into the current frame and compares the expected to
 * the measured range per pixel. Each pixel is classified as present, absent (the ray passes
 * through the expected surface), occluded, or unobserved. Results are written to the prior
 * fields of the frame data and do not affect any other module.
 */
class PriorVerifier {
 public:
  // Config.
  struct Config {
    int verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;

    // Maximum range of rendered object vertices in meters.
    float max_range = 8.f;

    // Radius of the disk each rendered vertex covers in meters. Should be about the mesh vertex
    // spacing, i.e. the map voxel size.
    float point_radius = 0.05f;

    // Maximum radius of the splat of a single vertex in pixels.
    int max_splat_radius = 6;

    // Range tolerance tau(r) = max(min_tolerance, tolerance_factor * r^2) in meters.
    float min_tolerance = 0.05f;
    float tolerance_factor = 0.01f;

    // Number of pixels to erode from the silhouette of each rendered object, in addition to the
    // splat radius at the silhouette. Silhouette pixels are most affected by pose errors and would
    // otherwise produce false absence evidence.
    int silhouette_erosion = 2;

    // Exclude pixels next to a measured range discontinuity larger than this in meters. Set to <=
    // 0 to disable.
    float range_edge_threshold = 0.2f;
  } const config;

  // Construction.
  explicit PriorVerifier(const Config& config);
  virtual ~PriorVerifier() = default;

  /**
   * @brief Render the priors into the frame and compute the per-pixel and per-object evidence.
   * @param priors Object priors to verify, in the same frame as the frame data.
   * @param data Frame data to read the measurements from and write the results to.
   */
  void processInput(const ObjectPriorPtrs& priors, FrameData& data) const;

 protected:
  // Renders the priors and returns the splat radius of the drawn vertex per pixel as 8UC1.
  cv::Mat renderPriors(const ObjectPriorPtrs& priors, FrameData& data) const;
  cv::Mat computeExclusionMask(const FrameData& data, const cv::Mat& splat_radii) const;
  void classifyPixels(const cv::Mat& exclusion_mask, FrameData& data) const;
  void computeAbsentComponents(FrameData& data) const;
  float tolerance(float range) const;
};

void declare_config(PriorVerifier::Config& config);

}  // namespace khronos
