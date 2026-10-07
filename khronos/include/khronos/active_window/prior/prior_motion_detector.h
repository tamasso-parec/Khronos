#pragma once

#include <unordered_map>

#include <config_utilities/config_utilities.h>
#include <hydra/common/global_info.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/active_window/prior/object_prior_store.h"
#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Detects when a verified object prior starts to move, and turns the evidence into inputs
 * for the free-space motion detector:
 * - Onset: the absent fraction of a prior rises clearly above its running baseline and the absent
 *   pixels form a connected patch, i.e. surface the object covered is now exposed.
 * - Seeds: current pixels of the same label close to a moving prior are marked as motion seeds.
 * - Suppression: pixels matching a confidently static prior are marked to suppress free-space
 *   seeds, e.g. caused by pose drift.
 */
class PriorMotionDetector {
 public:
  // Config.
  struct Config {
    int verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;

    // Frames with fewer present + absent pixels for a prior are ignored.
    int min_informative_pixels = 100;

    // Number of informative frames to establish the baseline before onset can be detected.
    int min_baseline_frames = 5;

    // Update rate of the exponential running baseline of the absent fraction.
    float baseline_rate = 0.2f;

    // Onset requires the absent fraction to be at least this and to exceed the baseline by this.
    // A sideways shift by a fraction s of the object's width exposes about s of its surface.
    float min_absent_fraction = 0.15f;
    float min_absent_increase = 0.15f;

    // Priors whose baseline absent fraction is above this do not explain the measurement well
    // enough while static (e.g. thin structures) and are not used for onset detection.
    float max_baseline = 1.f;

    // Onset requires a connected patch of absent pixels of at least this size.
    int min_absent_component = 150;

    // Number of consecutive onset frames before a prior is considered moving.
    int min_onset_frames = 2;

    // Seed current pixels of moving priors.
    bool enable_seeds = true;

    // Seeds are searched within the prior's bounding box inflated by this in meters.
    float search_margin = 0.5f;

    // Seeds are only created if at least this many candidate pixels are displaced, i.e. not
    // explained by the prior. Absence without displaced pixels is a removal, not a motion, and
    // absence while the object still matches its prior is model noise.
    int min_displaced_pixels = 300;

    // Only count displaced pixels whose point lies in a voxel previously observed as free. Newly
    // seen parts of a partially modeled object lie in unobserved space and are not counted.
    bool displacement_requires_free_space = true;

    // Only seed pixels whose semantic label matches the prior. If no label image is available, no
    // seeds are created.
    bool require_label_match = true;

    // Suppress free-space seeds on pixels matching a confidently static prior.
    bool enable_suppression = true;

    // Priors with a higher absent fraction in the current frame do not suppress.
    float max_suppression_absent_fraction = 0.1f;

    // Remove the state of priors not seen for this long in seconds.
    float state_timeout = 30.f;
  } const config;

  // Construction.
  explicit PriorMotionDetector(const Config& config);
  virtual ~PriorMotionDetector() = default;

  /**
   * @brief Update the motion state of all verified priors and write the seed and suppression
   * masks. Requires the prior verifier to have run on this frame.
   * @param priors The priors that were verified.
   * @param data Frame data to read the evidence from and write the results to.
   * @param tracking_layer Free space history of the active window. If null, displaced pixels are
   * counted regardless of free space.
   */
  void processInput(const ObjectPriorPtrs& priors,
                    FrameData& data,
                    const TrackingLayer* tracking_layer = nullptr);

 private:
  struct State {
    float baseline = 0.f;
    int baseline_frames = 0;
    int onset_frames = 0;
    TimeStamp last_seen = 0;
  };
  std::unordered_map<uint64_t, State> states_;

  void updateState(PriorObservation& obs, State& state) const;
  int seedPixels(const ObjectPrior& prior,
                 int prior_index,
                 const TrackingLayer* tracking_layer,
                 FrameData& data) const;
  void suppressPixels(FrameData& data) const;
};

void declare_config(PriorMotionDetector::Config& config);

}  // namespace khronos
