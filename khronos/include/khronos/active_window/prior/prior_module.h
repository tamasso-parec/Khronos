#pragma once

#include <config_utilities/config_utilities.h>
#include <hydra/common/global_info.h>
#include <hydra/common/shared_dsg_info.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/active_window/data/reconstruction_types.h"
#include "khronos/active_window/data/track.h"
#include "khronos/active_window/prior/object_prior_store.h"
#include "khronos/active_window/prior/prior_motion_detector.h"
#include "khronos/active_window/prior/prior_verifier.h"
#include "khronos/active_window/prior/track_model_store.h"

namespace khronos {

/**
 * @brief Verifies object priors against each frame and derives motion cues from them. Priors are
 * the static tracks in the active window and, optionally, the static objects of the scene graph
 * that have left the active window.
 *
 * Usage per frame: processInput() before motion detection, update() after tracking.
 */
class PriorModule {
 public:
  // Config.
  struct Config {
    int verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;

    // Use the static tracks in the active window as priors.
    bool use_tracks = true;

    // Use the static objects in the scene graph as priors. Requires a source to be set.
    bool use_scene_graph = true;

    // Scene graph priors overlapping a track model of the same label by more than this fraction
    // are not used, the track model is more recent.
    float track_overlap = 0.3f;

    ObjectPriorStore::Config scene_graph;
    TrackModelStore::Config track_models;
    PriorVerifier::Config verifier;
    PriorMotionDetector::Config motion;
  } const config;

  // Construction.
  explicit PriorModule(const Config& config);
  virtual ~PriorModule() = default;

  void setSceneGraphSource(hydra::SharedDsgInfo::Ptr source);

  /**
   * @brief Verify all priors in the current frame and compute the motion cues. Call before motion
   * detection.
   */
  void processInput(const VolumetricMap& map, FrameData& data);

  /**
   * @brief Update the track models with the current frame. Call after tracking.
   */
  void update(const FrameData& data, const Tracks& tracks);

 private:
  ObjectPriorStore scene_graph_;
  TrackModelStore track_models_;
  PriorVerifier verifier_;
  PriorMotionDetector motion_detector_;

  ObjectPriorPtrs collectPriors() const;
};

void declare_config(PriorModule::Config& config);

}  // namespace khronos
