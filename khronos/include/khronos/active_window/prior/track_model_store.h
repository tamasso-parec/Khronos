#pragma once

#include <limits>
#include <unordered_map>

#include <config_utilities/config_utilities.h>
#include <hydra/common/global_info.h>
#include <spatial_hash/grid.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/active_window/data/track.h"
#include "khronos/active_window/prior/object_prior_store.h"
#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Maintains a point model of every static track in the active window, accumulated from the
 * measured points of its semantic detections. The models are used as object priors to detect
 * when a static object starts to move.
 */
class TrackModelStore {
 public:
  // Config.
  struct Config {
    int verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;

    // Resolution at which model points are stored in meters. At most one point per cell.
    float resolution = 0.03f;

    // Maximum number of points per model. Further points are ignored.
    int max_points = 20000;

    // Number of integrated frames before a model is used as a prior.
    int min_observations = 3;

    // Do not integrate a frame into a model if the model's absent fraction in that frame is above
    // this, as the object may be moving.
    float max_update_absent_fraction = 0.2f;

    // A model point is only added once its cell was observed in this many frames. Filters spurious
    // points, e.g. from depth interpolated across object boundaries.
    int min_point_observations = 2;

    // Skip pixels next to a range discontinuity larger than this in meters, as their depth may be
    // interpolated between object and background. Set to <= 0 to disable.
    float range_edge_threshold = 0.1f;

    // Skip pixels on the boundary of the semantic region, where labels and depth may disagree.
    bool skip_label_boundary = true;
  } const config;

  // Construction.
  explicit TrackModelStore(const Config& config);
  virtual ~TrackModelStore() = default;

  /**
   * @brief Append the models that are ready to be used as priors.
   */
  void getPriors(ObjectPriorPtrs& priors) const;

  /**
   * @brief Integrate the current semantic detections of static tracks into their models and remove
   * models of tracks that no longer exist. Must be called after tracking.
   * @param data Current frame data, including the prior observations of this frame.
   * @param tracks Current tracks.
   */
  void update(const FrameData& data, const Tracks& tracks);

  size_t size() const { return models_.size(); }

 private:
  struct Model {
    ObjectPrior prior;
    GlobalIndexMap<int> cell_observations;
    Eigen::Vector3f min = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f max = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
    int observations = 0;
    bool frozen = false;  // Stop integrating once the object was detected to move.
  };

  std::unordered_map<int, Model> models_;
  const spatial_hash::Grid<GlobalIndex> grid_;

  void integrate(const FrameData& data, const MeasurementCluster& cluster, Model& model) const;
};

void declare_config(TrackModelStore::Config& config);

}  // namespace khronos
