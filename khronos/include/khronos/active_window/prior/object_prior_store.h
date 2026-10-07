#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <config_utilities/config_utilities.h>
#include <hydra/common/global_info.h>
#include <hydra/common/shared_dsg_info.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Geometry and semantics of an object that is expected to be observed, used to verify the
 * object against the current measurement.
 */
struct ObjectPrior {
  PriorSource source = PriorSource::kSceneGraph;

  // Scene graph node ID or track ID, depending on the source.
  uint64_t id = 0;

  // Surface points of the object in world (odometry) frame.
  Points vertices;

  // Bounding box of the object in world frame.
  BoundingBox bounding_box;

  spark_dsg::SemanticLabel semantic_label = 0;

  // First and last time the object was observed.
  TimeStamp first_observed = 0;
  TimeStamp last_observed = 0;

  // Unique key over both sources.
  uint64_t key() const { return key(source, id); }
  static uint64_t key(PriorSource source, uint64_t id) {
    return source == PriorSource::kTrack ? (id | (1ull << 63)) : id;
  }
};
using ObjectPriors = std::vector<ObjectPrior>;
using ObjectPriorPtrs = std::vector<const ObjectPrior*>;

/**
 * @brief Fraction of the smaller box that overlaps the other box, using the axis-aligned extents.
 */
float boxOverlapRatio(const BoundingBox& a, const BoundingBox& b);

/**
 * @brief Keeps a thread-safe snapshot of the static objects in a shared scene graph so
 * the active window can use them without holding the scene graph lock.
 */
class ObjectPriorStore {
 public:
  // Config.
  struct Config {
    int verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;

    // Minimum time between two snapshots of the scene graph in seconds.
    float refresh_period = 1.f;

    // Keep every n-th mesh vertex of each object.
    int vertex_subsampling = 1;

    // Objects of the same label overlapping by more than this fraction are considered duplicates,
    // of which only the most recently observed one is kept. Set to > 1 to disable.
    float duplicate_overlap = 0.5f;
  } const config;

  // Construction.
  explicit ObjectPriorStore(const Config& config);
  virtual ~ObjectPriorStore() = default;

  /**
   * @brief Set the scene graph to read object priors from. The scene graph must be in the same
   * frame as the active window.
   */
  void setSource(hydra::SharedDsgInfo::Ptr source);
  bool hasSource() const { return source_ != nullptr; }

  /**
   * @brief Update the snapshot if the refresh period elapsed and the source changed. Only objects
   * not yet in the snapshot are copied.
   * @param stamp Current time stamp.
   * @returns True if the snapshot was updated.
   */
  bool refresh(TimeStamp stamp);

  // Access to the deduplicated priors.
  const ObjectPriors& getPriors() const { return priors_; }

 private:
  hydra::SharedDsgInfo::Ptr source_;
  std::unordered_map<NodeId, ObjectPrior> cache_;  // All valid objects in the source.
  ObjectPriors priors_;                            // Deduplicated objects.
  uint64_t last_sequence_number_ = 0;
  TimeStamp last_refresh_ = 0;
  bool has_refreshed_ = false;

  bool createPrior(NodeId id, const KhronosObjectAttributes& attrs, ObjectPrior& prior) const;
  void deduplicate();
};

void declare_config(ObjectPriorStore::Config& config);

}  // namespace khronos
