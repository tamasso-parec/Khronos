#include "khronos/active_window/prior/object_prior_store.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace khronos {

void declare_config(ObjectPriorStore::Config& config) {
  using namespace config;
  name("ObjectPriorStore");
  field(config.verbosity, "verbosity");
  field(config.refresh_period, "refresh_period", "s");
  field(config.vertex_subsampling, "vertex_subsampling");
  field(config.duplicate_overlap, "duplicate_overlap");
  check(config.refresh_period, GE, 0.f, "refresh_period");
  check(config.vertex_subsampling, GT, 0, "vertex_subsampling");
  check(config.duplicate_overlap, GE, 0.f, "duplicate_overlap");
}

float boxOverlapRatio(const BoundingBox& a, const BoundingBox& b) {
  const Eigen::Vector3f a_min = a.world_P_center - 0.5f * a.dimensions;
  const Eigen::Vector3f a_max = a.world_P_center + 0.5f * a.dimensions;
  const Eigen::Vector3f b_min = b.world_P_center - 0.5f * b.dimensions;
  const Eigen::Vector3f b_max = b.world_P_center + 0.5f * b.dimensions;
  const Eigen::Vector3f overlap =
      (a_max.cwiseMin(b_max) - a_min.cwiseMax(b_min)).cwiseMax(Eigen::Vector3f::Zero());
  const float smaller = std::min(a.dimensions.prod(), b.dimensions.prod());
  return smaller > 0.f ? overlap.prod() / smaller : 0.f;
}

ObjectPriorStore::ObjectPriorStore(const Config& config) : config(config::checkValid(config)) {}

void ObjectPriorStore::setSource(hydra::SharedDsgInfo::Ptr source) {
  source_ = std::move(source);
  cache_.clear();
  priors_.clear();
  last_sequence_number_ = 0;
  has_refreshed_ = false;
}

bool ObjectPriorStore::refresh(TimeStamp stamp) {
  if (!source_) {
    return false;
  }
  if (has_refreshed_ && stamp < last_refresh_ + fromSeconds(config.refresh_period)) {
    return false;
  }
  last_refresh_ = stamp;
  has_refreshed_ = true;

  size_t num_new = 0;
  {  // Critical section: hold the scene graph lock only while copying.
    std::lock_guard<std::mutex> lock(source_->mutex);
    if (source_->sequence_number == last_sequence_number_ || !source_->graph) {
      return false;
    }
    last_sequence_number_ = source_->sequence_number;

    const auto* layer = source_->graph->findLayer(DsgLayers::OBJECTS);
    if (!layer) {
      return false;
    }

    // Only copy objects not yet in the cache and drop the ones that were removed.
    std::unordered_set<NodeId> current;
    for (const auto& [id, node] : layer->nodes()) {
      current.insert(id);
      if (cache_.count(id)) {
        continue;
      }
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      ObjectPrior prior;
      if (attrs && createPrior(id, *attrs, prior)) {
        cache_.emplace(id, std::move(prior));
        ++num_new;
      }
    }
    for (auto it = cache_.begin(); it != cache_.end();) {
      it = current.count(it->first) ? std::next(it) : cache_.erase(it);
    }
  }  // End critical section.

  deduplicate();
  CLOG(4) << "[Object Prior Store] Updated snapshot: " << cache_.size() << " objects (" << num_new
          << " new), " << priors_.size() << " after removing duplicates.";
  return true;
}

void ObjectPriorStore::deduplicate() {
  // Keep the most recently observed object of each group of duplicates.
  std::vector<const ObjectPrior*> sorted;
  sorted.reserve(cache_.size());
  for (const auto& [id, prior] : cache_) {
    sorted.push_back(&prior);
  }
  std::sort(sorted.begin(), sorted.end(), [](const ObjectPrior* a, const ObjectPrior* b) {
    return a->last_observed > b->last_observed;
  });

  priors_.clear();
  for (const ObjectPrior* candidate : sorted) {
    const bool duplicate =
        std::any_of(priors_.begin(), priors_.end(), [&](const ObjectPrior& kept) {
          return kept.semantic_label == candidate->semantic_label &&
                 boxOverlapRatio(kept.bounding_box, candidate->bounding_box) >
                     config.duplicate_overlap;
        });
    if (!duplicate) {
      priors_.push_back(*candidate);
    }
  }
}

bool ObjectPriorStore::createPrior(NodeId id,
                                   const KhronosObjectAttributes& attrs,
                                   ObjectPrior& prior) const {
  // Only static objects with geometry are used as priors.
  if (!attrs.trajectory_positions.empty() || attrs.mesh.numVertices() == 0) {
    return false;
  }

  prior.source = PriorSource::kSceneGraph;
  prior.id = id;
  prior.bounding_box = attrs.bounding_box;
  prior.semantic_label = attrs.semantic_label;
  if (!attrs.first_observed_ns.empty()) {
    prior.first_observed = attrs.first_observed_ns.front();
  }
  if (!attrs.last_observed_ns.empty()) {
    prior.last_observed = attrs.last_observed_ns.back();
  }

  // Object meshes are stored relative to the bounding box center.
  const Point& offset = attrs.bounding_box.world_P_center;
  prior.vertices.reserve(attrs.mesh.numVertices() / config.vertex_subsampling + 1);
  for (size_t i = 0; i < attrs.mesh.numVertices(); i += config.vertex_subsampling) {
    prior.vertices.emplace_back(attrs.mesh.pos(i) + offset);
  }
  return true;
}

}  // namespace khronos
