#include "khronos/active_window/prior/track_model_store.h"

#include <cmath>
#include <unordered_set>

namespace khronos {

void declare_config(TrackModelStore::Config& config) {
  using namespace config;
  name("TrackModelStore");
  field(config.verbosity, "verbosity");
  field(config.resolution, "resolution", "m");
  field(config.max_points, "max_points");
  field(config.min_observations, "min_observations", "frames");
  field(config.max_update_absent_fraction, "max_update_absent_fraction");
  field(config.min_point_observations, "min_point_observations", "frames");
  field(config.range_edge_threshold, "range_edge_threshold", "m");
  field(config.skip_label_boundary, "skip_label_boundary");
  check(config.resolution, GT, 0.f, "resolution");
  check(config.max_points, GT, 0, "max_points");
  check(config.min_observations, GT, 0, "min_observations");
  check(config.min_point_observations, GT, 0, "min_point_observations");
}

TrackModelStore::TrackModelStore(const Config& config)
    : config(config::checkValid(config)), grid_(config.resolution) {}

void TrackModelStore::getPriors(ObjectPriorPtrs& priors) const {
  for (const auto& [id, model] : models_) {
    if (model.observations >= config.min_observations) {
      priors.push_back(&model.prior);
    }
  }
}

void TrackModelStore::update(const FrameData& data, const Tracks& tracks) {
  Timer timer("active_window/track_models", data.input.timestamp_ns);
  const TimeStamp stamp = data.input.timestamp_ns;

  // Index this frame's evidence and clusters for lookup.
  std::unordered_map<uint64_t, const PriorObservation*> evidence;
  for (const auto& obs : data.prior_observations) {
    if (obs.source == PriorSource::kTrack) {
      evidence.emplace(obs.id, &obs);
    }
  }
  std::unordered_map<int, const MeasurementCluster*> clusters;
  for (const auto& cluster : data.semantic_clusters) {
    clusters.emplace(cluster.id, &cluster);
  }

  std::unordered_set<int> existing;
  for (const Track& track : tracks) {
    existing.insert(track.id);
    if (track.is_dynamic || track.observations.empty()) {
      continue;
    }
    const Observation& last = track.observations.back();
    if (last.stamp != stamp || last.semantic_cluster_id < 0 || last.dynamic_cluster_id >= 0) {
      continue;
    }
    const auto cluster = clusters.find(last.semantic_cluster_id);
    if (cluster == clusters.end()) {
      continue;
    }

    auto& model = models_[track.id];
    model.prior.source = PriorSource::kTrack;
    model.prior.id = track.id;
    if (track.semantics && track.semantics->category_id >= 0) {
      model.prior.semantic_label = track.semantics->category_id;
    }
    if (model.observations == 0) {
      model.prior.first_observed = stamp;
    }

    // Never integrate an object that is or may be moving, to keep the model of where it was.
    const auto obs = evidence.find(track.id);
    if (obs != evidence.end()) {
      model.frozen |= obs->second->moving;
      if (obs->second->absentFraction() > config.max_update_absent_fraction) {
        continue;
      }
    }
    if (model.frozen) {
      continue;
    }

    integrate(data, *cluster->second, model);
    model.prior.last_observed = stamp;
    ++model.observations;
  }

  // Remove models of tracks that were removed from the tracker.
  for (auto it = models_.begin(); it != models_.end();) {
    it = existing.count(it->first) ? std::next(it) : models_.erase(it);
  }
}

void TrackModelStore::integrate(const FrameData& data,
                                const MeasurementCluster& cluster,
                                Model& model) const {
  const cv::Mat& ranges = data.input.range_image;
  const cv::Mat& labels = data.input.label_image;
  const bool has_dynamic = !data.dynamic_image.empty();
  const bool check_labels =
      config.skip_label_boundary && !labels.empty() && labels.type() == CV_32SC1;

  // Each cell is counted at most once per frame.
  GlobalIndexSet observed_this_frame;
  for (const Pixel& pixel : cluster.pixels) {
    if (model.prior.vertices.size() >= static_cast<size_t>(config.max_points)) {
      break;
    }
    if (has_dynamic && data.dynamic_image.at<FrameData::DynamicImageType>(pixel.v, pixel.u)) {
      continue;
    }
    const float range = ranges.at<InputData::RangeType>(pixel.v, pixel.u);
    if (!(range > 0.f) || !data.input.inRange(range)) {
      continue;
    }

    // Skip pixels at depth discontinuities and label boundaries.
    bool is_boundary = false;
    for (const Pixel& neighbor : pixel.neighbors8()) {
      if (!neighbor.isInImage(ranges)) {
        continue;
      }
      const float neighbor_range = ranges.at<InputData::RangeType>(neighbor.v, neighbor.u);
      if (config.range_edge_threshold > 0.f &&
          (!(neighbor_range > 0.f) ||
           std::abs(neighbor_range - range) > config.range_edge_threshold)) {
        is_boundary = true;
        break;
      }
      if (check_labels && labels.at<InputData::LabelType>(neighbor.v, neighbor.u) !=
                              labels.at<InputData::LabelType>(pixel.v, pixel.u)) {
        is_boundary = true;
        break;
      }
    }
    if (is_boundary) {
      continue;
    }

    const auto& vertex = data.input.vertex_map.at<InputData::VertexType>(pixel.v, pixel.u);
    const Point point(vertex[0], vertex[1], vertex[2]);
    const GlobalIndex cell = grid_.toIndex(point);
    if (!observed_this_frame.insert(cell).second) {
      continue;
    }
    // Add the point once the cell was observed often enough.
    if (++model.cell_observations[cell] != config.min_point_observations) {
      continue;
    }
    model.prior.vertices.push_back(point);
    model.min = model.min.cwiseMin(point);
    model.max = model.max.cwiseMax(point);
  }

  if (!model.prior.vertices.empty()) {
    model.prior.bounding_box = BoundingBox(model.max - model.min, 0.5f * (model.max + model.min));
  }
}

}  // namespace khronos
