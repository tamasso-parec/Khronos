#include "khronos/active_window/prior/prior_module.h"

#include <algorithm>
#include <utility>

namespace khronos {

void declare_config(PriorModule::Config& config) {
  using namespace config;
  name("PriorModule");
  field(config.verbosity, "verbosity");
  field(config.use_tracks, "use_tracks");
  field(config.use_scene_graph, "use_scene_graph");
  field(config.track_overlap, "track_overlap");
  field(config.scene_graph, "scene_graph");
  field(config.track_models, "track_models");
  field(config.verifier, "verifier");
  field(config.motion, "motion");
  check(config.track_overlap, GE, 0.f, "track_overlap");
}

PriorModule::PriorModule(const Config& config)
    : config(config::checkValid(config)),
      scene_graph_(config.scene_graph),
      track_models_(config.track_models),
      verifier_(config.verifier),
      motion_detector_(config.motion) {}

void PriorModule::setSceneGraphSource(hydra::SharedDsgInfo::Ptr source) {
  scene_graph_.setSource(std::move(source));
}

void PriorModule::processInput(const VolumetricMap& map, FrameData& data) {
  if (config.use_scene_graph) {
    scene_graph_.refresh(data.input.timestamp_ns);
  }
  const ObjectPriorPtrs priors = collectPriors();
  verifier_.processInput(priors, data);
  motion_detector_.processInput(priors, data, map.getTrackingLayer());
}

void PriorModule::update(const FrameData& data, const Tracks& tracks) {
  if (config.use_tracks) {
    track_models_.update(data, tracks);
  }
}

ObjectPriorPtrs PriorModule::collectPriors() const {
  ObjectPriorPtrs priors;
  if (config.use_tracks) {
    track_models_.getPriors(priors);
  }
  if (!config.use_scene_graph) {
    return priors;
  }

  // Scene graph objects that are being re-observed are represented by their track model.
  const size_t num_tracks = priors.size();
  for (const ObjectPrior& prior : scene_graph_.getPriors()) {
    const bool tracked = std::any_of(
        priors.begin(), priors.begin() + num_tracks, [&](const ObjectPrior* track_prior) {
          return track_prior->semantic_label == prior.semantic_label &&
                 boxOverlapRatio(track_prior->bounding_box, prior.bounding_box) >
                     config.track_overlap;
        });
    if (!tracked) {
      priors.push_back(&prior);
    }
  }
  return priors;
}

}  // namespace khronos
