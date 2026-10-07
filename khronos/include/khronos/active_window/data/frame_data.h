/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * AUTHORS:      Lukas Schmid <lschmid@mit.edu>, Marcus Abate <mabate@mit.edu>,
 *               Yun Chang <yunchang@mit.edu>, Luca Carlone <lcarlone@mit.edu>
 * AFFILIATION:  MIT SPARK Lab, Massachusetts Institute of Technology
 * YEAR:         2024
 * SOURCE:       https://github.com/MIT-SPARK/Khronos
 * LICENSE:      BSD 3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <hydra/common/global_info.h>
#include <hydra/input/input_data.h>
#include <opencv2/core.hpp>

#include "khronos/active_window/data/measurement_clusters.h"
#include "khronos/common/common_types.h"

namespace khronos {

using hydra::InputData;

/**
 * @brief Where an object prior comes from.
 */
enum class PriorSource : uint8_t {
  kSceneGraph = 0,  // Static object in the scene graph that has left the active window.
  kTrack = 1,       // Static track inside the active window.
};

/**
 * @brief State of a pixel covered by a rendered object prior.
 */
enum class PriorPixelState : uint8_t {
  kNone = 0,    // No prior rendered at this pixel.
  kExcluded,    // Prior rendered but excluded from verification (silhouette or depth edge).
  kPresent,     // Measurement matches the expected surface.
  kAbsent,      // Measurement lies behind the expected surface, the ray passes through the object.
  kOccluded,    // Measurement lies in front of the expected surface.
  kUnobserved,  // No valid measurement.
};

/**
 * @brief Evidence for a single object prior in a single frame.
 */
struct PriorObservation {
  // Source of the prior and its ID: the scene graph node ID or the track ID.
  PriorSource source = PriorSource::kSceneGraph;
  uint64_t id = 0;
  spark_dsg::SemanticLabel semantic_label = 0;

  // Number of pixels per state.
  int rendered = 0;
  int excluded = 0;
  int present = 0;
  int absent = 0;
  int occluded = 0;
  int unobserved = 0;

  // Histogram of measured semantic labels over all present pixels.
  std::map<int, int> labels;

  // Size of the largest 4-connected patch of absent pixels.
  int largest_absent_component = 0;

  // Motion state set by the prior motion detector.
  float baseline = -1.f;  // Running absent fraction while static, -1 if not yet established.
  bool moving = false;    // Whether motion onset was detected for this prior.
  int seed_pixels = 0;    // Number of current pixels seeded as dynamic for this prior.

  // Fraction of informative pixels that indicate absence, -1 if there are none.
  float absentFraction() const {
    const int informative = present + absent;
    return informative > 0 ? absent / static_cast<float>(informative) : -1.f;
  }
};

/**
 * @brief Data structure that collects also other data to be passed around with
 * each input.
 */
struct FrameData {
  using Ptr = std::shared_ptr<FrameData>;
  using ConstPtr = std::shared_ptr<const FrameData>;

  // Associated raw input data.
  const InputData input;

  // Dynamic clusters in this frame.
  std::vector<MeasurementCluster> dynamic_clusters;

  // ID of detected dynamic clusters in this frame as 8UC1. 0 indicates static.
  Image dynamic_image;
  using DynamicImageType = int;

  // Semantic clusters in this frame.
  std::vector<MeasurementCluster> semantic_clusters;

  // ID of detected semantic clusters in this frame as 8UC1. 0 indicates background.
  // NOTE(lschmid): Per frame we store a maximum of 255 objects. If a larger number is
  // needed, the allocation and projective integration (interpolators) need to be updated.
  Image object_image;
  using ObjectImageType = int;

  // Scene graph object priors rendered in this frame and the evidence for each. Images are empty
  // if prior verification is disabled.
  std::vector<PriorObservation> prior_observations;

  // Index + 1 into prior_observations of the rendered prior per pixel as 32SC1. 0 indicates none.
  Image prior_id_image;
  using PriorIdImageType = int;

  // Expected range of the rendered prior per pixel in meters as 32FC1. Infinity indicates none.
  Image prior_range_image;
  using PriorRangeImageType = float;

  // PriorPixelState per pixel as 8UC1.
  Image prior_state_image;
  using PriorStateImageType = uint8_t;

  // Pixels of moving priors to be used as motion seeds as 8UC1, non-zero indicates a seed.
  Image prior_seed_mask;

  // Pixels that match a confidently static prior as 8UC1. Free-space motion seeds are suppressed
  // where non-zero.
  Image prior_suppress_mask;

  explicit FrameData(const hydra::InputData& input) : input(input) {}
};

}  // namespace khronos
