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

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <hydra/common/global_info.h>
#include <hydra/input/input_data.h>
#include <opencv2/core.hpp>

#include "khronos/active_window/data/measurement_clusters.h"
#include "khronos/common/common_types.h"

namespace khronos {

using hydra::InputData;

/**
 * @brief Data structure that collects also other data to be passed around with
 * each input.
 */
struct FrameData {
  using Ptr = std::shared_ptr<FrameData>;
  using ConstPtr = std::shared_ptr<const FrameData>;

  // Associated raw input data. Only correctPose() may change it, before anything else uses the pose.
  InputData input;

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

  explicit FrameData(const hydra::InputData& input) : input(input) {}

  /**
   * @brief Replaces the body pose of the input, e.g. with one the motion detector refined, and moves a
   * world-frame vertex map along. Only valid right after motion detection: tracking, integration and
   * object extraction then use the corrected pose. The vertex map is written to a new buffer, so
   * shallow copies of the input held elsewhere keep the original.
   * @param world_T_body The corrected pose of the body.
   */
  void correctPose(const Eigen::Isometry3d& world_T_body) {
    if (input.points_in_world_frame && !input.vertex_map.empty()) {
      const Eigen::Isometry3f correction = (world_T_body * input.world_T_body.inverse()).cast<float>();
      cv::Mat vertex_map(input.vertex_map.size(), input.vertex_map.type());
      for (int v = 0; v < vertex_map.rows; ++v) {
        const auto* from = input.vertex_map.ptr<InputData::VertexType>(v);
        auto* to = vertex_map.ptr<InputData::VertexType>(v);
        for (int u = 0; u < vertex_map.cols; ++u) {
          const Eigen::Vector3f point = correction * Eigen::Vector3f(from[u][0], from[u][1], from[u][2]);
          to[u] = InputData::VertexType(point.x(), point.y(), point.z());
        }
      }
      input.vertex_map = vertex_map;
    }
    input.world_T_body = world_T_body;
    pose_corrected = true;
  }

  // Whether correctPose() replaced the pose the input arrived with.
  bool pose_corrected = false;
};

}  // namespace khronos
