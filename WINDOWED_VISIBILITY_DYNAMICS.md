# Windowed-visibility dynamics evidence

`dynamic_psg`'s correspondence-free estimator `WindowedVisibilityDynamics` projects every
point of the current frame into the last W - 1 frames and counts the frames that saw
through it (free-space conflicts) against those that saw it; an a contrario test on these
counts decides whether the point has moved. It never consults the Khronos map. The method,
parameters and measured results are documented in `dynamic_psg`
([windowed_visibility.md](https://github.com/utn-mi/dynamic_psg/blob/main/docs/windowed_visibility.md)).
The optical-flow estimator ([OPTICAL_FLOW_DYNAMICS.md](OPTICAL_FLOW_DYNAMICS.md)) is kept
as the baseline.

## Plugin loading

The estimator is not linked into Khronos. The `dynamic_psg_khronos` package builds a plugin
library that registers it with `config_utilities` as a `DynamicsEstimator`; `khronos_node`
loads the libraries listed under the top-level `external_library_paths` before building the
pipeline, so a mapper config selects it by type:

```yaml
external_library_paths: [dynamic_psg_khronos]  # Resolved on LD_LIBRARY_PATH.

active_window:
  dynamics_estimator:
    type: WindowedVisibilityDynamics
    window: {size: 10}
```

As with the optical-flow estimator, the estimate is stored in `FrameData::dynamics`, its
runtime is logged as `active_window/dynamics` (with the estimator's own
`dynamics/windowed_visibility/*` timers), and it has no effect on Khronos's binary
`dynamic_image` mask.

## Run

Inside Docker (with `docker_run.sh` mounting `../dynamic_psg`), build and source the
workspace, then:

```bash
ros2 launch khronos_ros windowed_visibility_dynamics.launch.yaml \
  bag_path:=/data/tesse_cd_apartment
```

The launch reuses `optical_flow_dynamics_test.launch.yaml` (GT poses, uHumans2 wiring,
`pose_noise_level`, `rviz:=false`, `dataset:=tesse_cd_office`) with the mapper
[windowed_visibility_dynamics.yaml](khronos_ros/config/mapper/windowed_visibility_dynamics.yaml)
(W = 10, a keyframe every 0.3 s) and the RViz layout
[windowed_visibility_dynamics.rviz](khronos_ros/config/rviz/windowed_visibility_dynamics.rviz).
Pass `mapper_config:=...` to try other estimator settings.

## Visualization

The RViz layout shows, from `/khronos_node/visualization/`:

- **Decision** (`dynamics_cloud`, RGB channel): the current frame's points, red for
  dynamic (NFA <= epsilon), blue for static, gray where the window could not decide (too few
  conclusive tests, depth edges, invalid depth). Red and blue shade with
  `motion_evidence` = 1 / (1 + NFA / epsilon).
- **Free-space conflicts k** and **conclusive tests n** (same cloud, intensity of
  `free_space_count` and `observations`, 0 to 9; disabled by default): the two counts
  the decision is made from.
- **Decision image** (`dynamics_evidence`) next to the camera image.
- **Missing evidence** (`dynamics_failures`, disabled): why some past frame gave no evidence
  for a pixel: magenta outside its view, cyan occluded, at a depth edge or out of range.

The topic list and recording commands of
[OPTICAL_FLOW_DYNAMICS.md](OPTICAL_FLOW_DYNAMICS.md#topics) apply unchanged; flow-specific
outputs (`dynamics_regularized_evidence`, `dynamics_reliability`, `dynamics_track_alive`)
are empty.
