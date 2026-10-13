# Optical-flow dynamics evidence

Khronos can compute per-pixel motion evidence for every RGB-D frame with the
[dynamic_psg](https://github.com/utn-mi/dynamic_psg) package. The estimator never consults
the Khronos map: it tracks the current frame's pixels backward through recent frames with
optical flow and compares static and constant-velocity models of each point. Its method,
parameters, outputs and measured results are documented in that repository
([method](https://github.com/utn-mi/dynamic_psg/blob/main/docs/method.md),
[evaluation](https://github.com/utn-mi/dynamic_psg/blob/main/docs/evaluation.md)).

This page covers its integration in Khronos. The estimate is stored in
`FrameData::dynamics`, reaches existing sinks and is published for visualization; it has no
effect on Khronos's binary `dynamic_image` mask.

## Setup

`dynamic_psg` is listed in `install/*.rosinstall`, so `vcs import` places it next to Khronos.
The repository is private: build the Docker image with `docker build --ssh default -t
khronos:jazzy .`, which clones it with your SSH agent's key. `docker_run.sh` mounts a sibling
`../dynamic_psg` checkout into the container when present, so changes to it rebuild like
Khronos itself.

## Integration

[`khronos::OpticalFlowDynamics`](khronos/include/khronos/active_window/motion_detection/optical_flow_dynamics.h)
implements Khronos's `DynamicsEstimator` interface by converting each Hydra camera input
into a `dynamic_psg::RgbdFrame`: the sensor pose (body pose times extrinsics), intrinsics and
range limits, RGB, depth and Hydra's vertex map. It registers with `config_utilities` as
`OpticalFlowDynamics` and takes `dynamic_psg::OpticalFlowDynamics::Config`, so every
estimator parameter is set in the mapper YAML under `active_window.dynamics_estimator`.
Non-camera inputs reset the estimator. Estimator runtime is logged by Hydra's timing
recorder as `active_window/dynamics`.

## Run and configure

Inside Docker, build and source the workspace, then run:

```bash
ros2 launch khronos_ros optical_flow_dynamics_test.launch.yaml \
  bag_path:=/data/tesse_cd_apartment
```

The test mapper
([optical_flow_dynamics_test.yaml](khronos_ros/config/mapper/optical_flow_dynamics_test.yaml))
uses temporal scoring with CRF spatial support and DIS FAST flow, detecting motion on every
third frame. It supplies GT poses and disables binary object/motion detection, tracking and
change detection; the background Khronos pipeline still runs. Input requires RGB,
RGB-aligned depth, camera info, GT TF and GT labels for the closed-set receiver. Use
`rviz:=false` for headless operation or `dataset:=tesse_cd_office` for the office bag.

Playback defaults to real time (`play_rate:=1.0`). Hydra's bounded queues block instead
of dropping frames, so playback faster than the pipeline builds a backlog and the output
falls further behind the bag. The `active_window/all` row of `timing_stats.csv` in the
output directory shows the mean cost per frame; keep it below `0.05 s / play_rate`. The test
mapper keeps 300 s of TF: with the default 30 s, a backlog older than the TF buffer makes
every pose lookup fail and the input never recovers.

To change detector settings, copy the test mapper and edit its `dynamics_estimator`
section:

```bash
cp "$(ros2 pkg prefix --share khronos_ros)/config/mapper/optical_flow_dynamics_test.yaml" /tmp/dynamics.yaml
ros2 launch khronos_ros optical_flow_dynamics_test.launch.yaml \
  bag_path:=/data/tesse_cd_apartment mapper_config:=/tmp/dynamics.yaml
```

```yaml
active_window:
  dynamics_estimator:
    type: OpticalFlowDynamics
    spatial_enabled: true
    spatial_method: crf
    flow_method: dis_fast
    keyframe_interval: 3
    debug_output: true  # Fit and velocity diagnostics; keep off otherwise.
```

An existing mapper can merge
[optical_flow_dynamics.yaml](khronos_ros/config/mapper/optical_flow_dynamics.yaml); this small
fragment is not a complete `mapper_config`.

## Topics

Relative to `/khronos_node/visualization/`, published only when subscribed:

- `dynamics_cloud`: organized cloud of the reference frame with all evidence fields
  (layout in the `dynamic_psg` method documentation).
- `dynamics_motion_evidence`, `dynamics_regularized_evidence`, `dynamics_reliability`:
  32FC1 images; `dynamics_validity`: mono8 decisions.
- `dynamics_evidence`: RGB8 evidence image; `dynamics_failures`: RGB8 failure image;
  `dynamics_failure_flags`: 16UC1 failure bits; `dynamics_track_alive`: mono8.
- `dynamics_probability`: legacy scoring mode only.

ROS headers carry the reference timestamp, which is the current input's in temporal mode.
Record outputs for evaluation with, e.g.:

```bash
ros2 bag record -o /data/output/temporal_run \
  /khronos_node/visualization/dynamics_cloud \
  /khronos_node/visualization/dynamics_failure_flags
```

and score them with `ros2 run dynamic_psg_ros evaluate_dynamics_bag.py` (see the
`dynamic_psg` evaluation documentation).

## Tests

`test_optical_flow_dynamics_adapter` checks the conversion of Hydra camera inputs, including
extrinsics and world-frame vertex maps, and that the factory-created adapter produces the
same estimates as `dynamic_psg` on the converted frames. The estimator's own tests live in
`dynamic_psg`.
