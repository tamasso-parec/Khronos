# Recording measurements for offline motion experiments

The optional measurement sink publishes `/khronos_node/measurements`
(`khronos_msgs/msg/MeasurementFrame`) once per processed active-window frame.
It does not change detection, tracking, integration, or the window duration.
Export runs after integration and before output throttling and inactive-track
extraction. The publisher is reliable, volatile, with a queue depth of 10.

## Build and record

Build in the ROS environment used to run Khronos (the supplied container uses
ROS Jazzy), with the workspace dependencies already built:

```bash
cd /root/ros2_ws
source install/setup.bash
colcon build --packages-select khronos_msgs khronos_ros
source install/setup.bash
```

First start the recorder in another sourced terminal. The output directory must
not already exist; existing recordings are never overwritten:

If using Docker, run `./docker_exec.sh` from the Khronos folder on the host to
open additional shells in the **same running container**. It sources ROS and
the built workspace automatically. With multiple containers, select the one
running the launch, or use `./docker_exec.sh <container-name>`. Do not
run `docker_run.sh` again for each terminal: that creates separate containers.
Use the same container for launching, recording, and playback service calls.
The supplied Docker command uses private IPC, so separate containers sharing
host networking must not be assumed to have working ROS discovery/transport.

```bash
ros2 run khronos_ros record_measurements.py /data/motion_run_01
```

Then launch Khronos with export enabled and playback paused:

```bash
ros2 launch khronos_ros uhumans2_khronos.launch.yaml \
  publish_measurements:=true paused:=true
```

Resume playback once the recorder subscription has connected, using a second
terminal in the same container with the workspace sourced:

```bash
source /root/ros2_ws/install/setup.bash
ros2 service call /rosbag2_player/resume rosbag2_interfaces/srv/Resume '{}'
```

The player is launched as a subprocess, so pressing Space in the launch terminal
may not reach it. Use its ROS services instead. To pause again:

```bash
ros2 service call /rosbag2_player/pause rosbag2_interfaces/srv/Pause '{}'
```

If the call waits for the service, check `ros2 service list -t` in the container
running the launch and ensure the calling terminal uses the same ROS domain.
For a fresh discovery query, use `ros2 service list --no-daemon -t`.
The player must still be running. The same
`publish_measurements` argument is available on `khronos.launch.yaml`.
For other entry points, the pipeline configuration field is
`publish_measurements: true` (top level). Export defaults to false.

After playback ends, allow Khronos to finish processing its queued inputs, then
Ctrl-C **the recorder** to close the dataset. Do not stop recording as soon as
bag playback stops: processing may lag behind playback. The recorder does not
interpret a pause or an idle timeout as completion. Its `closed` flag only
indicates clean recorder shutdown, not completion of the mapping experiment.
An abrupt interruption leaves readable completed frame files and `closed: false`.

You can alternatively record the measurement topic with rosbag2 and replay it
to this recorder. Keep the recorder alive until replay has drained. Use a new
dataset directory for each run/replay; publisher restarts reset the sequence.

## What is exported

Each message is self-contained and stamped with the processed input timestamp:

| Field | Meaning |
| --- | --- |
| `header` | Input timestamp and odometry frame name used by ROS input poses |
| `sequence` | Zero-based processed-frame counter, incremented even without subscribers |
| `world_t_sensor` | Sensor-to-world pose; world here is the input odometry frame, not a later loop-closure-corrected map |
| `depth` | Processed planar depth, meters, `32FC1` |
| `vertices` | Exact world-coordinate XYZ for every pixel, `32FC3` |
| `semantic_labels` | Processed/remapped integer semantic labels, `32SC1` |
| `dynamic_labels` | Frame-local dynamic cluster IDs, `32SC1` |
| `object_labels` | Frame-local semantic object cluster IDs, `32SC1` |
| `tracks` | Persistent track IDs, dynamic/activity flags, confidence, first/last seen, and current-frame cluster associations |

The image headers share the packet header; `vertices` contains world-frame
coordinates. Exact XYZ is exported instead of camera calibration so offline
code need not duplicate sensor projection or independently synchronize depth,
labels, and poses. RGB is not duplicated; it can be read from the original bag.

Track association `-1` means no corresponding observation in this frame. Both
semantic and dynamic associations may be present. Tracks that were not observed
in this frame may still be active. Export includes inactive tracks still in the
window, before the output stage removes them. Persistent IDs are meaningful
within one publisher/mapping run; detection IDs are local to each frame.

This initial interface targets the current closed-set RGB-D pipeline. It does
not export open-set embeddings, class probabilities, pre-filter motion seeds,
full TSDF/free-space history, or final extraction events. The current motion
detector also caps detection IDs at 255, so frames with more clusters retain
that upstream ambiguity. The sink exports existing results without correcting
or reinterpreting them.

## Check a recording

After building and sourcing the workspace, run:

```bash
ros2 run khronos_ros check_measurements.py /data/motion_run_01
```

No manual Python-path changes are needed. The script validates the dataset,
reads every frame's geometry and motion mask, and looks up the points for each
dynamic track association. It prints frame counts, duration, unique track count,
dynamic pixel/valid-point counts, association counts, and sequence gaps.
It exits with status 1 for empty or unclosed recordings, sequence gaps, missing
associated clusters, or invalid data. Zero dynamic detections is valid. A pass
checks data access and consistency, not detector accuracy or unrecorded trailing
frames. The checker does not modify the recording.

## Load in Python without ROS

The installed `khronos_measurements` module only needs NumPy for offline use.
Sourcing the built workspace makes it available to your Python scripts.
Files contain numerical arrays and JSON metadata, with no pickle dependency.

```python
from khronos_measurements import Dataset

dataset = Dataset("/data/motion_run_01")
print(dataset.validate())

for frame in dataset:
    xyz = frame.vertices                  # H x W x 3, world coordinates
    motion_guess = frame.dynamic_labels != 0
    for track in frame.tracks:
        cluster_id = track["dynamic_cluster_id"]
        if cluster_id != -1:
            points = xyz[frame.dynamic_labels == cluster_id]
            # Accumulate measurements under track["id"] for batch refinement.
```

`Frame.world_t_sensor` is a seven-element NumPy array: translation XYZ followed
by quaternion XYZW. Timestamps are integer nanoseconds. Iteration is lazy; frames
are loaded one at a time. An optimizer can explicitly gather a temporal subset
or the whole sequence. No optimization or feedback into Khronos is implemented.

Storage is one uncompressed NPZ per frame plus `metadata.json`. Uncompressed
writing avoids compression work in subscription callbacks. Disk use is about
28 bytes per pixel per frame (depth, XYZ, and three label images), plus metadata;
at 640 x 480 this is about 8.6 MB/frame or 172 MB/s at 20 Hz. Use short initial
runs and slower bag playback if recording cannot keep up. Exact geometry costs
bandwidth but makes the first experiments reproducible. Reliability does not
guarantee completeness if queues overflow or recording starts/stops too late.

`Dataset.validate()` checks every frame, array shape/type, pose, chronological
order, and the final file count. It reports missing sequence intervals,
including a late subscription at the start. It cannot detect missing trailing
frames without an expected final producer count. Keep the original input bag
and Khronos's experiment configuration logs beside the dataset for provenance.

## Interpretation for batch refinement

- Zero in a detection image means **no accepted detection**, not confidently
  static/background. Missed or filtered detections are also zero.
- Semantic labels in the default uHumans2 launch are ground truth.
- Track confidence is Khronos's observation-count heuristic, not a calibrated
  motion probability or covariance.
- Keep points outside the initial detection masks if the optimizer should be
  able to recover false negatives. Check finite XYZ and valid positive depth
  before constructing geometric residuals.
- Exported geometry precedes backend deformation/loop-closure correction.
- The sink observes the post-integration state. Refined labels do not revise
  already integrated geometry; this is a measurement interface for independent
  batch experiments.

## Checks

The NumPy dataset and message-decoding tests run without ROS:

```bash
PYTHONPATH=khronos/khronos_ros python3 -m pytest \
  khronos/khronos_ros/tests/test_measurement_dataset.py
```

They cover persistence of geometry/labels/track IDs, padded big-endian ROS
images, timestamp/header mismatches, missing sequences, restart rejection,
incomplete recordings, and corrupt frame counts.
