# Optical-flow dynamics evidence

`OpticalFlowDynamics` is an optional `config_utilities`-registered Khronos module.
It tracks registered RGB-D input with Farneback flow and never consults the Khronos
map. In temporal mode every input becomes the reference: its pixels are tracked
backward through a sliding window of recent frames, so the output geometry is the
current frame's world points. Temporal mode fits a statistical static point and a
velocity without modifying that geometry. Legacy mode keeps non-overlapping windows
anchored on their first frame. Results live in `FrameData::dynamics`, reach existing
sinks, and have no effect on Khronos's binary `dynamic_image` mask.

## Run and configure

Inside Docker, build and source the workspace, then run:

```bash
ros2 launch khronos_ros optical_flow_dynamics_test.launch.yaml \
  bag_path:=/data/tesse_cd_apartment
```

This uses temporal scoring with CRF spatial support and DIS FAST flow, detecting
motion on every third frame (`keyframe_interval: 3`, `flow_method: dis_fast`). Use `rviz:=false` for headless operation or
`dataset:=tesse_cd_office` for the office bag. Playback defaults to `play_rate:=0.25`;
this changes playback speed, not sensor timestamps. Faster playback than the pipeline
can process builds a backlog: Hydra's queues block instead of dropping frames, and
the output falls further behind the bag. The `active_window/all` row of
`timing_stats.csv` in the output directory shows the mean per-frame cost; keep it
below `0.05 s / play_rate`.

Detector settings belong to `active_window.dynamics_estimator` in the mapper YAML.
To change them, copy the complete test mapper and add the desired overrides:

```bash
cp "$(ros2 pkg prefix --share khronos_ros)/config/mapper/optical_flow_dynamics_test.yaml" /tmp/dynamics.yaml
# Edit /tmp/dynamics.yaml, then:
ros2 launch khronos_ros optical_flow_dynamics_test.launch.yaml \
  bag_path:=/data/tesse_cd_apartment mapper_config:=/tmp/dynamics.yaml
```

For example, enable CRF support and diagnostics with this section in that file:

```yaml
active_window:
  dynamics_estimator:
    type: OpticalFlowDynamics
    spatial_enabled: true
    spatial_method: crf
    debug_output: true
```

Keep the remaining mapper sections from the copied file. For one-pass support, use
`spatial_method: support`; for the previous scorer, use `scoring_mode: legacy`.
The former launch arguments `scoring_mode`, `spatial_enabled`, `spatial_method` and
`debug_output` are now YAML settings only. Launch arguments cover inputs, playback,
output locations and viewers; they no longer override detector settings in YAML.

The supplied YAML files rely on C++ defaults, retaining only `type` and the preset's
`spatial_method: crf` override. Every detector option remains available through
`config_utilities`, with its existing name, default and validation. An existing
mapper can merge
[khronos_ros/config/mapper/optical_flow_dynamics.yaml](khronos_ros/config/mapper/optical_flow_dynamics.yaml).
This small fragment is not a complete replacement for `mapper_config`.

The test setup supplies GT poses and disables binary object/motion detection,
tracking and change detection. The background Khronos pipeline still runs. Input
requires RGB, RGB-aligned depth, camera info, GT TF, and GT labels for the existing
closed-set receiver. Publishers serialize only subscribed topics. Keep
`debug_output: false` unless fit/velocity diagnostics are needed.

Record outputs into the mounted directory, e.g.:

```bash
ros2 bag record -o /data/output/temporal_run \
  /khronos_node/visualization/dynamics_cloud \
  /khronos_node/visualization/dynamics_failure_flags
```

Use a different bag name per ablation. RGB/depth/GT association uses **reference
sensor timestamps**, not bag receipt times. The inherited receiver uses ROS
ApproximateTime and retains the RGB timestamp in `InputData`; the estimator cannot
recover depth/RGB skew after normalization. Correct registration, synchronization,
rectification and intrinsics are therefore upstream requirements. GT poses include
`world_T_body * body_T_sensor`; they and calibration are treated as exact.

## Sliding window (temporal mode)

```text
input        retained history (window_size 5)    reference / output geometry
frame 4      0 1 2 3 4                            frame 4
frame 5        1 2 3 4 5                          frame 5
frame 6          2 3 4 5 6                        frame 6
```

Each retained frame caches its gray image, depth, texture, structure tensor and the
flow pair to its predecessor (current-to-previous for tracking, previous-to-current
for the cycle check). A new input only adds the flow pair to the previous frame; the
two directions run concurrently. The current frame then starts a fresh reference:
every valid pixel is tracked back one cached flow at a time, with look-back time
`t_reference - t_frame` (the reported velocity is negated back to physical time).

Before flow is computed, frames expire once the history would exceed `window_size`
or `max_window_duration`; an expired interval is never bridged. A gap above
`max_frame_gap`, a timestamp rewind or duplicate, a sensor/calibration change or
malformed input clears the history. Outputs are emitted from the first frame;
fits need `min_fit_observations` accepted samples, so with the defaults evidence
appears once four frames are retained.

### Keyframes

`keyframe_interval: K` computes flow and texture for every input but detects motion
only on every K-th input since the history started. Between keyframes the input
only extends the history and `processInput` returns nullptr, so Khronos publishes
no dynamics for that frame. A keyframe's estimate is identical to the one every-frame
detection would produce for that input: the reference is still tracked back through
every retained frame, so the per-step error model and the observations are unchanged.
Only the output rate drops (20/K Hz for a 20 Hz camera). Frame-to-frame motion is
small, so consecutive references mostly repeat each other's evidence. Every input
still costs its flow pair (about 57 ms at 720x480 with Farneback, texture and tensor),
which bounds the saving.

### Flow method

`flow_method` selects Farneback (default) or OpenCV's DIS flow (Dense Inverse Search)
with its `dis_fast` or `dis_medium` preset. Farneback fits local quadratic polynomials
over a Gaussian window and runs on one thread, so the two directions of a pair run as
concurrent tasks. DIS matches a sparse grid of overlapping patches with inverse
compositional Lucas-Kanade, averages them into a dense field and refines it
variationally; it parallelizes internally, so the two directions run in sequence.
Everything downstream of the flow fields is unchanged. The noise, cycle-check and
aperture parameters were tuned with Farneback and have not been retuned for DIS;
`flow_window_size` still sets the aperture structure-tensor window.

On two 200-frame apartment segments DIS FAST computed a flow pair in 7 ms instead of
48 ms, scored more of the image and gave fewer background false positives with higher
human recall (see the [flow comparison](OPTICAL_FLOW_EVALUATION.md#flow-method-comparison-2026-10-12)).

Because the reference is the newest frame, the cloud and its timestamp no longer
age by up to `window_size - 1` frames as in legacy mode; the remaining delay is
processing time plus queueing. The price is that each frame re-runs reference setup,
`window_size - 1` tracking passes, the per-pixel fits and, with the CRF, its pairwise
weights. Evidence is recomputed from scratch for each reference, so a decision can
change from one frame to the next as frames enter and leave the window.

## Measurement conventions and association

Depth is optical-axis depth in meters, not Euclidean range. Range checks apply to
unprojected sensor coordinates. World-frame and sensor-frame reference vertex maps
are both supported. Consecutive subpixel flow fields are composed by bilinear
sampling. No gate compares displacement/depth to the initial point to decide if
an observation is valid: actual lateral and axial motion remain admissible.

Legacy mode preserves the previous 2x2 depth-footprint spread gate and hard
`min_texture` rejection. Temporal mode examines the interpolation footprint plus
a small neighborhood expanded by bounded endpoint uncertainty. Adjacent depth
samples must satisfy

```text
abs(z1 - z2) <= max_depth_jump + depth_jump_relative * min(z1, z2).
```

This detects incompatible layers without rejecting a smooth surface solely because
its total depth spread over the footprint is large. It checks the reference as
well as later observations. Invalid samples and layers are missing evidence:
foreground/background depth is never averaged into a fictitious observation.
Texture edges on a continuous depth surface remain usable. If flow is reliable,
a missing depth comparison can be skipped while continuing the track. Nonfinite
flow, leaving the image or failing the cycle check terminates its correspondence.
Loss is marked in `track_alive`, `failures` and `failure_flags`. A skipped or
terminated observation is missing evidence only: the fit, evidence and validity of
previously accepted observations are kept, not reset to unknown or to static.

### Correspondence ambiguity (temporal mode)

Exact poses predict where a static reference point must appear in frame `t`:
`p_static = project(T_camera_world,t * x_reference)`. Farneback can drift on flat or
repetitive regions while its forward/backward check still passes, because both
directions are consistently wrong. Such a drift looks exactly like coherent motion.
Therefore, when `|p_flow - p_static| > ambiguity_min_separation`, the 5x5 reference
patch is compared with the current image at both locations (mean absolute 8-bit gray
difference `d`). If

```text
d(p_static) <= d(p_flow) + photometric_tolerance,
```

the image does not distinguish the flow correspondence from the static prediction.
The observation gets `kAmbiguousCorrespondence` and is skipped as missing evidence
for **both** fits; the flow track continues. This tests identifiability rather than
rejecting static-hypothesis violations: textured moving surfaces fail the test (the
static location shows different content), and textured static surfaces never reach
it (flow and prediction agree). It is still data-dependent selection that relies on
exact poses and brightness constancy, and it further changes the ideal null law of
the fit statistics. A disagreement that cannot be tested (static prediction behind
the camera, or either patch touching the image border) is also ambiguous: without
this, flat-surface drift reappeared as red patches along the image border. Untextured moving
surfaces become unknown, which is honest: their motion is not observable from flow.

### Visibility of the static prediction (temporal mode)

For every valid reference pixel and every other retained frame, independent of its flow track,
the static prediction `p_static` and its predicted optical-axis depth `z_pred` are
compared with the observed 3x3 depth around `p_static`, with the depth-layer
tolerance `max_depth_jump + depth_jump_relative*z_pred`:

| Observed depth | State | Use |
| --- | --- | --- |
| some sample within tolerance | present | none (also consistent with sideways motion) |
| all valid samples nearer | occluded | the flow observation is skipped as missing evidence (`kOccluded`) |
| all valid samples farther | free space | violation of the static hypothesis, counted |
| mixed, < 5 valid, outside image | untestable | none |

Occlusion explains most former false positives next to moving people: their flow
matches jumped onto the occluder. Depth alone cannot tell an occluder from the same
surface approaching the camera, so approaching surfaces lose flow evidence once
they are about one tolerance nearer; receding ones gain free-space evidence.

Free space is decided **a contrario**: under the static hypothesis a violation arises
only from depth or registration error, at rate `p0 = free_space_false_rate` per test.
With `n = window_size - 1` tests per window and `N` image pixels, the minimum count is
the smallest `k` with

```text
N * P(Binomial(n, p0) >= k) <= free_space_max_false_alarms,
```

i.e. at most that many expected chance detections per image. Defaults give `k = 3` of
4 at 720x480; the measured uHumans2 background rate was about 2e-4 per test, so the
default `p0 = 1e-3` is conservative. Neighboring tests are spatially correlated, so the
false-alarm bound is approximate. A pixel reaching `k` becomes `kDynamic` with evidence
`max(flow evidence, 1 - exp(-violations))` and CRF unary
`max(flow unary, min(crf_max_unary, violations * free_space_log_odds))`. It overrides
static or missing flow results (a static point cannot be seen through), but not a
current poor fit to both models, which stays unknown. Counts are recomputed for each
reference over the retained frames against a threshold fixed by `window_size`.
`free_space_count` and `visibility_tests` are exposed per pixel.

Temporal weak texture records a warning and inflates uncertainty, rather than
stopping the track. The heuristic variance multiplier is:

```text
h = 1 + texture_variance_scale * max(0, 1 - texture/min_texture)
      + fb_variance_scale * (cycle_error/max_forward_backward_error)^2
quality = 1/sqrt(h).
```

The texture is the minimum structure-tensor eigenvalue at the old/new endpoint.
The first observation uses only the texture term. Reference localization and each
new flow-step variance use this multiplier. Track reliability is the mean quality
of accepted observations. This is an uncalibrated confidence heuristic; weak
texture cannot establish material-point identity, even when flow is consistent.

## Anisotropic covariance

Let `q=[(u-cx)/fx, (v-cy)/fy, 1]`, optical-axis depth `z`, and local depth gradient
`g`. On a consistent surface, the unprojection Jacobian is

```text
J = [ (z/fx,0,0)^T + q*g_u, (0,z/fy,0)^T + q*g_v, q ].
```

Depth noise is `sigma_z(z)=depth_sigma+depth_sigma_scale*z^2`. For independent
bilinear footprint samples, its sampled variance is `sum_k w_k^2*sigma_z(z_k)^2`.
The gradient columns include coupling between uncertain pixel location and depth
sampling; this smooth model is never used through a detected depth discontinuity.

Pixel covariance propagates through `A=I+gradient(flow)`:

```text
P_pixel,t = A * P_pixel,t-1 * A^T + h*flow_sigma^2*I + Q_aperture
Q_aperture = aperture_sigma^2 * min_texture * (G + 0.01*min_texture*I)^-1
B = R_world_camera * J
C_t = B[:,0:2]*P_pixel,t*B[:,0:2]^T
      + variance_depth*B[:,2]*B[:,2]^T + position_sigma_floor^2*I.
```

`G` is the 2x2 image structure tensor at the source pixel of the flow step, averaged
over `flow_window_size` and scaled to `cornerMinEigenVal` units (3x3 Sobel, 3x3 block
sum), so `min_texture` has the same meaning in both. Along an image direction whose
eigenvalue equals `min_texture`, the aperture term adds `aperture_sigma` pixels of
flow standard deviation; it decreases as `1/sqrt(eigenvalue)` on textured patches
and is capped at `10*aperture_sigma`. This follows the Lucas-Kanade observation that
flow along directions without image gradient is unobservable. It is a heuristic for
Farneback, whose errors on flat regions are interpolation **bias** rather than
independent noise; it is not a calibrated covariance. Pushed through `A`, an
untextured patch gets large uncertainty along its own surface while depth noise is
not inflated, so sliding along a wall is not motion evidence but motion along the
surface normal still is.

The isotropic texture/cycle multiplier `h` was insufficient on real data: on the
uHumans2 apartment sequence without the aperture term, 48% of scored weak-texture
observations were classified as coherent motion, versus 5% of textured ones.

Temporal covariances are finite, symmetric and strictly positive definite. The
positive 3D floor accounts for unmodeled association error and regularizes the
reference even if localization uncertainty is configured to zero. Accepted
observations are checked with Cholesky; invalid covariances become numerical
failures. Poses are exact. All noise parameters are standard deviations, not
variances; default noise assumptions require validation on the actual sensor.

## Temporal model comparison

All accepted observations, including a valid reference, enter the same two fits:

```text
static:   x_i = a + epsilon_i
velocity: x_i = a + v*(t_i-t_center) + epsilon_i
J_static  = sum_i (x_i-a)^T C_i^-1 (x_i-a)
J_dynamic = sum_i (x_i-a-v*(t_i-t_center))^T C_i^-1 (x_i-a-v*(t_i-t_center))
delta_J = max(0, J_static-J_dynamic).
```

The implementation accumulates fixed-size weighted sufficient statistics. It
translates coordinates by the first accepted point to avoid cancellation, centers
actual timestamps at their mean, and solves 3x3/6x6 systems. It performs no robust
sample removal or hypothesis-dependent reweighting. Both fits use identical accepted
samples and covariance weights. Velocity covariance is the velocity block of the
inverse normal system, computed by solving for its three columns; it is conditional on
the assumed measurement covariances. The fitted static point is a statistical
parameter exposed in debug output, not the visualization geometry.

Under independent Gaussian observations with known covariance, residual DOFs are
`3n-3` and `3n-6`, and `delta_J` has an ideal static-null chi-square(3) law. Here,
**temporal fitting treats marginal point covariances as independent**. Composed-flow
cross-correlations and reused images violate that approximation; association gates
and heuristic weights also alter the null law. No per-pixel dense temporal covariance
is allocated. Static/dynamic fit CDFs are explicitly approximate diagnostics.

A motion decision requires at least `min_fit_observations` (default 4), sufficient
actual temporal span, numerical rank, track reliability, a plausible velocity fit,
model improvement above the penalty, and meaningful displacement:

```text
J_dynamic fit CDF <= max_fit_cdf
|v|*span >= min_displacement
|v| >= min_displacement_sigma * sqrt(v_direction^T C_velocity v_direction)
delta_J > motion_penalty
raw_evidence = 1-exp(-(delta_J-motion_penalty)/motion_score_scale).
```

Fit-plausibility gates use precomputed chi-square quantiles per DOF, equivalent to
comparing the CDF with `max_fit_cdf`; CDF values are computed only for debug output.
The default penalty 11.345 is inspired by the ideal 99% chi-square(3) likelihood
ratio cutoff, not claimed to give 1% real-tracker false positives. Evidence is
bounded and **uncalibrated**, not a posterior or dynamic probability. A plausible
static fit returns evidence zero when meaningful coherent motion is unsupported.
A track poorly explained by both models remains unknown, with `kPoorModelFit`.
Two observations cannot provide a dynamic-fit quality check; short/degenerate
tracks are insufficient. Coherent tracking drift can still resemble real velocity.

## Optional spatial support

`spatial_enabled` turns the stage on; `spatial_method` selects `support` (C++ default)
or `crf` (preset in the supplied mapper files).

### `spatial_method: support`

A single local pass on the reference grid. Neighbors must
have reliable direct evidence, compatible reference RGB/depth and locally compatible
velocity. A checked connecting depth path inhibits crossing thin boundaries.
At least three neighbors and 70% weighted motion/static agreement are required by
default. Reliable direct targets are unchanged; other targets may borrow a weighted
score. Inferred neighbors are never used as sources, so there is no iterative
propagation or independent re-counting. Radius is capped at three pixels.

Raw evidence, regularized evidence and support codes remain distinct. Inference
never increases observation count, track reliability or direct validity. It cannot
fill large unsupported regions, and enforces local compatibility rather than a
single velocity for an entire human. Color/depth similarity cannot guarantee object
identity for coplanar, similarly colored surfaces.

### `spatial_method: crf`

A binary (moving/static) pairwise CRF on the reference pixel grid; every pixel keeps
its own value. Each pixel's own fit gives a signed log-odds unary

```text
moving:  L = reliability * clamp(0.5*(delta_J - motion_penalty),  0.5, crf_max_unary)
static:  L = reliability * clamp(0.5*(delta_J - motion_penalty), -crf_max_unary, -0.5)
unknown, unreliable or poor fit:  L = 0
```

Half the penalized cost difference is a log-likelihood ratio only under the ideal
Gaussian assumptions; the cap keeps overconfident real-data costs from becoming
certainty. Pairwise weights use a sparse stencil (8 directions at 1, 3 and 6 px):

```text
w_ij = exp(-|d_ij|^2 / (2 crf_spatial_sigma^2) - |rgb_i - rgb_j|^2 / (2 spatial_color_sigma^2))
```

and are zero unless every pixel on the connecting segment continues the same depth
layer (`max_depth_jump + depth_jump_relative*z`). They depend only on the reference
frame, so temporal mode recomputes them for every input. `crf_iterations` synchronous mean-field updates

```text
q_i = sigmoid(L_i + crf_pairwise_weight * sum_j w_ij (2 q_j - 1) / sum_stencil w_spatial)
```

let evidence travel about 6 px per iteration along a continuous surface. The
neighborhood contributes at most `crf_pairwise_weight` nats; with the defaults
(3 < `crf_max_unary` 4) unanimous neighbors cannot flip a confident direct pixel.

`regularized_evidence` then holds the marginal `q` (0.5 = undecided), not the
`support` score scale. Pixels with their own evidence get `spatial_support` 1. An
unknown pixel gets a value (`spatial_support` 2) only when its connected, confident
neighbor weight `sum_j w_ij |2q_j - 1| / sum_stencil w_spatial` reaches
`crf_min_support`; otherwise it stays NaN. Raw evidence, validity, reliability and
counts are untouched. Neighbors share flow and image noise, so the CRF is a smoothness
prior, not additional independent observations, and the marginal is not a calibrated
motion probability. Large regions without any connected direct evidence stay unknown.

## Configuration

All parameters are declared and validated through `config_utilities` in
[optical_flow_dynamics_config.cpp](khronos/src/active_window/motion_detection/optical_flow_dynamics_config.cpp).
Defaults live in [OpticalFlowDynamics::Config](khronos/include/khronos/active_window/motion_detection/optical_flow_dynamics.h).
The mapper files contain only preset overrides. Noise values are standard deviations;
calibrate them for the sensor. Older `residual_sigma` settings must use propagated
noise parameters instead.

| Parameter | Default | Units/meaning |
| --- | --- | --- |
| `scoring_mode` | temporal | temporal or legacy |
| `window_size` / `max_window_duration` | 5 / 1 | retained frames including the reference / seconds |
| `max_frame_gap` | 0.2 | seconds; gaps/rewinds/duplicates clear the history |
| `keyframe_interval` | 1 | inputs per motion detection, temporal only (test preset: 3) |
| `reference_pixel_sigma` / `flow_sigma` | 0.5 / 0.5 | pixels, per axis |
| `depth_sigma` / `depth_sigma_scale` | 0.01 / 0.001 | m / 1/m |
| `position_sigma_floor` | 0.001 | m; temporal only |
| `max_depth_jump` / `depth_jump_relative` | 0.1 / 0.02 | m / dimensionless, adjacent-layer tolerance |
| `depth_neighborhood_radius` | 1 | pixels; uncertainty expands it, capped at 3 |
| `min_texture` | 0.0001 | structure-tensor eigenvalue |
| `max_forward_backward_error` | 1 | pixels; correspondence consistency |
| `texture_variance_scale` / `fb_variance_scale` | 4 / 1 | dimensionless, heuristic |
| `aperture_sigma` | 1 | px at `min_texture`; structure-tensor flow uncertainty, 0 disables |
| `ambiguity_check` | true | skip flow correspondences the static prediction explains equally well |
| `ambiguity_min_separation` / `photometric_tolerance` | 1 / 3 | px / mean absolute 8-bit gray level over 5x5 patches |
| `visibility_check` | true | occlusion skipping and free-space evidence |
| `free_space_false_rate` / `free_space_max_false_alarms` | 0.001 / 1 | per-test chance rate / expected false detections per image |
| `free_space_log_odds` | 1.5 | nats per violation (CRF unary) |
| `min_fit_observations` / `min_temporal_span` | 4 / 0.1 | accepted samples including reference / s |
| `min_reliability` / `max_fit_cdf` | 0.25 / 0.99 | heuristic confidence / approximate fit diagnostic |
| `motion_penalty` / `motion_score_scale` | 11.345 / 10 | dimensionless |
| `min_displacement` / `min_displacement_sigma` | 0.02 / 3 | m / standard deviations |
| `spatial_enabled` / `spatial_radius` | false / 2 | ablation / pixels |
| `spatial_method` | support | `support` or `crf` (mapper presets use `crf`) |
| `crf_iterations` / `crf_pairwise_weight` / `crf_max_unary` | 5 / 3 / 4 | count / nats / nats |
| `crf_spatial_sigma` / `crf_min_support` | 4 / 0.2 | px / fraction of stencil weight |
| `spatial_min_neighbors` / `spatial_motion_fraction` | 3 / 0.7 | count / weighted agreement |
| `spatial_min_reliability` | 0.6 | minimum direct source reliability |
| `spatial_color_sigma` / `spatial_velocity_tolerance` | 30 / 1 | RGB8 Euclidean difference / m/s |
| `debug_output` | false | add fit/velocity diagnostics and debug matrices |
| `min_observations` | 2 | legacy accepted comparisons after the reference |
| `flow_method` | farneback | `farneback`, `dis_fast` or `dis_medium` (test preset: `dis_fast`) |
| `flow_levels` / `flow_window_size` / `flow_iterations` | 3 / 15 / 3 | Farneback parameters; the window also sizes the aperture tensor |

## Outputs and compatibility

All images and organized-cloud points use **reference pixels**; every snapshot
retains both reference and latest timestamps in C++. ROS headers carry the
reference timestamp, which in temporal mode is the current input's. Missing evidence
is NaN, confident static evidence is zero. Colored evidence/RViz uses gray unknown,
blue static, red increasing motion evidence. History bounds and resets are described
under [Sliding window](#sliding-window-temporal-mode). Partial windows are emitted; no
end-of-stream flush is required. Estimator runtime is logged by Hydra's timing
recorder as `active_window/dynamics`.

`probability` and `/dynamics_probability` are deprecated legacy names. They retain
the previous fixed-reference approximate CDF only in legacy mode. In temporal mode
the compatibility field is NaN and that image topic is not published; no new
semantics are silently assigned to it. Legacy uses the preceding five-state
conditional noise scorer and hard texture rejection, with three residual DOFs per
accepted comparison. It remains an approximate static-consistency diagnostic.

Relative topics under `/khronos_node/visualization/`:

- `dynamics_cloud`: reference XYZ, legacy `probability`, post-reference `observations`,
  `scoring_mode` (0 legacy/1 temporal), `motion_evidence`, `regularized_evidence`,
  `reliability`, `validity`, `spatial_support`, `accepted_observations` including
  reference, `temporal_span` in s, cumulative `failure_flags`, `failures`,
  `track_alive`, `rgba` and `free_space_count`. Core layout: 72 bytes/point.
- `dynamics_motion_evidence` / `dynamics_regularized_evidence`: 32FC1 raw/inferred scores.
- `dynamics_evidence`: RGB8 colored evidence; `dynamics_reliability`: 32FC1 heuristic confidence.
- `dynamics_validity`: mono8 direct states: 0 unavailable, 1 insufficient, 2 unreliable,
  3 static-compatible, 4 coherent motion. `spatial_support`: 0 none, 1 direct, 2 inferred.
- Existing failure-flags, colored failure and track-alive image topics remain available.
  Weak texture can be a warning on an accepted temporal observation. Bits are cumulative:
  1 invalid reference, 2 weak texture, 4 inconsistent flow, 8 bounds, 16 invalid depth,
  32 depth discontinuity, 64 range, 128 nonfinite flow, 256 numerical failure,
  512 poor model fit, 1024 ambiguous correspondence, 2048 occluded static prediction.
  Failure colors, by priority: gray invalid reference, magenta lost track (bounds, cycle
  check, nonfinite flow, numerical), orange poor fit to both models, yellow weak texture
  or ambiguous, cyan depth or occlusion, black none.
  Historical flags do not override current validity.

With `debug_output`, the cloud additionally includes flow cycle error (px), last raw
reference displacement (m), legacy innovation sigma (m), static/dynamic costs,
improvement, approximate fit CDFs, fitted static XYZ, world velocity XYZ (m/s), its
six symmetric covariance entries, temporal measurement sigma (m) and the number of
conclusive visibility tests. Debug layout: 160 bytes/point. `innovation_sigma` remains a legacy diagnostic; temporal point
covariance trace is named `measurement_sigma`. No fit CDF is called motion probability.
C++ snapshots expose the same matrices; fit and velocity-covariance matrices are
allocated only in debug mode. Velocity itself is retained for local support.

Temporal sufficient statistics plus 2x2 pixel covariance use about 368 bytes/pixel
(127 MB at 720x480), independent of window length. The frame history adds about
38 bytes/pixel per retained frame (gray, depth, depth-layer state, texture, structure tensor and two
flow fields: 66 MB for five 720x480 frames), in addition to immutable snapshots
retained by Khronos's bounded frame buffer. Legacy state is 240 bytes/pixel plus one
frame. There are no per-track heap histories or dense temporal covariance matrices.
Per-frame tracking is O(pixels*(window_size-1)), fitting O(pixels), optional support
O(pixels*radius^2); serialization is subscriber-driven. Debug payloads (and legacy
snapshot copies) are material costs, so keep debug off for production.

## Scope and references

This lightweight combination is not a reproduction of a cited system. Depth/flow
association errors, coherent drift, textureless material-point ambiguity, true
acceleration/deformation, uncertain calibration and covariance miscalibration remain
limitations. Human semantics are only a proxy for motion during rosbag evaluation.

- [Probabilistic RGB-D Odometry under depth uncertainty](https://arxiv.org/abs/1706.04034)
- [StaticFusion](https://raluca-scona.github.io/docs/conference-papers/2018_icra_staticfusion.pdf)
- [Multiframe Scene Flow](https://arxiv.org/abs/1710.02124)
- [SF2SE3](https://arxiv.org/abs/2209.08532)

## Code layout and extraction boundary

| File | Responsibility |
| --- | --- |
| `motion_detection/optical_flow_dynamics.cpp` | Input validation, window lifecycle, flow tracking, evidence and snapshots |
| `motion_detection/optical_flow_dynamics_config.cpp` | `config_utilities` field declarations and validation |
| `motion_detection/optical_flow_spatial.cpp` | Reference-grid support and CRF passes |
| `motion_detection/temporal_motion_model.{h,cpp}` | Eigen-only sufficient statistics and static/velocity fits |
| `motion_detection/motion_uncertainty.{h,cpp}` | Eigen-only RGB-D Jacobian and legacy uncertainty propagation |
| `motion_detection/dynamics_estimator.h` | Input/output interface and evidence definitions |
| `khronos_ros/.../dynamics_visualization.{h,cpp}` | ROS messages and subscriber-driven publishing |

The numerical models have no ROS or map dependencies. The estimator still depends
on Hydra's `InputData`, camera calibration, timestamps and pose conventions; the
spatial helpers currently use its configuration and estimate types. A separate
repository can carry these detector files and tests together, then replace the
Hydra input adapter and retain the `config_utilities` factory. Khronos integration
remains in `ActiveWindow`/`FrameData`; it is not part of the numerical method.

Tracking and spatial passes partition independent rows through OpenCV's existing
worker pool. CRF updates remain synchronous: workers finish before buffers swap.
No detector-specific thread parameter or global thread setting is introduced; Khronos
does not set one either, so OpenCV uses all cores. Farneback itself is single-threaded,
so the newest frame's two flow directions run as two concurrent tasks. Flow, texture
and tensors are computed once per input and reused by every later reference while
the frame is retained. Pixels without enough accepted samples update counts, span and
reliability without solving unusable fits; debug mode still computes partial
static-fit diagnostics. Static projection and texture samples are reused within each
pixel. Aperture tensors are computed only when enabled. Spatial reference RGB/depth
copies are retained only when spatial support is enabled. References with zero CRF
unaries skip weight construction and iterations when the support threshold is
positive: all marginals would remain 0.5 with no support. CRF weights are symmetric,
so each pair is computed once and mirrored, and a stencil segment's depth continuity
is read from per-pixel unit-step link maps instead of being re-tested per pair.
Each frame stores a one-byte depth-layer state per pixel (valid depth, layer break to
the right and below neighbor), so the temporal depth-footprint test scans bytes in the
original order instead of re-evaluating depth tolerances for every observation. The
ambiguity check reads the 5x5 reference patch once from integer pixels and compares
both candidate locations against it. 2x2 and 3x3 covariances are inverted with
`choleskyInverse`, which repeats the arithmetic of Eigen 3.4's `LLT::solve(Identity)`
without its blocked-kernel overhead; a test pins it to Eigen bit for bit.
Per-pixel fit state is cleared with a counter rather than reallocated. A temporal
snapshot takes ownership of the output images, which the next keyframe allocates
again; legacy snapshots copy them because legacy windows update in place. Either way
a snapshot is never modified after it is returned.

See the [refactor](OPTICAL_FLOW_EVALUATION.md#implementation-refactor-benchmark-2026-10-12)
and [sliding-window](OPTICAL_FLOW_EVALUATION.md#sliding-window-runtime-2026-10-12)
benchmarks for measured runtime and output-equivalence checks.

## Build and checks

Recorded-bag evaluation instructions and historical accuracy measurements are in
[OPTICAL_FLOW_EVALUATION.md](OPTICAL_FLOW_EVALUATION.md).

### Estimator build and checks

The implementation adds OpenCV's `video` component and uses installed Boost.Math
headers. All estimator options are declared and validated with `config_utilities`;
`OpticalFlowDynamics` is registered in its virtual factory.

With the normal Khronos dependencies installed:

```bash
colcon build --packages-select khronos khronos_ros --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
colcon test --packages-select khronos --ctest-args -R optical_flow_dynamics
```

`test_optical_flow_dynamics` exercises real OpenCV flow and Hydra camera/extrinsics,
including legacy regressions, static camera motion, coherent velocity, a biased
reference, retention of evidence after a missing final depth sample or track loss,
depth jumps, poor fits, short/degenerate tracks, uncertainty scaling,
continuous sloped surfaces, missing depth, reference indexing and spatial leakage.
Additional regressions compare serial/parallel execution and debug/normal evidence
through partial and completed windows, and check zero-evidence CRF behavior. Sliding
tests check that each output uses the current frame's geometry and timestamp, that
old frames expire by count and duration, and that gaps, duplicates and rewinds clear
the history. A DIS test checks that both presets separate a moving surface from
a static scene. A keyframe test checks that keyframe estimates equal every-frame
detection and that other inputs return no estimate. A stripe test checks
that the aperture problem does not become coherent motion (and that it does when
`aperture_sigma: 0`); the structure tensor is checked against `cornerMinEigenVal`. Numerical
tests check the unprojection Jacobian by finite differences, compare sequential
whitening against an independently assembled joint covariance (including a skipped
observation), and check chi-square moments on 3,000 simulated static sequences.
