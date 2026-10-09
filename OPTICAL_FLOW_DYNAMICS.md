# Optical-flow dynamics evidence

`OpticalFlowDynamics` is an optional `config_utilities`-registered Khronos module.
It tracks registered RGB-D input through non-overlapping short windows using
Farneback. It never consults the Khronos map. First-frame world points remain the
output geometry; temporal mode fits a statistical static point and a velocity
without modifying that geometry. Results live in `FrameData::dynamics`, reach
existing sinks, and have no effect on Khronos's binary `dynamic_image` mask.

## Launch and ablation

Inside Docker, from `/root/ros2_ws`, after building and sourcing the workspace:

```bash
# New temporal scorer, direct measurements only (default).
ros2 launch khronos_ros optical_flow_dynamics_test.launch.yaml \
  bag_path:=/data/tesse_cd_apartment scoring_mode:=temporal spatial_enabled:=false

# Same scorer, optional local spatial support.
ros2 launch khronos_ros optical_flow_dynamics_test.launch.yaml \
  bag_path:=/data/tesse_cd_apartment scoring_mode:=temporal spatial_enabled:=true

# Previous scorer, unchanged hard texture gate and fixed-reference CDF.
ros2 launch khronos_ros optical_flow_dynamics_test.launch.yaml \
  bag_path:=/data/tesse_cd_apartment scoring_mode:=legacy spatial_enabled:=false
```

Add `rviz:=false` for headless operation, `play_rate:=0.5` for slower playback, and
`debug_output:=true` for fit/velocity diagnostics in the cloud. Publishers only
serialize subscribed topics. Use `dataset:=tesse_cd_office` for the office bag.
The standalone setup supplies GT poses and disables binary object/motion detection,
tracking and change detection. The background Khronos pipeline still runs. Input
requires RGB, RGB-aligned depth, camera info, GT TF, and GT labels for the existing
closed-set receiver. An existing mapper can merge
[khronos_ros/config/mapper/optical_flow_dynamics.yaml](khronos_ros/config/mapper/optical_flow_dynamics.yaml).

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

For every valid reference pixel and every later frame, independent of its flow track,
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
current poor fit to both models, which stays unknown. Counts never decrease and the
threshold is fixed per window, so a decision does not revert. `free_space_count` and
`visibility_tests` are exposed per pixel.

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
inverse normal system, computed by solving for the identity; it is conditional on
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
or `crf` (default in the supplied mapper files and test launch).

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
frame and are computed once per window. `crf_iterations` synchronous mean-field updates

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

All parameters are validated through `config_utilities`. The old `residual_sigma`
was already replaced by propagated noise parameters; custom configs should use
the settings below. The supplied mapper files include all new defaults.

| Parameter | Default | Units/meaning |
| --- | --- | --- |
| `scoring_mode` | temporal | temporal or legacy |
| `window_size` / `max_window_duration` | 5 / 1 | frames including reference / seconds |
| `max_frame_gap` | 0.2 | seconds; gaps/rewinds/duplicates reset the window |
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
| `spatial_method` | support | `support` or `crf` (mapper files and launch use `crf`) |
| `crf_iterations` / `crf_pairwise_weight` / `crf_max_unary` | 5 / 3 / 4 | count / nats / nats |
| `crf_spatial_sigma` / `crf_min_support` | 4 / 0.2 | px / fraction of stencil weight |
| `spatial_min_neighbors` / `spatial_motion_fraction` | 3 / 0.7 | count / weighted agreement |
| `spatial_min_reliability` | 0.6 | minimum direct source reliability |
| `spatial_color_sigma` / `spatial_velocity_tolerance` | 30 / 1 | RGB8 Euclidean difference / m/s |
| `debug_output` | false | add fit/velocity diagnostics and debug matrices |
| `min_observations` | 2 | legacy accepted comparisons after the reference |
| `flow_levels` / `flow_window_size` / `flow_iterations` | 3 / 15 / 3 | Farneback parameters |

## Outputs and compatibility

All images and organized-cloud points use **reference pixels**; every snapshot
retains both reference and latest timestamps in C++. ROS headers retain the
reference timestamp. Missing evidence is NaN, confident static evidence is zero.
Colored evidence/RViz uses gray unknown, blue static, red increasing motion evidence.
Window size/time limits, gaps, rewinds, sensor changes and malformed input reset
bounded state. Partial windows are emitted; no end-of-stream flush is required. Estimator runtime
is logged by Hydra's timing recorder as `active_window/dynamics`.

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

State is fixed-size per pixel, independent of window length: temporal sufficient
statistics plus 2x2 pixel covariance use about 368 bytes/pixel (127 MB at 720x480)
plus a 12-byte/pixel previous-frame structure tensor,
in addition to current images and immutable snapshots retained by Khronos's bounded
frame buffer. Legacy state is 240 bytes/pixel. There are no per-track heap histories
or dense temporal covariance matrices. Per-frame tracking/fitting is O(pixels),
optional support O(pixels*radius^2); serialization is subscriber-driven. Snapshot
copies and debug payloads are material costs, so keep debug off for production.

## Measured behavior (uHumans2 apartment, 2026-10-09)

Full `tesse_cd_apartment` replay through `optical_flow_dynamics_test.launch.yaml`
(GT poses, `play_rate:=0.5`, 1739 frames, about 348 five-frame windows), evaluated
with `evaluate_dynamics_bag.py` at its default per-mode threshold (legacy CDF > 0.99,
temporal evidence > 0.5). "Human" is the semantic proxy described above, not a motion
annotation. Scored FPR counts background pixels with a usable score.

| Run | Coverage | Human coverage | Precision | Recall all / interior | Scored background FPR: all / static edge / smooth | AP (scored) |
| --- | --- | --- | --- | --- | --- | --- |
| legacy | 11.4% | 3.4% | 6.0% | 3.2% / 1.2% | 9.6% / 11.5% / 8.0% | 0.070 |
| temporal, `aperture_sigma: 0`, `ambiguity_check: false` | 58.9% | 28.1% | 2.3% | 24.2% / 29.6% | 38.4% / 9.9% / 42.1% | 0.023 |
| temporal, aperture only | 57.8% | 27.4% | 2.6% | 22.7% / 27.8% | 32.1% / 9.4% / 34.8% | 0.027 |
| temporal, defaults | 21.5% | 10.5% | 9.8% | 9.7% / 11.5% | 9.2% / 1.8% / 10.8% | 0.104 |
| temporal, defaults, `regularized_evidence` | 24.1% | 11.8% | 10.2% | 10.8% / 12.3% | 8.7% / 1.8% / 10.6% | 0.111 |

The legacy hard texture gate leaves 3.4% of human pixels scored and responds on
static geometric edges. Without the ambiguity check, temporal mode scores far more of
the scene but classifies flat, weakly textured static surfaces as moving (Farneback
drift under camera rotation; median apparent speed 0.8-1.4 m/s against 0.4-0.5 m/s
camera translation). With the defaults, static-edge false positives fall from 11.5%
(legacy) to 1.8% of scored edge pixels, human recall triples and precision improves.
Background within 3 px of humans remains the dominant false-positive region (60%
scored FPR): occlusion and disocclusion by moving people change correspondences there.

On a 200-frame human segment, `min_reliability: 0.5` reduced scored background FPR
from 12% to 2.7% but human recall from 12.7% to 1.8%: the humans are themselves weakly
textured. `photometric_tolerance` 6 or 10 lowered FPR only marginally while losing
human coverage. Defaults keep the higher-recall setting; raise `min_reliability` for a
conservative output.

Estimator-only runtime on the same 720x480 frames (Release build, single thread):
legacy 92 ms/frame, temporal 175 ms, plus spatial support 189 ms, plus debug output
200 ms. Neither mode keeps up with 20 Hz input; the replay pipeline queued frames
and drained them after playback rather than dropping them in these runs.

After the border rule (untestable correspondences are ambiguous) and with the CRF,
on the 200-frame human segment through the offline harness (regularized output
counted as moving when > 0.5; background rows over scored pixels, human recall over
all human pixels):

| Spatial stage | Background scored | Background FPR scored / all | Human recall all / interior | ms/frame |
| --- | --- | --- | --- | --- |
| none (raw) | 17.3% | 2.0% / 0.34% | 11.3% / 13.1% | 192 |
| `support` | 20.3% | 2.1% / 0.42% | 12.0% / 13.5% | 192 |
| `crf`, `crf_pairwise_weight` 1.5 | 24.6% | 2.5% / 0.61% | 21.6% / 23.2% | 271 |
| `crf`, 3 (default) | 35.1% | 3.4% / 1.19% | 37.7% / 39.1% | 271 |
| `crf`, 5 | 45.3% | 4.4% / 1.99% | 50.7% / 51.7% | 271 |

On the static segment, `crf` at 3 raised all-pixel background FPR from 0.29% to 1.08%.
`crf_pairwise_weight` is the recall/false-positive knob. The CRF adds about 80 ms per
frame (single thread); the full-sequence table above predates these changes.

Adding the visibility test (defaults, same segments and counting):

| Segment / output | Background FPR scored / all | Human recall all / interior |
| --- | --- | --- |
| humans, raw, without -> with visibility | 2.0% / 0.34% -> 1.4% / 0.25% | 11.3% / 13.1% -> 31.2% / 29.1% |
| humans, `crf` 3, without -> with | 3.4% / 1.19% -> 2.3% / 0.79% | 37.7% / 39.1% -> 53.7% / 52.7% |
| static, raw, without -> with | 1.5% / 0.29% -> 0.8% / 0.15% | n/a |
| static, `crf` 3, without -> with | 2.7% / 1.08% -> 1.3% / 0.53% | n/a |

False positives within 10 px of people fell from 21% to 1% of raw background false
positives. The remainder lies mostly on smooth surfaces where depth says the static
point is present but appearance changed (e.g. shading or shadows from people), and
on static depth edges in the camera-motion segment. Visibility adds about 5 ms/frame.

These numbers come from one synthetic sequence with exact poses and depth. They are
not a calibration and do not transfer automatically to real sensors.

## Scope and references

This lightweight combination is not a reproduction of a cited system. Depth/flow
association errors, coherent drift, textureless material-point ambiguity, true
acceleration/deformation, uncertain calibration and covariance miscalibration remain
limitations. Human semantics are only a proxy for motion during rosbag evaluation.

- [Probabilistic RGB-D Odometry under depth uncertainty](https://arxiv.org/abs/1706.04034)
- [StaticFusion](https://raluca-scona.github.io/docs/conference-papers/2018_icra_staticfusion.pdf)
- [Multiframe Scene Flow](https://arxiv.org/abs/1710.02124)
- [SF2SE3](https://arxiv.org/abs/2209.08532)

## Build and checks

### Offline evaluation of a recorded ROS 2 bag

Run inside Docker, from the workspace directory containing `src/` and the mounted
`datasets/` folder. The evaluator needs the recorded results **and** the original
dataset bag containing the semantic GT topic:

```bash
python3 src/khronos/khronos_ros/scripts/evaluate_dynamics_bag.py \
  --results-bag datasets/dynamics_eval \
  --ground-truth-bag datasets/tesse_cd_apartment \
  --dataset tesse_cd_apartment \
  --output datasets/dynamics_eval_report
```

Replace the original bag path with your actual directory. Both bags must be ROS 2
directories containing `metadata.yaml`. Source ROS and the workspace first. After
building `khronos_ros`, the script is also available through
`ros2 run khronos_ros evaluate_dynamics_bag.py` with the same arguments.

Default topics are `/khronos_node/visualization/dynamics_cloud` in the results bag
and `/tesse/seg_cam/converted/image_raw` in the original bag. Check the latter with
`ros2 bag info`; use `--labels-topic` if its name differs. It must contain scalar
label IDs, rather than RGB semantic colors. If the converted label topic was
produced during replay rather than stored in the original bag, record it alongside
the dynamics cloud and use that recording as `--ground-truth-bag`.

The script applies the selected dataset's Hydra label remap and reads its
`dynamic_labels`: apartment humans are mapped label 20; office humans are label 17.
Use `--dataset tesse_cd_office` for office. If the GT image already contains mapped
semantic IDs, add `--labels-are-remapped`. For custom label spaces, supply
`--label-remap /path/to/remap.yaml --foreground-labels ID ...`. A binary mask with
foreground 1 and background 0 needs `--labels-are-remapped --foreground-labels 1
--ignore-labels` (empty ignore list), so background 0 is included.

The script makes two passes over the result clouds and retains the **last recorded
snapshot per reference header timestamp**. It matches GT image header timestamps
exactly by default, preserving reference-frame pixel indexing. A partial final
window uses its last available snapshot. Match sensor timestamps rather than bag
receipt times. `--timestamp-tolerance-ms` permits nearest matches when necessary;
keep this small and inspect `gt_offset_ns` in the report. Unmatched windows are
counted and skipped; no matches fail evaluation. Relevant GT masks are compressed
into a temporary disk cache, removed after evaluation, to avoid retaining a long
sequence in RAM. Neither input bag is modified.

An eligible pixel has a finite first-frame XYZ and a valid mapped GT label. Unknown
label 0 and unmapped IDs are excluded by default. Cloud points are world coordinates;
their distance from the world origin is not used as a sensor range filter. A score
is usable when finite and its observation count reaches `--min-observations`
(default 2; match your estimator config).

Reports are written to a new/empty directory:

| File | Contents |
| --- | --- |
| `summary.json` | Matched/skipped windows, foreground/background coverage, failure rates, track survival and approximate scored-pixel AP |
| `threshold_metrics.csv` | Exact precision, recall, F1, IoU and background FPR at mode-specific thresholds, for all regions |
| `precision_recall.csv` | Histogram-based PR curve with scored-only and full recall; default 10,000 score bins |
| `windows.csv` | Exact threshold metrics and interior/boundary recall per reference window, plus GT timestamp offsets |

The evaluator auto-detects `scoring_mode` and selects `motion_evidence` for temporal
mode or `probability` for legacy. Use `--score-field regularized_evidence` to evaluate
spatial inference separately. Temporal thresholds default to 0.25, 0.5, 0.75;
legacy defaults remain 0.95, 0.99, 0.999. Temporal scores have no nominal static
exceedance rate; that CSV column is empty. Inferred pixels are eligible for spatial
segmentation metrics but do not become direct observations or reliable tracks.

`recall_all` includes unknown foreground pixels as missed detections;
`recall_scored` evaluates only pixels with usable scores. Report both with coverage.
Similarly, compare `background_fpr_scored` with `background_fpr_all`, since rejecting
background pixels can lower the latter. For legacy noise calibration, compare scored
background exceedance with `ideal_static_exceedance = 1 - threshold`; actual rates
can differ because the assumed noise model and selection gates are approximate.

Foreground interior is an erosion of the GT foreground mask by a configurable
radius (default 3 pixels); the removed foreground band is its boundary. Background
within the same dilation radius is reported separately from far background.
`background_geometric_edge` / `background_geometric_smooth` split `background_far`
by whether a reference pixel lies within the same radius of a static geometric
discontinuity: a 4-neighbor world-point jump above `--geometric-edge-threshold`
(default 0.1 m) or an invalid neighbor. Their scored FPRs isolate static-boundary
false positives from human-adjacent ones.
Foreground/background and interior/boundary failure fractions expose where
tracking evidence is lost. Cumulative failure bits can overlap, so their fractions
do not sum to one. Each reference window receives pixel-weighted aggregation;
`windows.csv` permits separate per-window analysis.

These labels evaluate semantic foreground as a **proxy for movement**. A stationary
human still belongs to that foreground, and moving objects outside the chosen
classes are labeled background. Use actual motion masks to establish true motion
accuracy. AP/PR are quantized by `--score-bins`; fixed-threshold counts are exact.
The script does not estimate detection latency or motion trajectories from these
messages, whose header carries only the reference timestamp.

Tests, including generated-bag integration, can be run with ROS sourced:

```bash
python3 -m unittest discover -s src/khronos/khronos_ros/tests -v
```

### Estimator build and checks

The implementation adds OpenCV's `video` component and uses installed Boost.Math
headers. All estimator options are declared and validated with `config_utilities`;
`OpticalFlowDynamics` is registered in its virtual factory.

With the normal Khronos dependencies installed:

```bash
colcon build --packages-select khronos khronos_ros --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select khronos --ctest-args -R optical_flow_dynamics
```

`test_optical_flow_dynamics` exercises real OpenCV flow and Hydra camera/extrinsics,
including legacy regressions, static camera motion, coherent velocity, a biased
reference, retention of evidence after a missing final depth sample or track loss,
depth jumps, poor fits, short/degenerate tracks, uncertainty scaling,
continuous sloped surfaces, missing depth, reference indexing and spatial leakage. A stripe test checks
that the aperture problem does not become coherent motion (and that it does when
`aperture_sigma: 0`); the structure tensor is checked against `cornerMinEigenVal`. Numerical
tests check the unprojection Jacobian by finite differences, compare sequential
whitening against an independently assembled joint covariance (including a skipped
observation), and check chi-square moments on 3,000 simulated static sequences.
