# Optical-flow dynamics evaluation

See [OPTICAL_FLOW_DYNAMICS.md](OPTICAL_FLOW_DYNAMICS.md) for the method, current
launch interface, configuration and code layout. The historical measurements below
cover earlier stages of the same method; their timings predate the October 12
implementation refactor and are not current throughput claims.

## Offline evaluation of a recorded ROS 2 bag

Run inside Docker, from the workspace directory containing `src/`, with bags mounted
under `/data`. The evaluator needs the recorded results **and** the original
dataset bag containing the semantic GT topic:

```bash
python3 src/khronos/khronos_ros/scripts/evaluate_dynamics_bag.py \
  --results-bag /data/output/temporal_run \
  --ground-truth-bag /data/tesse_cd_apartment \
  --dataset tesse_cd_apartment \
  --output /data/output/temporal_run_report
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

## Implementation refactor benchmark (2026-10-12)

Compared with commit `5eab706`, using the same `khronos:jazzy` container, Release
build and OpenCV 4.6. Each sample processes five preconstructed 640x480 RGB-D frames
of a textured plane at 2 m, shifted by 2.5 px/frame at 50 ms intervals, with exact
stationary poses. Timings include `processInput()` and its snapshot copies, exclude
input generation/ROS, and report the median of three windows after one warm-up.
Debug output is off. Thread counts below are OpenCV worker-pool settings in the
benchmark, not new detector parameters.

| Scorer | OpenCV threads | Before, ms/window | After, ms/window | Time reduction |
| --- | --- | --- | --- | --- |
| Temporal | 1 | 818.9 | 755.9 | 8% |
| Temporal | 2 | 894.0 | 601.2 | 33% |
| Temporal | default on this machine (20) | 808.8 | 432.2 | 47% |
| Temporal + CRF | 2 | 1250.8 | 620.3 | 50% |

These short synthetic runs measure implementation cost; load and image content
change the timings. They are not full-bag throughput or accuracy measurements.
Parallel rows provide most of the temporal speedup; avoiding early fits also helps
with one thread. CRF additionally skips unsupported early windows.

The detector's 39 regression tests pass, including serial/parallel equivalence,
unchanged evidence with debug output enabled, snapshot retention, and CRF's zero
support-threshold case. A separate before/after comparison used 128x96 versions of
the synthetic sequence with debug enabled: all 560 matrices (28 fields × 5 frames
× temporal/legacy/support/CRF) matched exactly, including missing-evidence masks.
The two simplified mapper presets also retain all their previous effective settings.
Full dataset replay was not repeated for this refactor.

## Sliding-window runtime (2026-10-12)

Steady-state cost per input after temporal mode switched to a current-frame reference
with a five-frame sliding history. Same container and Release build; 30 synthetic
720x480 frames (uHumans2 intrinsics) of a static textured background at 3 m with a
textured 120x200 px box at 2 m moving 3 px/frame, stationary exact poses. Times are
the median over frames 6-30 of three runs, including snapshot copies; debug off.
"Concurrent flow" is the version that runs the two Farneback directions at once and
solves only the velocity block of the fit covariance.

| Scorer | OpenCV threads | Sequential flow, ms/frame | Concurrent flow, ms/frame |
| --- | --- | --- | --- |
| Legacy | 2 | 151 | 115 |
| Legacy | 20 (default) | 104 | 67 |
| Temporal | 2 | 358 | 318 |
| Temporal | 20 (default) | 181 | 146 |
| Temporal + CRF | 2 | 472 | 427 |
| Temporal + CRF | 20 (default) | 211 | 177 |

With `keyframe_interval` (flow on every input, detection on every K-th; same scene,
20 threads, mean over all inputs after the first five, including non-keyframes):

| Scorer | K=1 | K=3 | K=5 |
| --- | --- | --- | --- |
| Temporal, ms/input | 151 | 86 | 72 |
| Temporal + CRF, ms/input | 184 | 98 | 77 |

The worst single input stays near the K=1 cost (a keyframe: 200 ms with CRF at K=3).
Keyframe estimates are byte-identical to every-frame detection at the same input.

At 20 threads with CRF, a stage breakdown of the concurrent version is about 57 ms
frame preparation (one Farneback, 43 ms, plus texture/tensor), 19 ms reference setup,
55 ms for the four backward tracking passes, 17 ms fits, 39 ms CRF and 3 ms
snapshot. Tracking, fits and CRF weights are per-reference work that the earlier
non-overlapping window amortized over five frames.

All 41 detector tests pass. With debug output on, all 336 output matrices (28 fields
x 12 frames, temporal + CRF) were identical before and after the concurrent-flow change.

In the full pipeline (`tesse_cd_apartment`, temporal + CRF, 20 OpenCV threads) the
detector took 205 ms per frame and the whole active window 249 ms (max 370 ms). At
`play_rate:=0.5` a frame arrives every 100 ms of wall time. Hydra's bounded queues
block rather than drop, so unprocessed frames accumulated upstream and the processed
input fell about 0.3 s of sensor time further behind per second. After about 130 s
the backlog exceeded the 30 s TF buffer: every pose lookup then failed after five
0.1 s retries, and the input never recovered. The test mapper now keeps 300 s of TF,
detects on every third frame and the test launch defaults to `play_rate:=0.25`
(200 ms per frame); check `active_window/all` in `timing_stats.csv` after a replay. Full-bag
accuracy has not been re-measured with the sliding window; the tables below were
recorded with non-overlapping windows.

## Flow method comparison (2026-10-12)

Offline harness: the estimator runs directly on two 200-frame `tesse_cd_apartment`
segments with exact GT poses: camera motion only (starting 20 s into the bag) and
camera motion with people (from 39 s; humans cover up to 14% of a frame). Every
output with a full five-frame window (196 per segment) is scored against that frame's
GT human mask, a proxy for motion. Temporal scoring with the CRF, all other settings
default and tuned with Farneback; Release build, 20 OpenCV threads, sliding window.
Coverage is the fraction of pixels with a static or dynamic decision; raw FPR is
over scored background pixels and over all background pixels; CRF counts
`regularized_evidence > 0.5`; recall is over all human pixels, interior over humans
eroded by 3 px.

| Segment | Flow | Flow pair ms | Estimator ms/frame (K=1 / K=3) | Background coverage % | Raw FPR scored / all % | CRF FPR all % | Human coverage % | Raw recall / interior % | CRF recall / interior % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camera only | Farneback | 48.6 | 168 / - | 19.7 | 0.83 / 0.165 | 0.587 | - | - | - |
| camera only | DIS fast | 7.1 | 134 / - | 37.9 | 0.15 / 0.059 | 0.222 | - | - | - |
| camera only | DIS medium | 22.7 | 164 / - | 52.6 | 0.21 / 0.111 | 0.359 | - | - | - |
| with people | Farneback | 46.8 | 188 / 93 | 17.6 | 1.44 / 0.254 | 0.786 | 26.0 | 25.6 / 24.3 | 44.5 / 43.6 |
| with people | DIS fast | 7.1 | 147 / 57 | 41.2 | 0.26 / 0.106 | 0.344 | 30.2 | 29.5 / 28.6 | 49.9 / 49.7 |
| with people | DIS medium | 23.3 | 170 / 75 | 54.1 | 0.58 / 0.313 | 0.891 | 33.1 | 32.4 / 31.9 | 55.0 / 55.1 |

DIS FAST is better than Farneback on every metric here: about twice the scored
background, 36-44% of Farneback's all-pixel false-positive rate (raw and CRF) and
18% of its scored rate, and higher human recall,
with a sevenfold cheaper flow pair. DIS MEDIUM scores the most pixels and has the
highest recall, but its all-pixel FPR is above Farneback's. The estimator time falls
less than the flow time because more surviving tracks mean more tracking and fitting
work; with keyframes (K=3) the flow share dominates and DIS FAST halves the per-frame
cost. These are two short synthetic segments with exact depth and poses, scored by
the semantic proxy, and DIS has not been tuned; they motivate a full replay. The C++
default stays Farneback; the test mapper uses DIS FAST.

## Keyframe cost optimizations (2026-10-12)

Exact changes to the keyframe work: symmetric CRF weights computed once per pair with
precomputed depth links, per-pixel fit state cleared with a counter, and temporal
snapshots taking ownership of their buffers instead of copying them. Profiled on the
segment with people with the test preset (DIS FAST, K=3, CRF), same harness and threads:

| Keyframe stage, ms | Before | After |
| --- | --- | --- |
| Reference setup | 19.3 | 12.5 |
| Backward tracking (4 passes) | 52.1 | 52.7 |
| Fits | 7.2 | 7.2 |
| CRF weights | 29.4 | 12.3 |
| CRF iterations and output | 10.9 | 11.5 |
| Snapshot | 3.7 | 0.0 |

Each input also costs about 13 ms for flow, gray/texture and structure tensor. The mean
per input fell from about 57 ms to 46 ms, and a keyframe from about 135 ms to 109 ms.
64-bit FNV-1a hashes of all 28 output matrices for every estimate were identical before and
after on that segment for the preset (200 inputs), and for Farneback with CRF, DIS with
one-pass support, and legacy scoring (60 inputs each), all with debug output enabled.
Tracking is now the largest keyframe cost.

### Tracking optimizations

A callgrind profile of 12 inputs attributed about 27% of all instructions to the
depth-layer neighborhood test in `samplePoint`, about 15% to Eigen's blocked
triangular-solve kernel running 2x2 and 3x3 covariance inversions, and 4% to the
ambiguity patch comparison. Three exact changes followed: a per-frame depth-layer state
map scanned in the original order, `choleskyInverse` for the small inversions, and a
shared integer-sampled reference patch. Solving the inversions column by column with
vector right-hand sides was faster too, but changed the last bits of a few velocities
and costs, so it was not used. Same segment, preset and threads:

| Keyframe stage, ms | Before | After |
| --- | --- | --- |
| Reference setup | 12.5 | 10.5 |
| Backward tracking (4 passes) | 52.7 | 29.6 |
| Fits | 7.2 | 7.0 |
| CRF weights + iterations | 23.8 | 22.3 |

A keyframe now costs about 83 ms including its own flow and texture, and the mean per
input fell from about 46 ms to 37 ms (36-39 ms over repeated runs). Output hashes of all
28 matrices were again identical for the preset, Farneback with CRF, DIS with support
and legacy scoring.

## Measured behavior (uHumans2 apartment, 2026-10-09)

Full `tesse_cd_apartment` replay through `optical_flow_dynamics_test.launch.yaml`
(GT poses, `play_rate:=0.5`, 1739 frames, about 348 five-frame windows), evaluated
with `evaluate_dynamics_bag.py` at its default per-mode threshold (legacy CDF > 0.99,
temporal evidence > 0.5). "Human" is the semantic foreground proxy described in the evaluation instructions, not a motion
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

