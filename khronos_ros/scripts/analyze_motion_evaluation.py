#!/usr/bin/env python3
"""Compare the motion detection of one or more runs written by the MotionEvaluationLogger.

Ground truth are pixels of dynamic labels (e.g. humans) within range. Reports per run:
- pixel precision / recall / F1 of the dynamic image,
- cluster precision (clusters with mostly ground truth pixels) and false positive clusters,
- per "episode" (consecutive frames with a visible dynamic object) whether and how quickly a true
  positive cluster was detected,
- contributions of the object priors: seed and suppression pixels and how many were ground truth.

Usage:
  analyze_motion_evaluation.py RUN_DIR_OR_CSV [RUN_DIR_OR_CSV ...] [--min-gt-pixels 200]
"""

import argparse
import csv
import os


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "motion_evaluation.csv")
    with open(path) as f:
        return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]


def ratio(a, b):
    return a / b if b else float("nan")


def episodes(rows, min_gt_pixels, max_gap_s):
    """Split frames with a visible dynamic object into episodes of consecutive frames."""
    result, current, last_t = [], [], None
    for r in rows:
        if r["gt_pixels"] < min_gt_pixels:
            continue
        t = r["timestamp_ns"] * 1e-9
        if current and t - last_t > max_gap_s:
            result.append(current)
            current = []
        current.append(r)
        last_t = t
    if current:
        result.append(current)
    return result


def analyze(name, rows, args):
    s = lambda key: sum(r[key] for r in rows)
    precision = ratio(s("tp_pixels"), s("detected_pixels"))
    recall = ratio(s("tp_pixels"), s("gt_pixels"))
    f1 = ratio(2 * precision * recall, precision + recall)
    fp_clusters = s("clusters") - s("tp_clusters")

    eps = [e for e in episodes(rows, args.min_gt_pixels, args.max_gap) if len(e) >= args.min_episode_frames]
    detected, latencies = 0, []
    for e in eps:
        hit = next((r for r in e if r["tp_clusters"] > 0), None)
        if hit:
            detected += 1
            latencies.append((hit["timestamp_ns"] - e[0]["timestamp_ns"]) * 1e-9)
    visible = [r for r in rows if r["gt_pixels"] >= args.min_gt_pixels]

    return {
        "run": name,
        "frames": len(rows),
        "pix P": precision,
        "pix R": recall,
        "pix F1": f1,
        "clusters": s("clusters"),
        "clu P": ratio(s("tp_clusters"), s("clusters")),
        "FP clu/frame": ratio(fp_clusters, len(rows)),
        "vis frames det": ratio(sum(r["tp_clusters"] > 0 for r in visible), len(visible)),
        "episodes": len(eps),
        "ep detected": ratio(detected, len(eps)),
        "latency [s]": ratio(sum(latencies), len(latencies)),
        "moving prior frames": sum(r["moving_priors"] > 0 for r in rows),
        "seed px": s("prior_seed_pixels"),
        "seed P": ratio(s("prior_seed_gt_pixels"), s("prior_seed_pixels")),
        "suppr px": s("suppress_pixels"),
        "suppr on GT": s("suppress_gt_pixels"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", help="run output directories or motion_evaluation.csv files")
    parser.add_argument("--min-gt-pixels", type=int, default=200,
                        help="frames with fewer ground truth pixels count as no visible dynamic object")
    parser.add_argument("--max-gap", type=float, default=0.5,
                        help="frames further apart in seconds start a new episode")
    parser.add_argument("--min-episode-frames", type=int, default=3,
                        help="ignore episodes shorter than this")
    args = parser.parse_args()

    results = []
    for path in args.runs:
        name = os.path.basename(os.path.normpath(path if os.path.isdir(path) else os.path.dirname(path)))
        results.append(analyze(name, load(path), args))

    keys = list(results[0].keys())
    width = max(len(k) for k in keys) + 2
    print("".ljust(width) + "".join(r["run"].rjust(16) for r in results))
    for k in keys[1:]:
        cells = []
        for r in results:
            v = r[k]
            cells.append((f"{v:.3f}" if isinstance(v, float) else str(v)).rjust(16))
        print(k.ljust(width) + "".join(cells))
    print(
        "\nGround truth: pixels with a dynamic label (humans) within range, including humans standing "
        "still. 'latency' is the time from a dynamic object becoming visible to the first true "
        "positive cluster."
    )


if __name__ == "__main__":
    main()
