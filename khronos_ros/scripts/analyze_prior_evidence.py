#!/usr/bin/env python3
"""Summarize and plot the per-frame evidence written by the PriorEvidenceLogger.

Prints one row per scene graph object prior and, if matplotlib is available, saves a plot of the
absent fraction over time next to the CSV. Ground truth change times can be overlaid as vertical
lines. All times are simulation time in seconds, which is also what gt_changes.csv uses. Only the
standard library is required for the text summary.

Usage:
  analyze_prior_evidence.py OUTPUT_DIR/prior_evidence.csv \
      [--gt-changes /data/tesse_cd_apartment/gt_changes.csv]
"""

import argparse
import csv
import os
from collections import defaultdict


def read_evidence(path):
    frames = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            frames[row["node_id"]].append(
                {
                    "t": int(row["timestamp_ns"]),
                    "label": int(row["semantic_label"]),
                    "present": int(row["present"]),
                    "absent": int(row["absent"]),
                    "occluded": int(row["occluded"]),
                    "unobserved": int(row["unobserved"]),
                }
            )
    return frames


def read_gt_changes(path):
    """Return (object, disappeared_at_ns) for all objects that disappear before the end."""
    changes = []
    with open(path) as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    end = max(int(r["DisappearedAt"]) for r in rows)
    for r in rows:
        if int(r["DisappearedAt"]) < end:
            changes.append((r["ObjectSymbol"], int(r["DisappearedAt"])))
        if int(r["AppearedAt"]) > 0:
            changes.append((r["ObjectSymbol"] + " appears", int(r["AppearedAt"])))
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("evidence", help="prior_evidence.csv written by the PriorEvidenceLogger")
    parser.add_argument("--gt-changes", help="gt_changes.csv of the dataset")
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=50,
        help="frames with fewer present + absent pixels are ignored (default: 50)",
    )
    parser.add_argument(
        "--absent-threshold",
        type=float,
        default=0.5,
        help="absent fraction above which a frame counts as 'absent' (default: 0.5)",
    )
    args = parser.parse_args()

    frames = read_evidence(args.evidence)
    if not frames:
        print("No prior observations logged. Did objects leave the active window and get revisited?")
        return

    def sec(t):
        return t * 1e-9

    # Per-object summary over informative frames.
    rows = []
    for node_id, obs in frames.items():
        informative = [f for f in obs if f["present"] + f["absent"] >= args.min_pixels]
        absent_frames = [
            f for f in informative if f["absent"] / (f["present"] + f["absent"]) > args.absent_threshold
        ]
        total_present = sum(f["present"] for f in informative)
        total_absent = sum(f["absent"] for f in informative)
        rows.append(
            {
                "node_id": node_id,
                "label": obs[0]["label"],
                "rendered": len(obs),
                "informative": len(informative),
                "absent_frames": len(absent_frames),
                "absent_fraction": total_absent / (total_present + total_absent)
                if informative
                else float("nan"),
                "first_seen": sec(informative[0]["t"]) if informative else float("nan"),
                "first_absent": sec(absent_frames[0]["t"]) if absent_frames else float("nan"),
            }
        )
    rows.sort(key=lambda r: (-r["absent_frames"], r["node_id"]))

    header = (
        f"{'object':>10} {'label':>5} {'frames':>6} {'inform.':>7} {'absent':>6} "
        f"{'abs.frac':>8} {'first [s]':>9} {'1st abs [s]':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['node_id']:>10} {r['label']:>5} {r['rendered']:>6} {r['informative']:>7} "
            f"{r['absent_frames']:>6} {r['absent_fraction']:>8.2f} {r['first_seen']:>9.1f} "
            f"{r['first_absent']:>11.1f}"
        )
    print(
        f"\n{len(rows)} objects rendered, {sum(r['informative'] > 0 for r in rows)} with informative "
        f"frames, {sum(r['absent_frames'] > 0 for r in rows)} with at least one absent frame."
    )

    changes = read_gt_changes(args.gt_changes) if args.gt_changes else []
    if changes:
        print("\nGround truth changes (object ids are from the GT scene graph, not this run):")
        for name, t in sorted(changes, key=lambda c: c[1]):
            print(f"  {t * 1e-9:7.1f} s  {name}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for r in rows:
        informative = [
            f for f in frames[r["node_id"]] if f["present"] + f["absent"] >= args.min_pixels
        ]
        if not informative:
            continue
        times = [sec(f["t"]) for f in informative]
        fractions = [f["absent"] / (f["present"] + f["absent"]) for f in informative]
        if r["absent_frames"] == 0:
            ax.plot(times, fractions, ".", markersize=2, color="0.7")
            continue
        # Break the line where the object was not observed for more than a second.
        segments = [[0]]
        for i in range(1, len(times)):
            if times[i] - times[i - 1] > 1.0:
                segments.append([])
            segments[-1].append(i)
        color = None
        for segment in segments:
            (line,) = ax.plot(
                [times[i] for i in segment],
                [fractions[i] for i in segment],
                ".-",
                markersize=3,
                color=color,
                label=r["node_id"] if color is None else None,
            )
            color = line.get_color()
    for name, t in changes:
        ax.axvline(t * 1e-9, color="k", linestyle="--", lw=0.8)
        ax.text(t * 1e-9, 0.98, name, rotation=90, fontsize=7, va="top", ha="right")
    ax.axhline(args.absent_threshold, color="r", lw=0.5)
    ax.set_xlabel("simulation time [s]")
    ax.set_ylabel("absent fraction")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Scene graph object priors: absent / (present + absent) per frame")
    if any(r["absent_frames"] > 0 for r in rows):
        ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    out = os.path.splitext(args.evidence)[0] + ".png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot to '{out}'.")


if __name__ == "__main__":
    main()
