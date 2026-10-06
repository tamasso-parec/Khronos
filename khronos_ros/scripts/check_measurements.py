#!/usr/bin/env python3
"""Validate a recorded dataset and exercise geometry/track access for batch work."""

import argparse
import json

import numpy as np

from khronos_measurements import Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Recorded dataset directory")
    args = parser.parse_args()

    try:
        dataset = Dataset(args.dataset)
        report = dataset.validate()
        track_ids = set()
        dynamic_pixels = valid_dynamic_points = associations = missing_clusters = 0
        first_stamp = last_stamp = None
        for frame in dataset:
            if first_stamp is None:
                first_stamp = frame.timestamp_ns
            last_stamp = frame.timestamp_ns
            xyz = frame.vertices
            motion_guess = frame.dynamic_labels != 0
            valid = np.isfinite(xyz).all(axis=2) & np.isfinite(frame.depth) & (frame.depth > 0)
            dynamic_pixels += int(np.count_nonzero(motion_guess))
            valid_dynamic_points += int(np.count_nonzero(motion_guess & valid))
            for track in frame.tracks:
                track_ids.add(track["id"])
                cluster_id = track["dynamic_cluster_id"]
                if cluster_id != -1:
                    points = xyz[frame.dynamic_labels == cluster_id]
                    associations += 1
                    if len(points) == 0:
                        missing_clusters += 1

        report.update(
            duration_s=0.0 if first_stamp is None else (last_stamp - first_stamp) / 1e9,
            unique_tracks=len(track_ids),
            dynamic_pixels=dynamic_pixels,
            valid_dynamic_points=valid_dynamic_points,
            dynamic_track_observations=associations,
            associations_without_cluster=missing_clusters,
        )
        print(json.dumps(report, indent=2))
        if (not report["frames"] or not report["closed"]
                or report["missing_sequence_intervals"] or missing_clusters):
            parser.exit(1, "CHECK FAILED: empty/incomplete recording, sequence gaps, or missing clusters.\n")
        print("PASS: recorded frames and dynamic-track associations are readable and consistent.")
        print("This does not verify detector accuracy or missing frames after the last recorded frame.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"CHECK FAILED: {exc}\n")


if __name__ == "__main__":
    main()
