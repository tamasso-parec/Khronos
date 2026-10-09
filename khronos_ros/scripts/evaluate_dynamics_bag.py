#!/usr/bin/env python3
"""Evaluate first-frame dynamics scores against semantic masks from a ROS 2 bag.

Uses the last recorded cloud per reference timestamp. Reports semantic foreground
accuracy, abstention coverage, boundary/interior detection and failure evidence.
Semantic foreground is a proxy for motion; it does not establish actual movement.
"""

import argparse
import bisect
import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml


FAILURES = {
    1: "invalid_reference", 2: "low_texture", 4: "inconsistent_flow",
    8: "outside_image", 16: "invalid_depth", 32: "depth_discontinuity",
    64: "out_of_range", 128: "nonfinite_flow", 256: "numerical_failure", 512: "poor_model_fit",
    1024: "ambiguous_correspondence", 2048: "occluded",
}
FIELD_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
IMAGE_DTYPES = {
    "mono8": "u1", "8UC1": "u1", "mono16": "u2", "16UC1": "u2",
    "16SC1": "i2", "32SC1": "i4", "32UC1": "u4", "32FC1": "f4",
}
REGIONS = ("all", "foreground", "foreground_interior", "foreground_boundary",
           "background", "background_near_foreground", "background_far",
           "background_geometric_edge", "background_geometric_smooth")


def timestamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def read_bag(path, topic, expected_type):
    # ROS imports stay lazy so decoding and metric tests can also run outside ROS.
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise RuntimeError("ROS Python modules unavailable; source your ROS/workspace setup inside Docker") from error
    path = Path(path)
    if not (path / "metadata.yaml").is_file():
        raise ValueError(f"Expected a ROS 2 bag directory containing metadata.yaml: {path}")
    metadata = rosbag2_py.Info().read_metadata(str(path), "")
    reader_class = (rosbag2_py.SequentialCompressionReader if metadata.compression_format
                    else rosbag2_py.SequentialReader)
    reader = reader_class()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=metadata.storage_identifier),
                rosbag2_py.ConverterOptions("", ""))
    topics = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    if topic not in topics:
        raise ValueError(f"Topic {topic} missing from {path}. Available topics: {', '.join(sorted(topics))}")
    if topics[topic] != expected_type:
        raise ValueError(f"Topic {topic} has type {topics[topic]}, expected {expected_type}")
    message_type = get_message(expected_type)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    index = 0
    while reader.has_next():
        _, serialized, record_time = reader.read_next()
        yield index, deserialize_message(serialized, message_type), record_time
        index += 1


def decode_cloud(message):
    """Respect byte order, row padding and field offsets; retain organized pixel indexing."""
    if message.height < 1 or message.width < 1 or message.point_step < 1:
        raise ValueError("Empty or invalid point cloud")
    if message.row_step < message.width * message.point_step or len(message.data) < message.height * message.row_step:
        raise ValueError("Point cloud buffer/row_step is inconsistent with its dimensions")
    names, formats, offsets = [], [], []
    endian = ">" if message.is_bigendian else "<"
    for field in message.fields:
        if field.name in names or field.datatype not in FIELD_DTYPES or field.count < 1:
            raise ValueError(f"Invalid point cloud field: {field.name}")
        dtype = np.dtype(endian + FIELD_DTYPES[field.datatype])
        if field.offset < 0 or field.offset + field.count * dtype.itemsize > message.point_step:
            raise ValueError(f"Point cloud field exceeds point_step: {field.name}")
        names.append(field.name)
        formats.append(dtype if field.count == 1 else (dtype, (field.count,)))
        offsets.append(field.offset)
    required = ("x", "y", "z", "probability", "observations")
    if any(name not in names for name in required):
        raise ValueError(f"Cloud must contain fields {required}; found {names}")
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                      "itemsize": message.point_step})
    cloud = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                       strides=(message.row_step, message.point_step))
    if any(cloud[name].ndim != 2 for name in required):
        raise ValueError("Required cloud fields must be scalar")
    return cloud


def decode_labels(message):
    if message.encoding not in IMAGE_DTYPES:
        raise ValueError(f"Expected integer label image, received {message.encoding}; use the converted label topic")
    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype(endian + IMAGE_DTYPES[message.encoding])
    if (message.height < 1 or message.width < 1 or message.step < message.width * dtype.itemsize
            or len(message.data) < message.height * message.step):
        raise ValueError("Label image buffer/step is inconsistent with its dimensions")
    labels = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                        strides=(message.step, dtype.itemsize))
    if not np.isfinite(labels).all() or (np.issubdtype(dtype, np.floating) and not (labels == np.floor(labels)).all()):
        raise ValueError("Label image contains nonfinite or noninteger IDs")
    return labels.astype(np.int64)


def load_remap(path):
    with Path(path).open() as stream:
        entries = yaml.safe_load(stream)
    if not isinstance(entries, list) or not entries:
        raise ValueError("Label remap must be a list of {sub_id: ..., super_id: ...}")
    mapping = {}
    for entry in entries:
        source, target = int(entry["sub_id"]), int(entry["super_id"])
        if source in mapping and mapping[source] != target:
            raise ValueError(f"Conflicting remap for label {source}")
        mapping[source] = target
    return mapping


def remap_labels(labels, mapping):
    if mapping is None:
        return labels
    unique, inverse = np.unique(labels, return_inverse=True)
    mapped = np.array([mapping.get(int(label), -1) for label in unique], dtype=np.int64)
    return mapped[inverse].reshape(labels.shape)


def hydra_config(relative):
    source = Path(__file__).resolve().parents[3] / "hydra" / "config"
    if (source / relative).is_file():
        return source / relative
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("hydra")) / "config" / relative
    except (ImportError, LookupError) as error:
        raise ValueError("Cannot locate Hydra configs; provide --label-remap and --foreground-labels explicitly") from error


def neighborhoods(foreground, radius):
    if radius == 0:
        return foreground.copy(), foreground.copy()
    padded = np.pad(foreground, radius, constant_values=False)
    interior = np.ones_like(foreground)
    expanded = np.zeros_like(foreground)
    h, w = foreground.shape
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            view = padded[y:y + h, x:x + w]
            interior &= view
            expanded |= view
    return interior, expanded


def geometric_edges(cloud, threshold):
    """Reference pixels whose 4-neighbor world points jump by more than threshold or are invalid."""
    points = np.stack([cloud[axis] for axis in ("x", "y", "z")], axis=-1).astype(np.float64)
    edges = np.zeros(points.shape[:2], dtype=bool)
    h, w = edges.shape
    for dy, dx in ((0, 1), (1, 0)):
        with np.errstate(invalid="ignore"):
            jump = ~(np.linalg.norm(points[:h - dy, :w - dx] - points[dy:, dx:], axis=-1) <= threshold)
        edges[:h - dy, :w - dx] |= jump
        edges[dy:, dx:] |= jump
    return edges


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


class Metrics:
    def __init__(self, thresholds, bins, score_field="probability"):
        self.thresholds = thresholds
        self.score_field = score_field
        self.positive_hist = np.zeros(bins, dtype=np.int64)
        self.negative_hist = np.zeros(bins, dtype=np.int64)
        self.positive = self.negative = self.known_positive = self.known_negative = 0
        self.tp = np.zeros(len(thresholds), dtype=np.int64)
        self.fp = np.zeros(len(thresholds), dtype=np.int64)
        self.diagnostic_pixels = self.failed_pixels = self.alive_pixels = self.alive_denominator = 0
        self.failure_pixels = {name: 0 for name in FAILURES.values()}

    def add(self, cloud, foreground, eligible, known):
        positive = eligible & foreground
        negative = eligible & ~foreground
        kp, kn = positive & known, negative & known
        self.positive += int(positive.sum())
        self.negative += int(negative.sum())
        self.known_positive += int(kp.sum())
        self.known_negative += int(kn.sum())
        score = cloud[self.score_field]
        self.positive_hist += np.histogram(score[kp], bins=len(self.positive_hist), range=(0, 1))[0]
        self.negative_hist += np.histogram(score[kn], bins=len(self.negative_hist), range=(0, 1))[0]
        for i, threshold in enumerate(self.thresholds):
            prediction = known & (score > threshold)
            self.tp[i] += int((prediction & positive).sum())
            self.fp[i] += int((prediction & negative).sum())
        if "failure_flags" in cloud.dtype.names:
            flags = cloud["failure_flags"]
            self.diagnostic_pixels += int(eligible.sum())
            self.failed_pixels += int((eligible & (flags != 0)).sum())
            for bit, name in FAILURES.items():
                self.failure_pixels[name] += int((eligible & ((flags & bit) != 0)).sum())
        if "track_alive" in cloud.dtype.names:
            self.alive_denominator += int(eligible.sum())
            self.alive_pixels += int((eligible & (cloud["track_alive"] != 0)).sum())

    def summary(self):
        total = self.positive + self.negative
        return {
            "eligible_pixels": total, "foreground_pixels": self.positive,
            "background_pixels": self.negative,
            "coverage": ratio(self.known_positive + self.known_negative, total),
            "foreground_coverage": ratio(self.known_positive, self.positive),
            "background_coverage": ratio(self.known_negative, self.negative),
            "any_failure_fraction": ratio(self.failed_pixels, self.diagnostic_pixels),
            "failure_fractions": {name: ratio(count, self.diagnostic_pixels)
                                  for name, count in self.failure_pixels.items()},
            "track_alive_fraction": ratio(self.alive_pixels, self.alive_denominator),
        }

    def merge(self, other):
        self.positive_hist += other.positive_hist
        self.negative_hist += other.negative_hist
        self.tp += other.tp
        self.fp += other.fp
        for name in ("positive", "negative", "known_positive", "known_negative",
                     "diagnostic_pixels", "failed_pixels", "alive_pixels", "alive_denominator"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in self.failure_pixels:
            self.failure_pixels[name] += other.failure_pixels[name]

    def threshold_rows(self):
        for i, threshold in enumerate(self.thresholds):
            tp, fp = int(self.tp[i]), int(self.fp[i])
            fn = self.positive - tp  # Unknown foreground pixels count as missed detections.
            yield {
                "threshold": threshold, "tp": tp, "fp": fp, "fn_all": fn,
                "precision": ratio(tp, tp + fp), "recall_all": ratio(tp, self.positive),
                "recall_scored": ratio(tp, self.known_positive),
                "f1_all": ratio(2 * tp, 2 * tp + fp + fn),
                "iou_all": ratio(tp, tp + fp + fn),
                "background_fpr_all": ratio(fp, self.negative),
                "background_fpr_scored": ratio(fp, self.known_negative),
                "ideal_static_exceedance": 1.0 - threshold if self.score_field == "probability" else None,
            }

    def curve(self):
        tp = np.cumsum(self.positive_hist[::-1])
        fp = np.cumsum(self.negative_hist[::-1])
        for i in range(len(tp)):
            yield {
                "score_bin_lower_bound": (len(tp) - 1 - i) / len(tp),
                "precision": ratio(int(tp[i]), int(tp[i] + fp[i])),
                "recall_scored": ratio(int(tp[i]), self.known_positive),
                "recall_all": ratio(int(tp[i]), self.positive),
            }

    def binned_average_precision(self):
        if not self.known_positive:
            return None
        tp = np.cumsum(self.positive_hist[::-1])
        fp = np.cumsum(self.negative_hist[::-1])
        precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=(tp + fp) > 0)
        return float(np.sum(precision * self.positive_hist[::-1]) / self.known_positive)


def evaluate_window(cloud, labels, args):
    if cloud.shape != labels.shape:
        raise ValueError(f"Cloud dimensions {cloud.shape} differ from GT labels {labels.shape}; preserve organized indexing")
    foreground = np.isin(labels, args.foreground_labels)
    eligible = (labels >= 0) & ~np.isin(labels, args.ignore_labels)
    for axis in ("x", "y", "z"):
        eligible &= np.isfinite(cloud[axis])
    score_field = getattr(args, "score_field", "probability")
    score = cloud[score_field]
    finite = np.isfinite(score)
    if np.any(eligible & finite & ((score < 0) | (score > 1))) or np.any(cloud["observations"] < 0):
        raise ValueError("Cloud contains invalid probability or negative observation count")
    known = eligible & finite & (cloud["observations"] >= args.min_observations)
    if score_field != "probability" and "validity" in cloud.dtype.names:
        known &= cloud["validity"] >= 3
    if score_field == "regularized_evidence" and "spatial_support" in cloud.dtype.names:
        # Inferred support is evaluated as a segmentation output, not counted as a measurement.
        known |= eligible & finite & (cloud["spatial_support"] == 2)
    interior, expanded = neighborhoods(foreground, args.boundary_radius)
    regions = {
        "all": eligible, "foreground": eligible & foreground,
        "foreground_interior": eligible & interior,
        "foreground_boundary": eligible & foreground & ~interior,
        "background": eligible & ~foreground,
        "background_near_foreground": eligible & ~foreground & expanded,
        "background_far": eligible & ~expanded,
    }
    # Static-geometry boundaries away from the foreground band isolate static boundary responses.
    _, edges = neighborhoods(geometric_edges(cloud, getattr(args, "geometric_edge_threshold", 0.1)),
                             args.boundary_radius)
    regions["background_geometric_edge"] = regions["background_far"] & edges
    regions["background_geometric_smooth"] = regions["background_far"] & ~edges
    return foreground, known, regions


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        return
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def evaluate(args):
    if args.foreground_labels is None:
        with hydra_config(f"label_spaces/{args.dataset}_label_space.yaml").open() as stream:
            args.foreground_labels = yaml.safe_load(stream)["dynamic_labels"]
    if not args.foreground_labels or set(args.foreground_labels) & set(args.ignore_labels):
        raise ValueError("Foreground labels must be nonempty and must not overlap ignored labels")
    remap_path = args.label_remap or (None if args.labels_are_remapped else
                                    hydra_config(f"label_remaps/{args.dataset}.yaml"))
    mapping = None if args.labels_are_remapped else load_remap(remap_path)
    last = {}
    snapshots = 0
    for index, message, _ in read_bag(args.results_bag, args.cloud_topic, "sensor_msgs/msg/PointCloud2"):
        if index == 0:
            cloud = decode_cloud(message)
            temporal = "scoring_mode" in cloud.dtype.names and int(cloud["scoring_mode"][0, 0]) == 1
            if args.score_field is None:
                args.score_field = "motion_evidence" if temporal else "probability"
            if args.score_field not in cloud.dtype.names:
                raise ValueError(f"Requested score field {args.score_field} absent from cloud")
            if args.thresholds is None:
                args.thresholds = [0.25, 0.5, 0.75] if args.score_field != "probability" else [0.95, 0.99, 0.999]
        last[timestamp_ns(message)] = index
        snapshots += 1
    if not last:
        raise ValueError("No dynamics clouds recorded; start ros2 bag record before replay")
    references = sorted(last)
    tolerance = int(args.timestamp_tolerance_ms * 1.e6)
    output = args.output or args.results_bag.with_name(args.results_bag.name + "_evaluation")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {name: Metrics(args.thresholds, args.score_bins, args.score_field) for name in REGIONS}
    windows = []
    print(f"Indexed {snapshots} snapshots in {len(references)} reference windows.")
    # Cache only relevant GT frames on disk; a long bag need not fit in RAM.
    with tempfile.TemporaryDirectory(prefix="gt_cache_", dir=output) as cache:
        matches = {}
        for _, image, _ in read_bag(args.ground_truth_bag, args.labels_topic, "sensor_msgs/msg/Image"):
            stamp = timestamp_ns(image)
            begin = bisect.bisect_left(references, stamp - tolerance)
            end = bisect.bisect_right(references, stamp + tolerance)
            candidates = [ref for ref in references[begin:end]
                          if ref not in matches or abs(stamp - ref) < abs(matches[ref] - ref)]
            if candidates:
                labels = remap_labels(decode_labels(image), mapping)
                for ref in candidates:
                    np.savez_compressed(Path(cache) / f"{ref}.npz", labels=labels)
                    matches[ref] = stamp
        if not matches:
            raise ValueError("No GT header timestamps match reference clouds; check labels topic and simulation clocks")
        for index, message, record_time in read_bag(args.results_bag, args.cloud_topic, "sensor_msgs/msg/PointCloud2"):
            ref = timestamp_ns(message)
            if last[ref] != index or ref not in matches:
                continue
            cloud = decode_cloud(message)
            with np.load(Path(cache) / f"{ref}.npz", allow_pickle=False) as data:
                labels = data["labels"]
            foreground, known, regions = evaluate_window(cloud, labels, args)
            current = {}
            for name, mask in regions.items():
                metric = Metrics(args.thresholds, args.score_bins, args.score_field)
                metric.add(cloud, foreground, mask, known)
                aggregate[name].merge(metric)
                current[name] = metric
            for row in current["all"].threshold_rows():
                i = args.thresholds.index(row["threshold"])
                summary = current["all"].summary()
                windows.append({
                    "reference_timestamp_ns": ref, "gt_timestamp_ns": matches[ref],
                    "gt_offset_ns": matches[ref] - ref, "record_timestamp_ns": record_time,
                    "eligible_pixels": summary["eligible_pixels"],
                    "foreground_pixels": summary["foreground_pixels"],
                    "coverage": summary["coverage"],
                    "foreground_coverage": summary["foreground_coverage"],
                    "background_coverage": summary["background_coverage"],
                    **row,
                    "interior_recall_all": ratio(int(current["foreground_interior"].tp[i]),
                                                 current["foreground_interior"].positive),
                    "boundary_recall_all": ratio(int(current["foreground_boundary"].tp[i]),
                                                 current["foreground_boundary"].positive),
                })
    all_metrics = aggregate["all"]
    if not all_metrics.positive + all_metrics.negative:
        raise ValueError("No eligible pixels; check GT remapping and reference geometry")
    unmatched = [ref for ref in references if ref not in matches]
    summary = {
        "ground_truth_definition": "semantic foreground proxy, not actual motion annotations",
        "score_field": args.score_field,
        "score_semantics": "legacy approximate static CDF" if args.score_field == "probability" else "uncalibrated coherent-motion evidence",
        "foreground_labels": args.foreground_labels, "ignore_labels": args.ignore_labels,
        "label_remap": None if mapping is None else str(remap_path),
        "snapshot_selection": "last recorded cloud per reference header timestamp",
        "snapshots": snapshots, "reference_windows": len(references),
        "matched_windows": len(matches), "unmatched_windows": len(unmatched),
        "unmatched_reference_timestamps_first_20": unmatched[:20],
        "min_observations": args.min_observations, "boundary_radius_pixels": args.boundary_radius,
        "geometric_edge_threshold_m": args.geometric_edge_threshold,
        "score_bins": args.score_bins,
        "binned_average_precision_scored": all_metrics.binned_average_precision(),
        "regions": {name: metric.summary() for name, metric in aggregate.items()},
        "threshold_metrics": list(all_metrics.threshold_rows()),
    }
    write_csv(output / "windows.csv", windows)
    write_csv(output / "threshold_metrics.csv",
              ({"region": name, **row} for name, metric in aggregate.items() for row in metric.threshold_rows()))
    write_csv(output / "precision_recall.csv", all_metrics.curve())
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Evaluated {len(matches)}/{len(references)} windows. Reports: {output}")
    if unmatched:
        print(f"Skipped {len(unmatched)} windows without matching GT; see summary.json.")
    if not all_metrics.positive:
        print("No foreground pixels in the matched windows; foreground recall/AP are undefined.")
    print(f"Coverage: {all_metrics.summary()['coverage']:.1%}; "
          f"foreground coverage: {all_metrics.summary()['foreground_coverage']}")
    for row in all_metrics.threshold_rows():
        print(f"threshold={row['threshold']:g}: precision={row['precision']}, "
              f"recall_all={row['recall_all']}, background_fpr_scored={row['background_fpr_scored']}")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-bag", type=Path, required=True, help="Recorded dynamics bag directory")
    parser.add_argument("--ground-truth-bag", type=Path, required=True, help="Original dataset bag containing GT label images")
    parser.add_argument("--output", type=Path, help="Empty report directory; default: RESULTS_BAG_evaluation")
    parser.add_argument("--cloud-topic", default="/khronos_node/visualization/dynamics_cloud")
    parser.add_argument("--labels-topic", default="/tesse/seg_cam/converted/image_raw")
    parser.add_argument("--dataset", default="tesse_cd_apartment", help="Dataset whose Hydra label remap should be used")
    parser.add_argument("--label-remap", type=Path, help="Explicit Hydra sub_id/super_id YAML remap")
    parser.add_argument("--labels-are-remapped", action="store_true", help="GT topic already contains mapped semantic IDs")
    parser.add_argument("--foreground-labels", type=int, nargs="+", help="Mapped foreground IDs; default: dataset's dynamic_labels")
    parser.add_argument("--ignore-labels", type=int, nargs="*", default=[0], help="Mapped labels excluded from evaluation")
    parser.add_argument("--score-field", choices=["probability", "motion_evidence", "regularized_evidence"],
                        help="Default: auto-detect mode; choose regularized_evidence to evaluate spatial inference")
    parser.add_argument("--thresholds", type=float, nargs="+", help="Default: .95/.99/.999 legacy; .25/.5/.75 temporal")
    parser.add_argument("--min-observations", type=int, default=2, help="Match the estimator's min_observations")
    parser.add_argument("--boundary-radius", type=int, default=3, help="Foreground interior erosion radius in pixels")
    parser.add_argument("--geometric-edge-threshold", type=float, default=0.1,
                        help="Neighbor world-point jump [m] marking static geometric boundaries")
    parser.add_argument("--timestamp-tolerance-ms", type=float, default=0.0, help="Maximum GT header offset; exact matching by default")
    parser.add_argument("--score-bins", type=int, default=10000, help="Histogram bins for approximate PR/AP; fixed-threshold metrics are exact")
    args = parser.parse_args(argv)
    if (args.min_observations < 1 or args.boundary_radius < 0 or args.score_bins < 2
            or not np.isfinite(args.timestamp_tolerance_ms) or args.timestamp_tolerance_ms < 0
            or not np.isfinite(args.geometric_edge_threshold) or args.geometric_edge_threshold <= 0):
        parser.error("Invalid observation count, boundary radius, score bins, timestamp tolerance or edge threshold")
    if args.thresholds and any(not np.isfinite(value) or not 0 <= value <= 1 for value in args.thresholds):
        parser.error("Thresholds must be finite values in [0, 1]")
    if args.labels_are_remapped and args.label_remap:
        parser.error("Use either --labels-are-remapped or --label-remap")
    if args.foreground_labels and set(args.foreground_labels) & set(args.ignore_labels):
        parser.error("Foreground labels and ignored labels must not overlap")
    if args.thresholds:
        args.thresholds = sorted(set(args.thresholds))
    return args


def main(argv=None):
    try:
        evaluate(parse_args(argv))
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
