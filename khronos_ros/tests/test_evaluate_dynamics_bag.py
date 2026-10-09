"""Run with: python3 -m unittest discover -s khronos_ros/tests -v (ROS sourced for bag tests)."""

from array import array
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_dynamics_bag.py"
SPEC = importlib.util.spec_from_file_location("evaluate_dynamics_bag", SCRIPT)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def cloud_message(width=5, height=5, bigendian=False, padding=0):
    fields = [SimpleNamespace(name=name, offset=i * 4, datatype=datatype, count=1)
              for i, (name, datatype) in enumerate([
                  ("x", 7), ("y", 7), ("z", 7), ("probability", 7),
                  ("observations", 5), ("failure_flags", 6), ("track_alive", 6)])]
    rows = []
    endian = ">" if bigendian else "<"
    for _ in range(height):
        row = b"".join(struct.pack(endian + "ffffiII", 100, 100, 2, 0.1, 2, 0, 1)
                       for _ in range(width))
        rows.append(row + bytes(padding))
    return SimpleNamespace(width=width, height=height, point_step=28,
                           row_step=width * 28 + padding, fields=fields,
                           is_bigendian=bigendian, data=bytearray(b"".join(rows)))


def label_message(labels, bigendian=False, padding=0):
    dtype = np.dtype(">u2" if bigendian else "<u2")
    rows = [row.astype(dtype).tobytes() + bytes(padding) for row in labels]
    return SimpleNamespace(width=labels.shape[1], height=labels.shape[0],
                           encoding="16UC1", is_bigendian=bigendian,
                           step=labels.shape[1] * 2 + padding, data=b"".join(rows))


def evaluation_args():
    return SimpleNamespace(foreground_labels=[20], ignore_labels=[0], min_observations=2,
                           boundary_radius=1, thresholds=[0.99], score_bins=100)


class EvaluationTests(unittest.TestCase):
    def test_cloud_and_labels_decode_endianness_and_padding(self):
        for endian in (False, True):
            cloud = evaluator.decode_cloud(cloud_message(bigendian=endian, padding=12))
            self.assertEqual(cloud.shape, (5, 5))
            self.assertAlmostEqual(float(cloud["probability"][4, 4]), 0.1)
            self.assertEqual(int(cloud["observations"][3, 2]), 2)
            labels = np.array([[10, 464], [464, 20]])
            np.testing.assert_array_equal(evaluator.decode_labels(label_message(labels, endian, 4)), labels)

    def test_raw_ids_are_remapped_and_unlisted_ids_ignored(self):
        labels = np.array([[464, 20, 999]])
        mapped = evaluator.remap_labels(labels, {464: 20, 20: 5})
        np.testing.assert_array_equal(mapped, [[20, 5, -1]])

    def test_unknown_pixels_reduce_full_recall_and_coverage(self):
        message = cloud_message()
        cloud = evaluator.decode_cloud(message)
        labels = np.full((5, 5), 3)
        labels[1:4, 1:4] = 20
        cloud["probability"][1, 1:3] = 1.0  # Two true detections.
        cloud["probability"][0, 0] = 1.0    # One false detection.
        cloud["probability"][2, 2] = np.nan
        cloud["observations"][2, 2] = 1
        cloud["failure_flags"][2, 2] = 2
        cloud["track_alive"][2, 2] = 0
        args = evaluation_args()
        foreground, known, regions = evaluator.evaluate_window(cloud, labels, args)
        metrics = evaluator.Metrics(args.thresholds, args.score_bins)
        metrics.add(cloud, foreground, regions["all"], known)
        row = next(metrics.threshold_rows())
        self.assertEqual((row["tp"], row["fp"], row["fn_all"]), (2, 1, 7))
        self.assertAlmostEqual(row["precision"], 2 / 3)
        self.assertAlmostEqual(row["recall_all"], 2 / 9)
        self.assertAlmostEqual(row["recall_scored"], 2 / 8)
        self.assertAlmostEqual(row["background_fpr_scored"], 1 / 16)
        self.assertAlmostEqual(metrics.summary()["foreground_coverage"], 8 / 9)
        self.assertEqual(int(regions["foreground_interior"].sum()), 1)
        self.assertEqual(int((known & regions["foreground_interior"]).sum()), 0)
        self.assertEqual(int(regions["foreground_boundary"].sum()), 8)
        self.assertAlmostEqual(metrics.summary()["failure_fractions"]["low_texture"], 1 / 25)

    def test_histogram_ap_handles_ties_and_no_positives(self):
        metrics = evaluator.Metrics([0.5], 10)
        metrics.positive_hist[9] = 2
        metrics.negative_hist[9] = 1
        metrics.positive_hist[1] = 1
        metrics.known_positive = 3
        self.assertAlmostEqual(metrics.binned_average_precision(), (2 * (2 / 3) + 3 / 4) / 3)
        self.assertIsNone(evaluator.Metrics([0.5], 10).binned_average_precision())

    def test_invalid_geometry_and_labels_excluded_and_shape_mismatch_fails(self):
        cloud = evaluator.decode_cloud(cloud_message())
        labels = np.full((5, 5), 3)
        labels[0, 0] = -1
        cloud["x"][1, 1] = np.nan
        _, _, regions = evaluator.evaluate_window(cloud, labels, evaluation_args())
        self.assertEqual(int(regions["all"].sum()), 23)
        with self.assertRaises(ValueError):
            evaluator.evaluate_window(cloud, labels[:2], evaluation_args())

    def test_static_geometric_edges_split_far_background(self):
        cloud = evaluator.decode_cloud(cloud_message(width=9, height=9))
        cloud["z"][:, 5:] = 4  # Depth step between columns 4 and 5.
        labels = np.full((9, 9), 3)
        labels[0, 0] = 20
        args = evaluation_args()
        args.boundary_radius = 1
        _, _, regions = evaluator.evaluate_window(cloud, labels, args)
        edge, smooth = regions["background_geometric_edge"], regions["background_geometric_smooth"]
        self.assertFalse((edge & smooth).any())
        np.testing.assert_array_equal(edge | smooth, regions["background_far"])
        self.assertTrue(edge[4:, 3:7].all())
        self.assertFalse(edge[4:, :3].any() or edge[:, 7:].any())
        self.assertFalse(edge[:2, :2].any())  # Near-foreground band is excluded.

    def test_new_evidence_validity_and_spatial_support_have_distinct_coverage(self):
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("probability", "f4"),
                 ("observations", "i4"), ("motion_evidence", "f4"),
                 ("regularized_evidence", "f4"), ("validity", "u4"), ("spatial_support", "u4")]
        cloud = np.zeros((5, 5), dtype=dtype)
        cloud["z"][:] = 2
        cloud["observations"][:] = 3
        cloud["validity"][:] = 3
        cloud["motion_evidence"][2, 2] = np.nan
        cloud["regularized_evidence"][2, 2] = 0.9
        cloud["validity"][2, 2] = 0
        cloud["observations"][2, 2] = 0
        cloud["spatial_support"][2, 2] = 2
        labels = np.full((5, 5), 3)
        args = evaluation_args()
        args.score_field = "motion_evidence"
        _, raw, _ = evaluator.evaluate_window(cloud, labels, args)
        self.assertFalse(raw[2, 2])
        args.score_field = "regularized_evidence"
        _, spatial, _ = evaluator.evaluate_window(cloud, labels, args)
        self.assertTrue(spatial[2, 2])
        self.assertEqual(cloud["observations"][2, 2], 0)
        metrics = evaluator.Metrics([0.5], 100, args.score_field)
        self.assertIsNone(next(metrics.threshold_rows())["ideal_static_exceedance"])

    def test_source_dataset_configuration_is_discoverable(self):
        self.assertTrue(evaluator.hydra_config("label_remaps/tesse_cd_apartment.yaml").is_file())
        self.assertTrue(evaluator.hydra_config("label_spaces/tesse_cd_office_label_space.yaml").is_file())


class RealRosbagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import rosbag2_py
            from rclpy.serialization import serialize_message
            from sensor_msgs.msg import Image, PointCloud2, PointField
        except ImportError:
            raise unittest.SkipTest("Source ROS to run the generated-bag integration test")
        cls.rosbag = rosbag2_py
        cls.serialize = staticmethod(serialize_message)
        cls.Image, cls.PointCloud2, cls.PointField = Image, PointCloud2, PointField

    def write_bag(self, path, topic, message_type, records):
        writer = self.rosbag.SequentialWriter()
        writer.open(self.rosbag.StorageOptions(uri=str(path), storage_id="sqlite3"),
                    self.rosbag.ConverterOptions("", ""))
        try:  # Jazzy and newer require a topic id.
            metadata = self.rosbag.TopicMetadata(id=0, name=topic, type=message_type,
                                                 serialization_format="cdr")
        except TypeError:
            metadata = self.rosbag.TopicMetadata(name=topic, type=message_type,
                                                 serialization_format="cdr")
        writer.create_topic(metadata)
        for message, stamp in records:
            writer.write(topic, self.serialize(message), stamp)
        del writer

    def real_cloud(self, reference, final):
        simple = cloud_message()
        cloud = evaluator.decode_cloud(simple)
        if final:
            cloud["probability"][1, 1:3] = 1.0
            cloud["probability"][0, 0] = 1.0
            cloud["probability"][2, 2] = np.nan
            cloud["observations"][2, 2] = 1
            cloud["failure_flags"][2, 2] = 2
        else:
            cloud["probability"][:] = np.nan
            cloud["observations"][:] = 0
        message = self.PointCloud2()
        message.header.stamp.sec = reference
        message.height, message.width = simple.height, simple.width
        message.point_step, message.row_step = simple.point_step, simple.row_step
        message.fields = [self.PointField(name=f.name, offset=f.offset, datatype=f.datatype, count=f.count)
                          for f in simple.fields]
        message.data = array("B", simple.data)
        return message

    def test_real_bag_last_snapshot_gt_remap_and_unmatched_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result_bag, gt_bag = base / "results", base / "gt"
            cloud_topic = "/khronos_node/visualization/dynamics_cloud"
            labels_topic = "/tesse/seg_cam/converted/image_raw"
            self.write_bag(result_bag, cloud_topic, "sensor_msgs/msg/PointCloud2", [
                (self.real_cloud(1, False), 1_100_000_000),
                (self.real_cloud(1, True), 1_200_000_000),
                (self.real_cloud(2, True), 2_200_000_000)])
            labels = np.full((5, 5), 10, dtype=np.uint16)
            labels[1:4, 1:4] = 464
            image = self.Image()
            image.header.stamp.sec = 1
            image.width = image.height = 5
            image.encoding, image.step = "16UC1", 10
            image.data = array("B", labels.astype("<u2").tobytes())
            self.write_bag(gt_bag, labels_topic, "sensor_msgs/msg/Image", [(image, 1_000_000_000)])
            remap = base / "remap.yaml"
            remap.write_text("- {sub_id: 10, super_id: 3}\n- {sub_id: 464, super_id: 20}\n")
            output = base / "report"
            args = ["--results-bag", str(result_bag), "--ground-truth-bag", str(gt_bag),
                    "--label-remap", str(remap), "--foreground-labels", "20", "--output", str(output),
                    "--thresholds", "0.99", "--boundary-radius", "1", "--score-bins", "100"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluator.main(args), 0)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["snapshots"], 3)
            self.assertEqual(report["matched_windows"], 1)
            self.assertEqual(report["unmatched_windows"], 1)
            self.assertAlmostEqual(report["threshold_metrics"][0]["recall_all"], 2 / 9)
            self.assertAlmostEqual(report["regions"]["foreground"]["coverage"], 8 / 9)
            self.assertTrue((output / "precision_recall.csv").is_file())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(evaluator.main(args), 1)  # Existing reports are protected.


if __name__ == "__main__":
    unittest.main()
