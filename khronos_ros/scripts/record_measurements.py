#!/usr/bin/env python3
"""Record numerical Khronos frames for offline Python experiments."""

import argparse

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from khronos_msgs.msg import MeasurementFrame
from khronos_measurements import Dataset, DatasetWriter, frame_from_message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="New output directory (must not exist)")
    parser.add_argument("--topic", default="/khronos_node/measurements")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = Node("khronos_measurement_recorder")
    try:
        with DatasetWriter(args.output, topic=args.topic) as writer:
            def receive(msg):
                frame = frame_from_message(msg)
                expected = 0 if writer.last_sequence is None else writer.last_sequence + 1
                if frame.sequence != expected:
                    node.get_logger().warning(
                        f"Missing measurement sequences {expected}..{frame.sequence - 1}")
                writer.append(frame)

            subscription = node.create_subscription(
                MeasurementFrame, args.topic, receive,
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
            node.get_logger().info(f"Recording {args.topic} to {args.output}; Ctrl-C to close")
            try:
                rclpy.spin(node)
            except (KeyboardInterrupt, ExternalShutdownException):
                pass
        print(Dataset(args.output).validate())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
