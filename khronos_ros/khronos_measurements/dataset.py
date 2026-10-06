"""One NPZ per processed frame, with JSON metadata and no pickled objects."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


ARRAYS = ("depth", "vertices", "semantic_labels", "dynamic_labels", "object_labels")
TRACK_FIELDS = (
    "id", "is_dynamic", "is_active", "confidence", "first_seen_ns", "last_seen_ns",
    "semantic_cluster_id", "dynamic_cluster_id",
)


@dataclass
class Frame:
    sequence: int
    timestamp_ns: int
    frame_id: str
    # Translation XYZ followed by quaternion XYZW; sensor -> world.
    world_t_sensor: np.ndarray
    depth: np.ndarray
    vertices: np.ndarray
    semantic_labels: np.ndarray
    dynamic_labels: np.ndarray
    object_labels: np.ndarray
    tracks: list

    def validate(self):
        if self.sequence < 0 or self.timestamp_ns < 0:
            raise ValueError("Sequence and timestamp must be nonnegative")
        if not self.frame_id:
            raise ValueError("Missing world frame name")
        if self.world_t_sensor.shape != (7,) or not np.isfinite(self.world_t_sensor).all():
            raise ValueError("Expected finite XYZ + XYZW pose")
        if not np.isclose(np.linalg.norm(self.world_t_sensor[3:]), 1.0, atol=1e-5):
            raise ValueError("Pose quaternion is not normalized")
        if self.depth.ndim != 2 or 0 in self.depth.shape:
            raise ValueError("Expected nonempty H x W depth")
        if self.vertices.shape != (*self.depth.shape, 3):
            raise ValueError("Vertex/depth dimensions differ")
        for name in ARRAYS:
            value = getattr(self, name)
            expected = np.float32 if name in ("depth", "vertices") else np.int32
            if value.dtype != expected:
                raise ValueError(f"{name}: expected {expected}, got {value.dtype}")
            if name.endswith("labels") and value.shape != self.depth.shape:
                raise ValueError(f"{name}: label/depth dimensions differ")
        ids = [track["id"] for track in self.tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate persistent track ID")


def _image_array(msg, encoding, dtype, channels=1):
    """Respect ROS image row padding and byte order; return owned native arrays."""
    if msg.encoding != encoding:
        raise ValueError(f"Expected {encoding}, received {msg.encoding}")
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    row_bytes = msg.width * channels * dtype.itemsize
    if msg.step < row_bytes or len(msg.data) != msg.height * msg.step:
        raise ValueError("Invalid image stride or data length")
    shape = (msg.height, msg.width, channels)
    value = np.ndarray(shape, dtype=dtype, buffer=bytes(msg.data),
                       strides=(msg.step, channels * dtype.itemsize, dtype.itemsize))
    value = value.astype(dtype.newbyteorder("="), copy=True)
    return value[..., 0] if channels == 1 else value


def frame_from_message(msg):
    """Convert a MeasurementFrame without importing ROS in the dataset library."""
    for name in ARRAYS:
        header = getattr(msg, name).header
        if (header.stamp.sec, header.stamp.nanosec, header.frame_id) != (
            msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id
        ):
            raise ValueError(f"{name}: header differs from frame header")
    p, q = msg.world_t_sensor.position, msg.world_t_sensor.orientation
    frame = Frame(
        sequence=int(msg.sequence),
        timestamp_ns=int(msg.header.stamp.sec) * 1_000_000_000 + msg.header.stamp.nanosec,
        frame_id=msg.header.frame_id,
        world_t_sensor=np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64),
        depth=_image_array(msg.depth, "32FC1", np.float32),
        vertices=_image_array(msg.vertices, "32FC3", np.float32, 3),
        semantic_labels=_image_array(msg.semantic_labels, "32SC1", np.int32),
        dynamic_labels=_image_array(msg.dynamic_labels, "32SC1", np.int32),
        object_labels=_image_array(msg.object_labels, "32SC1", np.int32),
        tracks=[{field: getattr(track, field) for field in TRACK_FIELDS} for track in msg.tracks],
    )
    frame.validate()
    return frame


class DatasetWriter:
    """Create a new recording; never overwrite an existing output directory."""

    def __init__(self, path, *, topic="/khronos_node/measurements"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {"format": "khronos_measurements", "version": 1,
                         "topic": topic, "closed": False, "frames": 0}
        self.last_sequence = None
        self.last_stamp = None
        self._save_metadata()

    def _save_metadata(self):
        temporary = self.path / "metadata.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2) + "\n")
        temporary.replace(self.path / "metadata.json")

    def append(self, frame):
        if self.metadata["closed"]:
            raise ValueError("Recording is closed")
        frame.validate()
        if self.last_sequence is not None and frame.sequence <= self.last_sequence:
            raise ValueError("Sequence went backwards or repeated; use a new recording per run")
        if self.last_stamp is not None and frame.timestamp_ns <= self.last_stamp:
            raise ValueError("Timestamp went backwards or repeated")
        metadata = {"sequence": frame.sequence, "timestamp_ns": frame.timestamp_ns,
                    "frame_id": frame.frame_id, "tracks": frame.tracks}
        destination = self.path / f"{frame.sequence:020d}.npz"
        temporary = destination.with_suffix(".tmp")
        # Uncompressed output keeps CPU use out of the recording callback.
        with temporary.open("xb") as stream:
            np.savez(stream, metadata=json.dumps(metadata), world_t_sensor=frame.world_t_sensor,
                     **{name: getattr(frame, name) for name in ARRAYS})
        temporary.replace(destination)
        self.last_sequence, self.last_stamp = frame.sequence, frame.timestamp_ns
        self.metadata["frames"] += 1

    def close(self):
        self.metadata["closed"] = True
        self._save_metadata()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()


class Dataset:
    """Lazy frame iterator; only one frame need be resident in memory at a time."""

    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        if (self.metadata.get("format"), self.metadata.get("version")) != (
            "khronos_measurements", 1
        ):
            raise ValueError("Unsupported measurement dataset format")
        self.files = sorted(self.path.glob("*.npz"))

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for path in self.files:
            with np.load(path, allow_pickle=False) as data:
                frame = Frame(**json.loads(str(data["metadata"])),
                              world_t_sensor=data["world_t_sensor"],
                              **{name: data[name] for name in ARRAYS})
            frame.validate()
            yield frame

    def validate(self):
        """Scan contents and report missing sequence intervals, including a late start.

        No consumer can infer missing frames after its last received frame.
        'closed' means the recorder closed cleanly, not that mapping completed.
        """
        expected, last_stamp, count = 0, -1, 0
        missing = []
        for frame in self:
            if frame.sequence < expected or frame.timestamp_ns <= last_stamp:
                raise ValueError("Frames are not strictly ordered")
            if frame.sequence > expected:
                missing.append((expected, frame.sequence - 1))
            expected, last_stamp = frame.sequence + 1, frame.timestamp_ns
            count += 1
        if self.metadata["closed"] and count != self.metadata["frames"]:
            raise ValueError("Frame count differs from closed recording metadata")
        return {"frames": count, "missing_sequence_intervals": missing,
                "closed": self.metadata["closed"]}
