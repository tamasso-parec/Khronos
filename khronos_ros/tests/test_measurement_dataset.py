"""Dataset and wire decoding checks; runnable without a ROS installation."""

from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from khronos_measurements import Dataset, DatasetWriter, Frame, frame_from_message
from khronos_measurements.dataset import ARRAYS, _image_array


def frame(sequence=0):
    return Frame(sequence, 1_000_000_000 + sequence, "odom",
                 np.array([1., 2., 3., 0., 0., 0., 1.]),
                 np.array([[1, 2], [0, np.nan]], dtype=np.float32),
                 np.arange(12, dtype=np.float32).reshape(2, 2, 3),
                 np.array([[3, 3], [0, 0]], dtype=np.int32),
                 np.array([[0, 1], [0, 0]], dtype=np.int32),
                 np.array([[2, 2], [0, 0]], dtype=np.int32),
                 [{"id": 42, "is_dynamic": True, "is_active": True,
                   "confidence": 0.5, "first_seen_ns": 1_000_000_000,
                   "last_seen_ns": 1_000_000_000, "semantic_cluster_id": 2,
                   "dynamic_cluster_id": 1}])


def test_roundtrip_and_gaps(tmp_path):
    path = tmp_path / "recording"
    with DatasetWriter(path) as writer:
        writer.append(frame(2))
        writer.append(replace(frame(4), tracks=[]))
    dataset = Dataset(path)
    assert dataset.validate() == {"frames": 2, "closed": True,
                                  "missing_sequence_intervals": [(0, 1), (3, 3)]}
    actual, empty = list(dataset)
    assert actual.tracks == frame(2).tracks
    assert empty.tracks == []
    assert actual.timestamp_ns == frame(2).timestamp_ns
    for name in (*ARRAYS, "world_t_sensor"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(frame(2), name))
    with pytest.raises(FileExistsError):
        DatasetWriter(path)


def test_restarts_and_incomplete_recordings(tmp_path):
    path = tmp_path / "recording"
    writer = DatasetWriter(path)
    writer.append(frame())
    with pytest.raises(ValueError, match="Sequence"):
        writer.append(frame())
    assert Dataset(path).validate()["closed"] is False
    writer.close()
    next(path.glob("*.npz")).unlink()
    with pytest.raises(ValueError, match="Frame count"):
        Dataset(path).validate()


def image(array, encoding, header):
    # Exercise byte-swapped storage with padding at the end of every row.
    value = array.astype(array.dtype.newbyteorder(">"))
    rows = [row.tobytes() + b"\x00\x00\x00\x00" for row in value]
    return SimpleNamespace(height=array.shape[0], width=array.shape[1],
                           encoding=encoding, is_bigendian=True,
                           step=len(rows[0]), data=b"".join(rows), header=header)


def test_wire_conversion():
    source = frame()
    header = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0), frame_id="odom")
    message = SimpleNamespace(
        header=header, sequence=0,
        world_t_sensor=SimpleNamespace(position=SimpleNamespace(x=1., y=2., z=3.),
                                       orientation=SimpleNamespace(x=0., y=0., z=0., w=1.)),
        tracks=[SimpleNamespace(**track) for track in source.tracks],
        **{name: image(getattr(source, name),
                       "32FC3" if name == "vertices" else
                       "32FC1" if name == "depth" else "32SC1", header)
           for name in ARRAYS})
    converted = frame_from_message(message)
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(converted, name), getattr(source, name))
    message.depth.header = SimpleNamespace(stamp=header.stamp, frame_id="wrong")
    with pytest.raises(ValueError, match="header"):
        frame_from_message(message)


def test_invalid_images_and_geometry():
    msg = image(frame().depth, "32FC1", None)
    msg.step = 1
    with pytest.raises(ValueError, match="stride"):
        _image_array(msg, "32FC1", np.float32)
    with pytest.raises(ValueError, match="dimensions"):
        replace(frame(), vertices=np.zeros((1, 1, 3), dtype=np.float32)).validate()


@pytest.mark.parametrize("cluster_id, succeeds", [(1, True), (99, False)])
def test_checker(tmp_path, monkeypatch, capsys, cluster_id, succeeds):
    path = tmp_path / "recording"
    sample = frame()
    sample.tracks[0]["dynamic_cluster_id"] = cluster_id
    with DatasetWriter(path) as writer:
        writer.append(sample)
    script = Path(__file__).parents[1] / "scripts/check_measurements.py"
    monkeypatch.setattr(sys, "argv", [str(script), str(path)])
    if succeeds:
        runpy.run_path(str(script), run_name="__main__")
        assert "PASS:" in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit) as result:
            runpy.run_path(str(script), run_name="__main__")
        assert result.value.code == 1
