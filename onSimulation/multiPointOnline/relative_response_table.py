"""
OpenGATE relative detector-response table utilities.

The table is generated once with a unit source placed at many source positions
relative to the detector. During reconstruction each candidate voxel is mapped
into the local detector frame and the per-channel response is interpolated.
"""

from pathlib import Path
import csv
from functools import lru_cache

import numpy as np


DEFAULT_RESPONSE_COLUMN = "response_weighted_probability"


def _as_path(path):
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _bracket_axis(axis, value):
    """Return low/high indices and interpolation weight for one sorted axis."""
    value = float(value)
    if len(axis) == 1:
        return 0, 0, 0.0
    if value < axis[0] or value > axis[-1]:
        return None
    high = int(np.searchsorted(axis, value, side="left"))
    if high == 0:
        return 0, 0, 0.0
    if high >= len(axis):
        return len(axis) - 1, len(axis) - 1, 0.0
    low = high - 1
    span = float(axis[high] - axis[low])
    if span <= 0.0:
        return low, high, 0.0
    return low, high, float((value - axis[low]) / span)


class RelativeResponseTable:
    """
    Regular-grid OpenGATE response table.

    Values are probabilities/counts-per-unit-emission for each detector channel.
    The expected CSV columns are written by gate_build_single_voxel_h_response.py:
    source_local_x_cm, source_local_y_cm, source_local_z_cm, channel_index, and
    response_weighted_probability or response_raw_probability.
    """

    def __init__(self, csv_path, response_column=DEFAULT_RESPONSE_COLUMN):
        self.csv_path = _as_path(csv_path)
        if self.csv_path is None:
            raise ValueError("csv_path must be provided.")
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"OpenGATE relative response table not found: {self.csv_path}"
            )

        self.response_column = str(response_column)
        self._load_csv()

    def _load_csv(self):
        rows = []
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "source_local_x_cm",
                "source_local_y_cm",
                "source_local_z_cm",
                "channel_index",
                self.response_column,
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{self.csv_path} misses required columns: {sorted(missing)}"
                )
            for row in reader:
                rows.append(row)

        if not rows:
            raise ValueError(f"{self.csv_path} contains no response rows.")

        self.x_axis = np.array(
            sorted({float(row["source_local_x_cm"]) for row in rows}), dtype=float
        )
        self.y_axis = np.array(
            sorted({float(row["source_local_y_cm"]) for row in rows}), dtype=float
        )
        self.z_axis = np.array(
            sorted({float(row["source_local_z_cm"]) for row in rows}), dtype=float
        )
        self.channel_axis = np.array(
            sorted({int(row["channel_index"]) for row in rows}), dtype=int
        )
        self.num_channels = int(len(self.channel_axis))

        channel_to_index = {
            int(channel_index): i for i, channel_index in enumerate(self.channel_axis)
        }
        values = np.zeros(
            (
                len(self.x_axis),
                len(self.y_axis),
                len(self.z_axis),
                self.num_channels,
            ),
            dtype=float,
        )

        x_to_index = {float(v): i for i, v in enumerate(self.x_axis)}
        y_to_index = {float(v): i for i, v in enumerate(self.y_axis)}
        z_to_index = {float(v): i for i, v in enumerate(self.z_axis)}

        for row in rows:
            x_i = x_to_index[float(row["source_local_x_cm"])]
            y_i = y_to_index[float(row["source_local_y_cm"])]
            z_i = z_to_index[float(row["source_local_z_cm"])]
            channel_i = channel_to_index[int(row["channel_index"])]
            values[x_i, y_i, z_i, channel_i] = float(row[self.response_column])

        self.values = values

    def interpolate_channels(self, source_local_cm):
        """
        Return detector channel response for one source position in detector-local cm.

        Outside the sampled OpenGATE grid the response is set to zero instead of
        extrapolated. That makes reconstruction conservative until the table is
        expanded to cover the full relative pose space.
        """
        source_local_cm = np.asarray(source_local_cm, dtype=float)
        if source_local_cm.shape != (3,):
            raise ValueError("source_local_cm must be a 3-vector.")

        x_bracket = _bracket_axis(self.x_axis, source_local_cm[0])
        y_bracket = _bracket_axis(self.y_axis, source_local_cm[1])
        z_bracket = _bracket_axis(self.z_axis, source_local_cm[2])
        if x_bracket is None or y_bracket is None or z_bracket is None:
            raise ValueError(
                "OpenGATE relative response table does not cover "
                f"source_local_cm={source_local_cm.tolist()}. "
                f"Covered ranges: "
                f"x=[{self.x_axis[0]}, {self.x_axis[-1]}], "
                f"y=[{self.y_axis[0]}, {self.y_axis[-1]}], "
                f"z=[{self.z_axis[0]}, {self.z_axis[-1]}]."
            )

        (x0, x1, tx) = x_bracket
        (y0, y1, ty) = y_bracket
        (z0, z1, tz) = z_bracket

        response = np.zeros(self.num_channels, dtype=float)
        for xi, wx in ((x0, 1.0 - tx), (x1, tx)):
            for yi, wy in ((y0, 1.0 - ty), (y1, ty)):
                for zi, wz in ((z0, 1.0 - tz), (z1, tz)):
                    response += wx * wy * wz * self.values[xi, yi, zi, :]
        return response

    def global_response_channels(self, source_position_cm, detector_rotation, detector_translation_cm):
        """Map a world-space voxel into detector-local coordinates and interpolate."""
        source_position_cm = np.asarray(source_position_cm, dtype=float)
        detector_rotation = np.asarray(detector_rotation, dtype=float)
        detector_translation_cm = np.asarray(detector_translation_cm, dtype=float)
        source_local_cm = detector_rotation.T @ (source_position_cm - detector_translation_cm)
        return self.interpolate_channels(source_local_cm)


@lru_cache(maxsize=8)
def load_relative_response_table(csv_path, response_column=DEFAULT_RESPONSE_COLUMN):
    """Cached loader so online batches reuse the same OpenGATE table."""
    return RelativeResponseTable(csv_path=csv_path, response_column=response_column)
