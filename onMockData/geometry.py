"""
Geometry-related helper functions for the detector model.
"""

import math

import numpy as np


EPS = 1e-12


def normalize(vector):
    """Return a normalized copy of the vector."""
    norm = np.linalg.norm(vector)
    if norm < EPS:
        return vector.copy()
    return vector / norm


def axis_angle_rotation(axis, angle_rad):
    """Build a 3D rotation matrix from axis-angle parameters."""
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c

    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def rotation_from_local_axis(direction_world, local_forward):
    """
    Build a rotation matrix that maps a chosen local detector axis
    to the requested world-space direction.
    """
    a = normalize(local_forward)
    b = normalize(direction_world)

    c = np.clip(np.dot(a, b), -1.0, 1.0)

    if c > 1.0 - 1e-12:
        return np.eye(3)

    if c < -1.0 + 1e-12:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, helper))
        return axis_angle_rotation(axis, np.pi)

    axis = normalize(np.cross(a, b))
    angle = np.arccos(c)
    return axis_angle_rotation(axis, angle)


def infer_detector_forward_axis(diode_normals_local):
    """
    Infer one main detector "forward" direction from diode normals.

    For the current planar detector all diode normals are the same,
    so the mean normal is the natural choice.
    """
    mean_normal = np.mean(diode_normals_local, axis=0)
    if np.linalg.norm(mean_normal) > EPS:
        return normalize(mean_normal)

    # Fallback for exotic cases where normals cancel each other out.
    for normal in diode_normals_local:
        if np.linalg.norm(normal) > EPS:
            return normalize(normal)

    return np.array([0.0, 0.0, 1.0], dtype=float)


def make_detector_geometry(num_channels=6, detector_radius=0.04):
    """
    Create several diodes distributed on one plane.

    All diodes point along the local +Z axis.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_channels, endpoint=False)
    positions = np.zeros((num_channels, 3), dtype=float)

    for index, angle in enumerate(angles):
        positions[index, 0] = detector_radius * math.cos(angle)
        positions[index, 1] = detector_radius * math.sin(angle)
        positions[index, 2] = 0.0

    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (num_channels, 1))
    detector_forward_local = infer_detector_forward_axis(normals)

    return {
        "diode_positions_local": positions,
        "diode_normals_local": normals,
        "detector_forward_local": detector_forward_local,
    }


def make_detector_trajectory(
    geometry,
    source_position,
    num_measurements,
    rng,
    angular_jitter_deg=4.0,
):
    """
    Generate a moving detector trajectory around the source.

    The detector roughly points towards the source, but with small
    random orientation errors to make the problem less idealized.
    """
    rotations = np.zeros((num_measurements, 3, 3), dtype=float)
    translations = np.zeros((num_measurements, 3), dtype=float)

    angles = np.linspace(0.0, 2.0 * np.pi, num_measurements, endpoint=False)

    for i, angle in enumerate(angles):
        radius = 0.65 + 0.20 * math.sin(1.5 * angle) + 0.05 * rng.normal()
        height = 0.15 * math.sin(2.0 * angle) + 0.08 * rng.normal()

        detector_position = source_position + np.array(
            [
                radius * math.cos(angle),
                radius * math.sin(angle),
                0.18 + height,
            ]
        )

        ideal_view_direction = normalize(source_position - detector_position)
        rotation = rotation_from_local_axis(
            direction_world=ideal_view_direction,
            local_forward=geometry["detector_forward_local"],
        )

        # Small orientation mismatch to make the generated data a bit more realistic.
        jitter_axis = normalize(rng.normal(size=3))
        jitter_angle = math.radians(angular_jitter_deg) * rng.normal()
        rotation = axis_angle_rotation(jitter_axis, jitter_angle) @ rotation

        rotations[i] = rotation
        translations[i] = detector_position

    return rotations, translations
