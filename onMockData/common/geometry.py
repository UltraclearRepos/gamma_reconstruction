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


def make_detector_geometry(num_channels=16, diode_pitch=0.006):
    """
    Create a linear detector with diodes placed side by side.

    All diodes point along the local +Z axis.
    """
    positions = np.zeros((num_channels, 3), dtype=float)
    offsets = np.arange(num_channels, dtype=float) - 0.5 * (num_channels - 1)
    positions[:, 0] = offsets * diode_pitch

    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (num_channels, 1))
    detector_forward_local = infer_detector_forward_axis(normals)

    return {
        "diode_positions_local": positions,
        "diode_normals_local": normals,
        "detector_forward_local": detector_forward_local,
        "diode_pitch": float(diode_pitch),
    }


def make_detector_trajectory(
    geometry,
    scan_center,
    num_measurements,
    rng,
    trajectory_mode="grid",
    angular_jitter_deg=2.0,
    lateral_span=0.45,
    vertical_span=0.28,
    base_distance=0.65,
    distance_span=0.18,
    orbit_radius=0.65,
    orbit_radius_span=0.0,
    orbit_angle_span_deg=360.0,
    orbit_start_angle_deg=0.0,
    orbit_vertical_span=0.0,
    orientation_mode="fixed",
    fixed_view_direction=None,
    fixed_sweep_z_deg=0.0,
    fixed_sweep_x_deg=0.0,
):
    """
    Generate a detector trajectory around the scan center.

    trajectory_mode:
        - "grid": one-sided left/right, up/down scan at several distances.
        - "around_center": orbit around scan_center in the XY plane.

    The nominal viewing direction is either constant, or it can be re-aimed
    towards the provided scan_center.
    """
    rotations = np.zeros((num_measurements, 3, 3), dtype=float)
    translations = np.zeros((num_measurements, 3), dtype=float)

    lateral_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    standoff_axis = np.array([0.0, -1.0, 0.0], dtype=float)
    vertical_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    nominal_view_direction = -standoff_axis

    trajectory_mode = str(trajectory_mode or "grid").strip().lower().replace("-", "_")
    orientation_mode = str(orientation_mode or "fixed").strip().lower().replace("-", "_")
    if fixed_view_direction is not None:
        fixed_view_direction = normalize(np.asarray(fixed_view_direction, dtype=float))
        if np.linalg.norm(fixed_view_direction) < EPS:
            raise ValueError("fixed_view_direction must be a non-zero 3D vector.")

    if orientation_mode == "fixed":
        if fixed_view_direction is None:
            raise ValueError(
                "For orientation_mode == 'fixed' you must provide fixed_view_direction, "
                "e.g. (0, 1, 0) or (1, 0, 0)."
            )
        view_direction = fixed_view_direction
        base_rotation = rotation_from_local_axis(
            direction_world=view_direction,
            local_forward=geometry["detector_forward_local"],
        )
    elif orientation_mode == "towards_target":
        base_rotation = None
    else:
        raise ValueError("orientation_mode must be 'fixed' or 'towards_target'. Got: " f"{orientation_mode!r}")

    if trajectory_mode == "grid":
        num_rows = max(1, int(round(math.sqrt(max(num_measurements, 1) / 2.0))))
        num_columns = max(1, int(math.ceil(num_measurements / num_rows)))
        vertical_positions = (
            np.linspace(-vertical_span, vertical_span, num_rows)
            if num_rows > 1
            else np.array([0.0], dtype=float)
        )
    elif trajectory_mode == "around_center":
        full_orbit = abs(abs(float(orbit_angle_span_deg)) - 360.0) < 1e-9
        angle_denominator = num_measurements if full_orbit else max(num_measurements - 1, 1)
    else:
        raise ValueError(
            "trajectory_mode must be 'grid' or 'around_center'. Got: "
            f"{trajectory_mode!r}"
        )

    for i in range(num_measurements):
        if trajectory_mode == "grid":
            row = i // num_columns
            column = i % num_columns
            column_fraction = 0.5 if num_columns == 1 else column / (num_columns - 1)
            if row % 2 == 1:
                column_fraction = 1.0 - column_fraction

            lateral_offset = (2.0 * column_fraction - 1.0) * lateral_span
            vertical_offset = vertical_positions[row]
            distance_phase = i / max(num_measurements, 1)
            standoff_distance = base_distance + distance_span * math.sin(
                2.0 * np.pi * distance_phase + 0.6 * row
            )
            standoff_distance += 0.03 * rng.normal()
            standoff_distance = max(0.25, standoff_distance)

            detector_position = (
                scan_center
                + lateral_offset * lateral_axis
                + standoff_distance * standoff_axis
                + vertical_offset * vertical_axis
            )
            z_fraction = 2.0 * column_fraction - 1.0
            x_fraction = 0.0 if num_rows == 1 else 2.0 * row / (num_rows - 1) - 1.0
        else:
            orbit_fraction = 0.0 if num_measurements <= 1 else i / angle_denominator
            angle = math.radians(float(orbit_start_angle_deg))
            angle += math.radians(float(orbit_angle_span_deg)) * orbit_fraction
            radial_direction = (
                math.cos(angle) * standoff_axis
                + math.sin(angle) * lateral_axis
            )
            radius = orbit_radius + orbit_radius_span * math.sin(2.0 * np.pi * orbit_fraction)
            radius = max(0.25, radius)
            vertical_offset = orbit_vertical_span * math.sin(2.0 * np.pi * orbit_fraction)
            detector_position = scan_center + radius * radial_direction + vertical_offset * vertical_axis
            z_fraction = math.sin(angle)
            x_fraction = 0.0 if orbit_vertical_span == 0.0 else math.sin(2.0 * np.pi * orbit_fraction)

        if base_rotation is None:
            view_direction = scan_center - detector_position
            if np.linalg.norm(view_direction) < EPS:
                view_direction = nominal_view_direction
            rotation = rotation_from_local_axis(
                direction_world=view_direction,
                local_forward=geometry["detector_forward_local"],
            )
        else:
            z_angle = math.radians(float(fixed_sweep_z_deg)) * z_fraction
            x_angle = math.radians(float(fixed_sweep_x_deg)) * x_fraction
            sweep_rotation = (
                axis_angle_rotation(vertical_axis, z_angle)
                @ axis_angle_rotation(lateral_axis, x_angle)
            )
            rotation = sweep_rotation @ base_rotation

        # Small orientation mismatch to make the generated data a bit more realistic.
        jitter_axis = normalize(rng.normal(size=3))
        jitter_angle = math.radians(angular_jitter_deg) * rng.normal()
        rotation = axis_angle_rotation(jitter_axis, jitter_angle) @ rotation

        rotations[i] = rotation
        translations[i] = detector_position

    return rotations, translations
