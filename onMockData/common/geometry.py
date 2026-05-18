"""
Geometry-related helper functions for the detector model.
"""

import math

import numpy as np


EPS = 1e-12
MIN_TRAJECTORY_DISTANCE = 1.0


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


def make_detector_geometry(num_channels=16, diode_pitch=0.15875, channel_height=None):
    """
    Create a linear detector with diodes placed side by side.

    diode_pitch is the center-to-center channel pitch in centimeters.
    For adjacent channels without gaps, it is also the channel width.
    channel_height is the active channel height in centimeters. If it is not
    provided, channels are modeled as square diode_pitch x diode_pitch pixels.
    All diodes point along the local +Z axis.
    """
    if channel_height is None:
        channel_height = diode_pitch

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
        "channel_width": float(diode_pitch),
        "channel_height": float(channel_height),
        "channel_area": float(diode_pitch) * float(channel_height),
        "active_length": float(num_channels * diode_pitch),
        "surface_sample_count": 3,
    }


def detector_extent_points_local(geometry):
    """
    Return local detector points that cover the active channel line.

    Channel centers alone miss half a channel at both physical ends, so add
    the two edge points when the channel width is known.
    """
    points = np.asarray(geometry["diode_positions_local"], dtype=float)
    channel_width = float(geometry.get("channel_width", 0.0))
    if points.size == 0 or channel_width <= 0.0:
        return points

    min_x = float(np.min(points[:, 0])) - 0.5 * channel_width
    max_x = float(np.max(points[:, 0])) + 0.5 * channel_width
    edge_points = np.array(
        [
            [min_x, 0.0, 0.0],
            [max_x, 0.0, 0.0],
        ],
        dtype=float,
    )
    return np.vstack([points, edge_points])


def derive_search_volume_from_detector_window(
    geometry,
    rotations,
    translations,
    search_start_distance,
    search_stop_distance,
    search_start_radius,
    search_stop_radius,
):
    """
    Derive an axis-aligned search box from a detector-relative conical window.

    For every measurement, the searched points start at
    search_start_distance from every active detector-line point and end at
    search_stop_distance along that measurement's detector-forward direction.
    search_start_radius and search_stop_radius define the perpendicular radius
    of that tube/cone at the corresponding distances.
    The returned volume is the axis-aligned bounding box around all those
    start/stop cross sections.
    """
    start_distance = float(search_start_distance)
    stop_distance = float(search_stop_distance)
    start_radius = float(search_start_radius)
    stop_radius = float(search_stop_radius)
    if start_distance < 0.0:
        raise ValueError("search_start_distance must be >= 0 cm.")
    if stop_distance <= start_distance:
        raise ValueError("search_stop_distance must be greater than search_start_distance.")
    if start_radius < 0.0:
        raise ValueError("search_start_radius must be >= 0 cm.")
    if stop_radius < 0.0:
        raise ValueError("search_stop_radius must be >= 0 cm.")

    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    detector_points_local = detector_extent_points_local(geometry)
    detector_points_world = (
        translations[:, np.newaxis, :]
        + np.einsum("mij,pj->mpi", rotations, detector_points_local)
    )

    forward_local = np.asarray(geometry["detector_forward_local"], dtype=float)
    forward_world = np.einsum("mij,j->mi", rotations, forward_local)
    forward_norms = np.linalg.norm(forward_world, axis=1, keepdims=True)
    forward_world = forward_world / np.maximum(forward_norms, EPS)

    perpendicular_extent_start = start_radius * np.sqrt(
        np.maximum(0.0, 1.0 - forward_world[:, np.newaxis, :] ** 2)
    )
    perpendicular_extent_stop = stop_radius * np.sqrt(
        np.maximum(0.0, 1.0 - forward_world[:, np.newaxis, :] ** 2)
    )

    start_centers = detector_points_world + start_distance * forward_world[:, np.newaxis, :]
    stop_centers = detector_points_world + stop_distance * forward_world[:, np.newaxis, :]
    window_min = np.minimum(
        np.min(start_centers - perpendicular_extent_start, axis=(0, 1)),
        np.min(stop_centers - perpendicular_extent_stop, axis=(0, 1)),
    )
    window_max = np.maximum(
        np.max(start_centers + perpendicular_extent_start, axis=(0, 1)),
        np.max(stop_centers + perpendicular_extent_stop, axis=(0, 1)),
    )

    return {
        "search_min": window_min,
        "search_max": window_max,
        "search_start_distance": start_distance,
        "search_stop_distance": stop_distance,
        "search_distance_window": stop_distance - start_distance,
        "search_start_radius": start_radius,
        "search_stop_radius": stop_radius,
    }


def points_in_detector_window(
    points,
    geometry,
    rotations,
    translations,
    search_start_distance,
    search_stop_distance,
    search_start_radius,
    search_stop_radius,
):
    """Return a mask for points inside the union of detector-facing cones."""
    points = np.asarray(points, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    start_distance = float(search_start_distance)
    stop_distance = float(search_stop_distance)
    start_radius = float(search_start_radius)
    stop_radius = float(search_stop_radius)

    detector_points_local = detector_extent_points_local(geometry)
    detector_points_world = (
        translations[:, np.newaxis, :]
        + np.einsum("mij,pj->mpi", rotations, detector_points_local)
    )
    forward_local = np.asarray(geometry["detector_forward_local"], dtype=float)
    forward_world = np.einsum("mij,j->mi", rotations, forward_local)
    forward_world = forward_world / np.maximum(
        np.linalg.norm(forward_world, axis=1, keepdims=True),
        EPS,
    )

    inside = np.zeros(len(points), dtype=bool)
    distance_window = stop_distance - start_distance
    for measurement_index, forward in enumerate(forward_world):
        diffs = points[:, np.newaxis, :] - detector_points_world[measurement_index][np.newaxis, :, :]
        axial_distance = np.einsum("npj,j->np", diffs, forward)
        radius = start_radius + (
            (axial_distance - start_distance) / distance_window
        ) * (stop_radius - start_radius)
        perpendicular_squared = np.sum(diffs * diffs, axis=2) - axial_distance * axial_distance
        in_measurement = (
            (axial_distance >= start_distance)
            & (axial_distance <= stop_distance)
            & (perpendicular_squared <= radius * radius)
        )
        inside |= np.any(in_measurement, axis=1)

    return inside


def derive_search_volume_from_detector_box(
    geometry,
    rotations,
    translations,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
):
    """
    Derive an axis-aligned search box from detector-relative rectangular windows.

    For every measurement, the searched local detector box spans:
        x: [-search_width / 2, search_width / 2]
        y: [-search_height / 2, search_height / 2]
        z: [search_start_distance, search_stop_distance]

    Local +Z is the detector-forward axis.
    """
    start_distance = float(search_start_distance)
    stop_distance = float(search_stop_distance)
    width = float(search_width)
    height = float(search_height)
    half_width = 0.5 * width
    half_height = 0.5 * height
    if start_distance < 0.0:
        raise ValueError("search_start_distance must be >= 0 cm.")
    if stop_distance <= start_distance:
        raise ValueError("search_stop_distance must be greater than search_start_distance.")
    if width < 0.0:
        raise ValueError("search_width must be >= 0 cm.")
    if height < 0.0:
        raise ValueError("search_height must be >= 0 cm.")

    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    local_corners = np.array(
        [
            [x, y, z]
            for x in (-half_width, half_width)
            for y in (-half_height, half_height)
            for z in (start_distance, stop_distance)
        ],
        dtype=float,
    )
    world_corners = (
        translations[:, np.newaxis, :]
        + np.einsum("mij,cj->mci", rotations, local_corners)
    )
    window_min = np.min(world_corners, axis=(0, 1))
    window_max = np.max(world_corners, axis=(0, 1))

    return {
        "search_min": window_min,
        "search_max": window_max,
        "search_start_distance": start_distance,
        "search_stop_distance": stop_distance,
        "search_distance_window": stop_distance - start_distance,
        "search_width": width,
        "search_height": height,
    }


def points_in_detector_box(
    points,
    geometry,
    rotations,
    translations,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
):
    """Return a mask for points inside the union of detector-relative boxes."""
    points = np.asarray(points, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    start_distance = float(search_start_distance)
    stop_distance = float(search_stop_distance)
    width = float(search_width)
    height = float(search_height)
    half_width = 0.5 * width
    half_height = 0.5 * height

    inside = np.zeros(len(points), dtype=bool)
    for measurement_index, rotation in enumerate(rotations):
        local_points = (points - translations[measurement_index]) @ rotation
        in_measurement = (
            (local_points[:, 2] >= start_distance)
            & (local_points[:, 2] <= stop_distance)
            & (np.abs(local_points[:, 0]) <= half_width)
            & (np.abs(local_points[:, 1]) <= half_height)
        )
        inside |= in_measurement

    return inside


def make_detector_trajectory(
    geometry,
    trajectory_center,
    num_measurements,
    rng,
    trajectory_mode="grid",
    angular_jitter_deg=2.0,
    lateral_span=45.0,
    vertical_span=28.0,
    base_distance=65.0,
    distance_span=18.0,
    orbit_radius=65.0,
    orbit_radius_span=0.0,
    orbit_angle_span_deg=360.0,
    orbit_start_angle_deg=0.0,
    orbit_vertical_span=0.0,
    orientation_mode="fixed",
    fixed_view_direction=None,
    fixed_sweep_z_range_deg=0.0,
    fixed_sweep_x_range_deg=0.0,
):
    """
    Generate a detector trajectory around the trajectory center.

    All position-like values use centimeters.

    trajectory_mode:
        - "grid": one-sided left/right, up/down scan at several distances.
        - "line": one left-to-right scan line at trajectory_center height.
        - "around_center": orbit around trajectory_center in the XY plane.
          orbit_angle_span_deg controls the covered arc, e.g. 180 for a
          half-turn and 360 for a full orbit.

    The nominal viewing direction is either constant, or it can be re-aimed
    towards the provided trajectory_center.
    """
    trajectory_center = np.asarray(trajectory_center, dtype=float)
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
    elif trajectory_mode == "line":
        num_columns = max(1, int(num_measurements))
    elif trajectory_mode == "around_center":
        full_orbit = abs(abs(float(orbit_angle_span_deg)) - 360.0) < 1e-9
        angle_denominator = num_measurements if full_orbit else max(num_measurements - 1, 1)
    else:
        raise ValueError(
            "trajectory_mode must be 'grid', 'line', or 'around_center'. Got: "
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
            standoff_distance += 0.05 * max(abs(float(base_distance)), MIN_TRAJECTORY_DISTANCE) * rng.normal()
            standoff_distance = max(MIN_TRAJECTORY_DISTANCE, standoff_distance)

            detector_position = (
                trajectory_center
                + lateral_offset * lateral_axis
                + standoff_distance * standoff_axis
                + vertical_offset * vertical_axis
            )
        elif trajectory_mode == "line":
            column_fraction = 0.5 if num_columns == 1 else i / (num_columns - 1)
            lateral_offset = (2.0 * column_fraction - 1.0) * lateral_span
            standoff_distance = max(MIN_TRAJECTORY_DISTANCE, float(base_distance))
            detector_position = (
                trajectory_center
                + lateral_offset * lateral_axis
                + standoff_distance * standoff_axis
                + float(vertical_span) * vertical_axis
            )
        else:
            orbit_fraction = 0.0 if num_measurements <= 1 else i / angle_denominator
            angle = math.radians(float(orbit_start_angle_deg))
            angle += math.radians(float(orbit_angle_span_deg)) * orbit_fraction
            radial_direction = (
                math.cos(angle) * standoff_axis
                + math.sin(angle) * lateral_axis
            )
            radius = orbit_radius + orbit_radius_span * math.sin(2.0 * np.pi * orbit_fraction)
            radius = max(MIN_TRAJECTORY_DISTANCE, radius)
            vertical_offset = orbit_vertical_span * math.sin(2.0 * np.pi * orbit_fraction)
            detector_position = trajectory_center + radius * radial_direction + vertical_offset * vertical_axis

        if base_rotation is None:
            view_direction = trajectory_center - detector_position
            if np.linalg.norm(view_direction) < EPS:
                view_direction = nominal_view_direction
            rotation = rotation_from_local_axis(
                direction_world=view_direction,
                local_forward=geometry["detector_forward_local"],
            )
        else:
            rotation = base_rotation

        z_range = abs(float(fixed_sweep_z_range_deg))
        x_range = abs(float(fixed_sweep_x_range_deg))
        z_angle = math.radians(rng.uniform(-z_range, z_range))
        x_angle = math.radians(rng.uniform(-x_range, x_range))
        sweep_rotation = (
            axis_angle_rotation(vertical_axis, z_angle)
            @ axis_angle_rotation(lateral_axis, x_angle)
        )
        rotation = sweep_rotation @ rotation

        # Small orientation mismatch to make the generated data a bit more realistic.
        jitter_axis = normalize(rng.normal(size=3))
        jitter_angle = math.radians(angular_jitter_deg) * rng.normal()
        rotation = axis_angle_rotation(jitter_axis, jitter_angle) @ rotation

        rotations[i] = rotation
        translations[i] = detector_position

    return rotations, translations
