"""
Synthetic data generation for multi-point source reconstruction experiments.
"""

import numpy as np

from common.forward_model import compute_detector_response
from common.geometry import (
    derive_search_volume_from_detector_box,
    EPS,
    make_detector_geometry,
    make_detector_trajectory,
)


def gaussian_source_components(
    position,
    strength,
    spread_cm,
    component_spacing_cm=0.25,
    edge_relative_activity=0.25,
):
    """
    Approximate a diffuse 3D Gaussian source by a dense deterministic cloud.

    The returned components are the source used by the forward model. They are
    sampled on a fixed spatial grid, so smaller spread_cm makes a smaller blob
    instead of packing the same number of points into a smaller volume.
    spread_cm is treated as the visible blob radius: edge_relative_activity
    controls how much activity remains at that radius compared with the center.
    Their strengths sum to the requested total strength. spread_cm == 0 keeps a
    point source.
    """
    position = np.asarray(position, dtype=float)
    strength = float(strength)
    spread_cm = float(spread_cm)
    component_spacing_cm = float(component_spacing_cm)
    if component_spacing_cm <= EPS:
        raise ValueError("component_spacing_cm must be > 0.")
    edge_relative_activity = float(edge_relative_activity)
    if not 0.0 < edge_relative_activity < 1.0:
        raise ValueError("edge_relative_activity must be between 0 and 1.")
    if spread_cm <= EPS:
        return [
            {
                "position": position.copy(),
                "strength": strength,
                "weight": 1.0,
            }
        ]

    radius_cm = spread_cm
    sigma_cm = radius_cm / np.sqrt(-2.0 * np.log(edge_relative_activity))
    max_offset_steps = int(np.ceil(radius_cm / component_spacing_cm))
    axis_offsets = (
        np.arange(-max_offset_steps, max_offset_steps + 1, dtype=float)
        * component_spacing_cm
    )
    axis_offsets = axis_offsets[np.abs(axis_offsets) <= radius_cm + EPS]

    offsets = []
    weights = []
    for x_offset in axis_offsets:
        for y_offset in axis_offsets:
            for z_offset in axis_offsets:
                offset = np.array([x_offset, y_offset, z_offset], dtype=float)
                radius_squared = float(np.dot(offset, offset))
                if radius_squared > radius_cm**2:
                    continue
                normalized_radius_squared = radius_squared / max(sigma_cm**2, EPS)
                offsets.append(offset)
                weights.append(float(np.exp(-0.5 * normalized_radius_squared)))

    weights = np.asarray(weights, dtype=float)
    weights /= max(float(np.sum(weights)), EPS)
    return [
        {
            "position": position + offset,
            "strength": strength * weight,
            "weight": weight,
        }
        for offset, weight in zip(offsets, weights)
    ]


def expected_counts_multi_source(
    sources,
    geometry,
    rotations,
    translations,
    channel_scales,
    backgrounds,
    k,
    distance_power,
    angular_peak_angle_deg=0.0,
    angular_width_deg=None,
    skin_thickness_cm=0.0,
    skin_mu_per_cm=0.0,
    detector_intrinsic_efficiency=1.0,
    response_scale=1.0,
):
    """Forward model for several possibly extended sources."""
    mean_counts = np.broadcast_to(backgrounds, (len(translations), len(backgrounds))).copy()

    for source in sources:
        components = source.get("components")
        if components is None:
            components = gaussian_source_components(
                position=source["position"],
                strength=source["strength"],
                spread_cm=source.get("spread_cm", 0.0),
            )

        for component in components:
            response = compute_detector_response(
                source_position=component["position"],
                geometry=geometry,
                rotations=rotations,
                translations=translations,
                channel_scales=channel_scales,
                k=k,
                distance_power=distance_power,
                angular_peak_angle_deg=angular_peak_angle_deg,
                angular_width_deg=angular_width_deg,
                skin_thickness_cm=skin_thickness_cm,
                skin_mu_per_cm=skin_mu_per_cm,
                detector_intrinsic_efficiency=detector_intrinsic_efficiency,
                response_scale=response_scale,
            )
            mean_counts += component["strength"] * response

    return mean_counts


def generate_synthetic_dataset(
    num_measurements,
    seed,
    orientation_mode,
    fixed_view_direction,
    source_specs,
    num_channels,
    diode_pitch,
    background_mean_per_second,
    background_channel_std_per_second,
    channel_scale_mean,
    channel_scale_std,
    generation_k,
    generation_distance_power,
    measurement_duration_seconds,
    trajectory_center,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
    trajectory_mode,
    angular_jitter_deg,
    lateral_span,
    vertical_span,
    base_distance,
    distance_span,
    orbit_radius,
    orbit_radius_span,
    orbit_angle_span_deg,
    orbit_start_angle_deg,
    orbit_vertical_span,
    fixed_sweep_z_range_deg,
    fixed_sweep_x_range_deg,
    generation_angular_peak_angle_deg=0.0,
    generation_angular_width_deg=None,
    generation_skin_thickness_cm=0.0,
    generation_skin_mu_per_cm=0.0,
    generation_detector_intrinsic_efficiency=1.0,
    generation_response_scale=1.0,
):
    """
    Generate one synthetic multi-source experiment.

    All source positions, detector offsets, and scan dimensions use centimeters.

    orientation_mode:
        - "fixed": keep detector pointing in one direction (default).
        - "towards_target": re-aim detector towards trajectory_center each measurement.

    trajectory_mode:
        - "grid": current one-sided scan.
        - "line": one left-to-right scan line at trajectory_center height.
        - "around_center": detector positions orbit around trajectory_center.

    fixed_view_direction:
        3D vector for "fixed" mode, e.g. (0, 1, 0) or (1, 0, 0).

    background_mean_per_second and source strength_per_second values are rates.
    They are multiplied by measurement_duration_seconds to generate counts for
    one measurement.

    The output keeps true_sources as a list because multi-point algorithms may
    reconstruct a map rather than one estimated position.
    """
    rng = np.random.default_rng(seed)

    geometry = make_detector_geometry(num_channels=num_channels, diode_pitch=diode_pitch)
    num_channels = len(geometry["diode_positions_local"])

    if not source_specs:
        raise ValueError("source_specs must contain at least one source.")

    measurement_duration_seconds = float(measurement_duration_seconds)
    if measurement_duration_seconds <= 0.0:
        raise ValueError("measurement_duration_seconds must be > 0.")

    true_sources = []
    for source_index, source in enumerate(source_specs, start=1):
        if "strength_per_second" not in source:
            raise KeyError(
                f"source_specs[{source_index}] must define strength_per_second."
            )
        strength_per_second = float(source["strength_per_second"])
        if strength_per_second < 0.0:
            raise ValueError("source strength_per_second must be >= 0.")
        spread_cm = float(source.get("spread_cm", 0.0))
        if spread_cm < 0.0:
            raise ValueError("source spread_cm must be >= 0.")
        strength = strength_per_second * measurement_duration_seconds
        components = gaussian_source_components(
            position=source["position"],
            strength=strength,
            spread_cm=spread_cm,
        )
        true_sources.append(
            {
                "position": np.array(source["position"], dtype=float),
                "strength_per_second": strength_per_second,
                "strength": strength,
                "spread_cm": spread_cm,
                "components": components,
            }
        )
    if trajectory_center is None:
        raise ValueError("trajectory_center must be provided explicitly.")
    trajectory_center = np.array(trajectory_center, dtype=float)

    background_mean = float(background_mean_per_second) * measurement_duration_seconds
    background_channel_std = float(background_channel_std_per_second) * measurement_duration_seconds

    true_backgrounds = rng.normal(
        loc=background_mean,
        scale=background_channel_std,
        size=num_channels,
    )
    true_channel_scales = rng.normal(
        loc=channel_scale_mean,
        scale=channel_scale_std,
        size=num_channels,
    )

    rotations, translations = make_detector_trajectory(
        geometry=geometry,
        trajectory_center=trajectory_center,
        num_measurements=num_measurements,
        rng=rng,
        trajectory_mode=trajectory_mode,
        angular_jitter_deg=angular_jitter_deg,
        lateral_span=lateral_span,
        vertical_span=vertical_span,
        base_distance=base_distance,
        distance_span=distance_span,
        orbit_radius=orbit_radius,
        orbit_radius_span=orbit_radius_span,
        orbit_angle_span_deg=orbit_angle_span_deg,
        orbit_start_angle_deg=orbit_start_angle_deg,
        orbit_vertical_span=orbit_vertical_span,
        orientation_mode=orientation_mode,
        fixed_view_direction=fixed_view_direction,
        fixed_sweep_z_range_deg=fixed_sweep_z_range_deg,
        fixed_sweep_x_range_deg=fixed_sweep_x_range_deg,
    )
    search_volume = derive_search_volume_from_detector_box(
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        search_start_distance=search_start_distance,
        search_stop_distance=search_stop_distance,
        search_width=search_width,
        search_height=search_height,
    )

    mean_counts = expected_counts_multi_source(
        sources=true_sources,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        channel_scales=true_channel_scales,
        backgrounds=true_backgrounds,
        k=generation_k,
        distance_power=generation_distance_power,
        angular_peak_angle_deg=generation_angular_peak_angle_deg,
        angular_width_deg=generation_angular_width_deg,
        skin_thickness_cm=generation_skin_thickness_cm,
        skin_mu_per_cm=generation_skin_mu_per_cm,
        detector_intrinsic_efficiency=generation_detector_intrinsic_efficiency,
        response_scale=generation_response_scale,
    )

    noisy_counts = rng.poisson(mean_counts).astype(float)

    return {
        "counts": noisy_counts,
        "mean_counts": mean_counts,
        "rotations": rotations,
        "translations": translations,
        "geometry": geometry,
        "true_sources": true_sources,
        "trajectory_center": trajectory_center,
        "search_min": np.array(search_volume["search_min"], dtype=float),
        "search_max": np.array(search_volume["search_max"], dtype=float),
        "search_start_distance": search_volume["search_start_distance"],
        "search_stop_distance": search_volume["search_stop_distance"],
        "search_distance_window": search_volume["search_distance_window"],
        "search_width": search_volume["search_width"],
        "search_height": search_volume["search_height"],
        "true_backgrounds": true_backgrounds,
        "true_channel_scales": true_channel_scales,
        "generation_k": generation_k,
        "generation_distance_power": generation_distance_power,
        "generation_angular_peak_angle_deg": generation_angular_peak_angle_deg,
        "generation_angular_width_deg": generation_angular_width_deg,
        "generation_skin_thickness_cm": generation_skin_thickness_cm,
        "generation_skin_mu_per_cm": generation_skin_mu_per_cm,
        "generation_detector_intrinsic_efficiency": generation_detector_intrinsic_efficiency,
        "generation_response_scale": generation_response_scale,
        "measurement_duration_seconds": measurement_duration_seconds,
        "background_mean_per_second": background_mean_per_second,
        "background_mean": background_mean,
        "background_channel_std_per_second": background_channel_std_per_second,
        "background_channel_std": background_channel_std,
        "channel_scale_mean": channel_scale_mean,
        "channel_scale_std": channel_scale_std,
        "trajectory_mode": trajectory_mode,
    }
