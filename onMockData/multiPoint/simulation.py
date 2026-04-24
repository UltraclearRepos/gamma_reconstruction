"""
Synthetic data generation for multi-point source reconstruction experiments.
"""

import numpy as np

from common.forward_model import compute_detector_response
from common.geometry import make_detector_geometry, make_detector_trajectory


def expected_counts_multi_source(
    sources,
    geometry,
    rotations,
    translations,
    channel_scales,
    backgrounds,
    k,
):
    """Forward model for several point sources."""
    mean_counts = np.broadcast_to(backgrounds, (len(translations), len(backgrounds))).copy()

    for source in sources:
        response = compute_detector_response(
            source_position=source["position"],
            geometry=geometry,
            rotations=rotations,
            translations=translations,
            channel_scales=channel_scales,
            k=k,
        )
        mean_counts += source["strength"] * response

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
    peak_total_source_signal_per_second,
    measurement_duration_seconds,
    scan_center,
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
    fixed_sweep_z_deg,
    fixed_sweep_x_deg,
):
    """
    Generate one synthetic multi-source experiment.

    orientation_mode:
        - "fixed": keep detector pointing in one direction (default).
        - "towards_target": re-aim detector towards scan_center each measurement.

    trajectory_mode:
        - "grid_scan": current one-sided scan.
        - "around_center": detector positions orbit around scan_center.

    fixed_view_direction:
        3D vector for "fixed" mode, e.g. (0, 1, 0) or (1, 0, 0).

    background_mean_per_second and peak_total_source_signal_per_second are
    rates. They are multiplied by measurement_duration_seconds to generate
    counts for one measurement.

    The output keeps true_sources as a list because multi-point algorithms may
    reconstruct a map rather than one estimated position.
    """
    rng = np.random.default_rng(seed)

    geometry = make_detector_geometry(num_channels=num_channels, diode_pitch=diode_pitch)
    num_channels = len(geometry["diode_positions_local"])

    if not source_specs:
        raise ValueError("source_specs must contain at least one source.")

    true_sources = [
        {
            "position": np.array(source["position"], dtype=float),
            "relative_strength": float(source["relative_strength"]),
        }
        for source in source_specs
    ]
    if scan_center is None:
        raise ValueError("scan_center must be provided explicitly, e.g. center of reconstruction search volume.")
    scan_center = np.array(scan_center, dtype=float)
    measurement_duration_seconds = float(measurement_duration_seconds)
    if measurement_duration_seconds <= 0.0:
        raise ValueError("measurement_duration_seconds must be > 0.")

    background_mean = float(background_mean_per_second) * measurement_duration_seconds
    background_channel_std = float(background_channel_std_per_second) * measurement_duration_seconds
    peak_total_source_signal = float(peak_total_source_signal_per_second) * measurement_duration_seconds

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
        scan_center=scan_center,
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
        fixed_sweep_z_deg=fixed_sweep_z_deg,
        fixed_sweep_x_deg=fixed_sweep_x_deg,
    )

    unit_sources = [
        {
            "position": source["position"],
            "strength": source["relative_strength"],
        }
        for source in true_sources
    ]
    unit_source_counts = expected_counts_multi_source(
        sources=unit_sources,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        channel_scales=true_channel_scales,
        backgrounds=np.zeros(num_channels, dtype=float),
        k=generation_k,
    )
    strongest_measurement_signal = max(float(np.max(np.mean(unit_source_counts, axis=1))), 1e-12)
    strength_scale = peak_total_source_signal / strongest_measurement_signal

    for source in true_sources:
        source["strength"] = strength_scale * source.pop("relative_strength")

    mean_counts = expected_counts_multi_source(
        sources=true_sources,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        channel_scales=true_channel_scales,
        backgrounds=true_backgrounds,
        k=generation_k,
    )

    noisy_counts = rng.poisson(mean_counts).astype(float)

    return {
        "counts": noisy_counts,
        "rotations": rotations,
        "translations": translations,
        "geometry": geometry,
        "true_sources": true_sources,
        "scan_center": scan_center,
        "true_backgrounds": true_backgrounds,
        "true_channel_scales": true_channel_scales,
        "generation_k": generation_k,
        "measurement_duration_seconds": measurement_duration_seconds,
        "background_mean_per_second": background_mean_per_second,
        "background_mean": background_mean,
        "background_channel_std_per_second": background_channel_std_per_second,
        "background_channel_std": background_channel_std,
        "channel_scale_mean": channel_scale_mean,
        "channel_scale_std": channel_scale_std,
        "peak_total_source_signal_per_second": peak_total_source_signal_per_second,
        "peak_total_source_signal": peak_total_source_signal,
        "trajectory_mode": trajectory_mode,
    }
