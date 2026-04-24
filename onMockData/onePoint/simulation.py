"""
Synthetic data generation for the point-source reconstruction demo.
"""

import numpy as np

from common.forward_model import compute_detector_response, expected_counts
from common.geometry import make_detector_geometry, make_detector_trajectory


def generate_synthetic_dataset(
    num_measurements,
    seed,
    orientation_mode,
    fixed_view_direction,
    true_source_position,
    num_channels,
    diode_pitch,
    background_mean_per_second,
    background_channel_std_per_second,
    channel_scale_mean,
    channel_scale_std,
    generation_k,
    peak_source_signal_per_second,
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
    Generate one synthetic experiment.

    orientation_mode:
        - "fixed": keep detector pointing in one direction (default).
        - "towards_target": re-aim detector towards scan_center each measurement.

    trajectory_mode:
        - "grid_scan": one-sided left/right, up/down scan.
        - "around_center": detector positions orbit around scan_center.

    fixed_view_direction:
        3D vector for "fixed" mode, e.g. (0, 1, 0) or (1, 0, 0).

    background_mean_per_second and peak_source_signal_per_second are rates.
    They are multiplied by measurement_duration_seconds to generate counts
    for one measurement.
    """
    rng = np.random.default_rng(seed)

    geometry = make_detector_geometry(num_channels=num_channels, diode_pitch=diode_pitch)
    num_channels = len(geometry["diode_positions_local"])
    true_source_position = np.array(true_source_position, dtype=float)
    if scan_center is None:
        raise ValueError("scan_center must be provided explicitly, e.g. center of reconstruction search volume.")
    scan_center = np.array(scan_center, dtype=float)
    measurement_duration_seconds = float(measurement_duration_seconds)
    if measurement_duration_seconds <= 0.0:
        raise ValueError("measurement_duration_seconds must be > 0.")

    background_mean = float(background_mean_per_second) * measurement_duration_seconds
    background_channel_std = float(background_channel_std_per_second) * measurement_duration_seconds
    peak_source_signal = float(peak_source_signal_per_second) * measurement_duration_seconds

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

    unit_response = compute_detector_response(
        source_position=true_source_position,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        channel_scales=true_channel_scales,
        k=generation_k,
    )
    strongest_measurement_response = max(float(np.max(np.mean(unit_response, axis=1))), 1e-12)
    true_source_strength = peak_source_signal / strongest_measurement_response

    mean_counts = expected_counts(
        source_position=true_source_position,
        source_strength=true_source_strength,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        channel_scales=true_channel_scales,
        backgrounds=true_backgrounds,
        k=generation_k,
    )

    # Photon counts are naturally modeled with a Poisson distribution.
    noisy_counts = rng.poisson(mean_counts).astype(float)

    return {
        "counts": noisy_counts,
        "rotations": rotations,
        "translations": translations,
        "geometry": geometry,
        "true_source_position": true_source_position,
        "true_source_strength": true_source_strength,
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
        "peak_source_signal_per_second": peak_source_signal_per_second,
        "peak_source_signal": peak_source_signal,
        "trajectory_mode": trajectory_mode,
    }
