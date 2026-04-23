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
    num_measurements=30,
    seed=11,
    orientation_mode="fixed",
    fixed_view_direction=(0.0, 1.0, 0.0),
):
    """
    Generate one synthetic multi-source experiment.

    orientation_mode:
        - "fixed": keep detector pointing in one direction (default).
        - "towards_target": re-aim detector towards scan_center each measurement.

    fixed_view_direction:
        3D vector for "fixed" mode, e.g. (0, 1, 0) or (1, 0, 0).

    The output keeps true_sources as a list because multi-point algorithms may
    reconstruct a map rather than one estimated position.
    """
    rng = np.random.default_rng(seed)

    geometry = make_detector_geometry(num_channels=16, diode_pitch=0.006)
    num_channels = len(geometry["diode_positions_local"])

    true_sources = [
        {"position": np.array([0.18, -0.12, 0.26], dtype=float), "relative_strength": 1.0},
        {"position": np.array([-0.10, -0.18, 0.16], dtype=float), "relative_strength": 0.65},
        {"position": np.array([0.34, -0.07, 0.38], dtype=float), "relative_strength": 0.45},
    ]
    scan_center = np.mean([source["position"] for source in true_sources], axis=0)

    background_mean = 1000.0
    background_channel_std = 50.0
    peak_total_source_signal = 5000.0

    true_backgrounds = rng.normal(
        loc=background_mean,
        scale=background_channel_std,
        size=num_channels,
    )
    true_channel_scales = rng.normal(loc=1.0, scale=0.08, size=num_channels)
    generation_k = 4.5

    rotations, translations = make_detector_trajectory(
        geometry=geometry,
        source_position=scan_center,
        num_measurements=num_measurements,
        rng=rng,
        angular_jitter_deg=3.0,
        lateral_span=0.60,
        vertical_span=0.36,
        orientation_mode=orientation_mode,
        fixed_view_direction=fixed_view_direction,
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
        "background_mean": background_mean,
        "peak_total_source_signal": peak_total_source_signal,
    }
