"""
Forward measurement model for the moving gamma detector.
"""

import numpy as np

from geometry import EPS


def directional_gain(cos_theta, k):
    """
    Very simple angular sensitivity model.

    cos_theta <= 0 means the source is behind the diode and the response is 0.
    """
    return np.maximum(0.0, cos_theta) ** k


def compute_detector_response(
    source_position,
    geometry,
    rotations,
    translations,
    channel_scales,
    k,
):
    """
    Compute the geometric part of the measurement model.

    Returned array has shape (num_measurements, num_channels) and contains:
        alpha_c * g(theta_ic) / r_ic^2
    """
    diode_positions_world = (
        np.einsum("mij,cj->mci", rotations, geometry["diode_positions_local"])
        + translations[:, None, :]
    )
    diode_normals_world = np.einsum("mij,cj->mci", rotations, geometry["diode_normals_local"])

    vector_to_source = source_position[None, None, :] - diode_positions_world
    distances = np.linalg.norm(vector_to_source, axis=2)
    direction_to_source = vector_to_source / np.maximum(distances[:, :, None], EPS)

    cos_theta = np.sum(direction_to_source * diode_normals_world, axis=2)
    gain = directional_gain(cos_theta, k)

    return channel_scales[None, :] * gain / np.maximum(distances * distances, EPS)


def expected_counts(
    source_position,
    source_strength,
    geometry,
    rotations,
    translations,
    channel_scales,
    backgrounds,
    k,
):
    """Full forward model: background + source_strength * response."""
    response = compute_detector_response(
        source_position=source_position,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        channel_scales=channel_scales,
        k=k,
    )
    return backgrounds[None, :] + source_strength * response
