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
    return max(0.0, cos_theta) ** k


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
    num_measurements = len(rotations)
    num_channels = len(channel_scales)
    response = np.zeros((num_measurements, num_channels), dtype=float)

    for i in range(num_measurements):
        rotation = rotations[i]
        translation = translations[i]

        for c in range(num_channels):
            diode_position_world = rotation @ geometry["diode_positions_local"][c] + translation
            diode_normal_world = rotation @ geometry["diode_normals_local"][c]

            vector_to_source = source_position - diode_position_world
            distance = np.linalg.norm(vector_to_source)
            direction_to_source = vector_to_source / max(distance, EPS)

            cos_theta = float(np.dot(direction_to_source, diode_normal_world))
            gain = directional_gain(cos_theta, k)

            response[i, c] = channel_scales[c] * gain / max(distance * distance, EPS)

    return response


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
