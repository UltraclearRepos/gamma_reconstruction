"""
Forward measurement model for the moving gamma detector.
"""

import numpy as np

from common.geometry import EPS


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
    distance_power=2.0,
):
    """
    Compute the geometric part of the measurement model.

    Returned array has shape (num_measurements, num_channels) and contains:
        alpha_c * g(theta_ic) / r_ic^distance_power
    """
    num_measurements = len(translations)
    num_channels = len(geometry["diode_positions_local"])
    distance_power = float(distance_power)
    if distance_power < 0.0:
        raise ValueError("distance_power must be >= 0.")
    response = np.zeros((num_measurements, num_channels), dtype=float)

    for measurement_index in range(num_measurements):
        rotation = rotations[measurement_index]
        detector_position = translations[measurement_index]

        for channel_index in range(num_channels):
            diode_position_local = geometry["diode_positions_local"][channel_index]
            diode_normal_local = geometry["diode_normals_local"][channel_index]

            diode_position_world = rotation @ diode_position_local + detector_position
            diode_normal_world = rotation @ diode_normal_local

            vector_to_source = source_position - diode_position_world
            distance = np.linalg.norm(vector_to_source)
            direction_to_source = vector_to_source / max(distance, EPS)

            cos_theta = float(np.dot(direction_to_source, diode_normal_world))
            gain = directional_gain(cos_theta, k)

            response[measurement_index, channel_index] = (
                channel_scales[channel_index]
                * gain
                / max(distance ** distance_power, EPS)
            )

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
    distance_power=2.0,
):
    """Full forward model: background + source_strength * response."""
    response = compute_detector_response(
        source_position=source_position,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        channel_scales=channel_scales,
        k=k,
        distance_power=distance_power,
    )
    return backgrounds[None, :] + source_strength * response
