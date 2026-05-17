"""
Forward measurement model for the moving gamma detector.
"""

import numpy as np

from common.detector_response_components import (
    angular_response,
    channel_surface_offsets,
    empirical_distance_factor,
    skin_attenuation,
    solid_angle_fraction,
)
from common.geometry import EPS


def directional_gain(cos_theta, k):
    """Backward-compatible wrapper for the old cos(theta)^k angular model."""
    return angular_response(cos_theta=cos_theta, k=k)


def compute_detector_response(
    source_position,
    geometry,
    rotations,
    translations,
    channel_scales,
    k,
    distance_power=2.0,
    angular_peak_angle_deg=0.0,
    angular_width_deg=None,
    skin_thickness_cm=0.0,
    skin_mu_per_cm=0.0,
    detector_intrinsic_efficiency=1.0,
    response_scale=1.0,
):
    """
    Compute the geometric part of the measurement model.

    Returned array has shape (num_measurements, num_channels) and contains:
        alpha_c * solid_angle * angular_response * skin_attenuation

    The finite channel surface is integrated numerically. This is important
    when the detector is close to the source/skin and the point-channel
    approximation is no longer reliable.
    """
    num_measurements = len(translations)
    num_channels = len(geometry["diode_positions_local"])
    distance_power = float(distance_power)
    if distance_power < 0.0:
        raise ValueError("distance_power must be >= 0.")
    detector_intrinsic_efficiency = float(detector_intrinsic_efficiency)
    if detector_intrinsic_efficiency < 0.0:
        raise ValueError("detector_intrinsic_efficiency must be >= 0.")
    response_scale = float(response_scale)
    if response_scale < 0.0:
        raise ValueError("response_scale must be >= 0.")

    surface_offsets_local, patch_area = channel_surface_offsets(geometry)
    response = np.zeros((num_measurements, num_channels), dtype=float)

    for measurement_index in range(num_measurements):
        rotation = rotations[measurement_index]
        detector_position = translations[measurement_index]

        for channel_index in range(num_channels):
            diode_position_local = geometry["diode_positions_local"][channel_index]
            diode_normal_local = geometry["diode_normals_local"][channel_index]

            diode_normal_world = rotation @ diode_normal_local

            channel_response = 0.0
            for surface_offset_local in surface_offsets_local:
                surface_point_local = diode_position_local + surface_offset_local
                surface_point_world = rotation @ surface_point_local + detector_position

                vector_to_source = source_position - surface_point_world
                distance = np.linalg.norm(vector_to_source)
                direction_to_source = vector_to_source / max(distance, EPS)

                cos_theta = float(np.dot(direction_to_source, diode_normal_world))
                if cos_theta <= 0.0:
                    continue

                solid_angle_patch = solid_angle_fraction(
                    patch_area=patch_area,
                    cos_theta=cos_theta,
                    distance=distance,
                )
                angle_gain = angular_response(
                    cos_theta=cos_theta,
                    k=k,
                    angular_peak_angle_deg=angular_peak_angle_deg,
                    angular_width_deg=angular_width_deg,
                )
                attenuation = skin_attenuation(
                    cos_theta=cos_theta,
                    skin_thickness_cm=skin_thickness_cm,
                    skin_mu_per_cm=skin_mu_per_cm,
                )
                empirical_distance = empirical_distance_factor(
                    distance=distance,
                    distance_power=distance_power,
                )
                channel_response += (
                    solid_angle_patch
                    * angle_gain
                    * attenuation
                    * empirical_distance
                )

            response[measurement_index, channel_index] = (
                channel_scales[channel_index]
                * detector_intrinsic_efficiency
                * response_scale
                * channel_response
            )

    return response


def compute_detector_response_diagnostics(
    source_position,
    geometry,
    rotations,
    translations,
    channel_scales,
    k,
    distance_power=2.0,
    angular_peak_angle_deg=0.0,
    angular_width_deg=None,
    skin_thickness_cm=0.0,
    skin_mu_per_cm=0.0,
    detector_intrinsic_efficiency=1.0,
    response_scale=1.0,
):
    """
    Compute response and intermediate component maps for one source position.

    Returned arrays have shape (num_measurements, num_channels). Component
    averages are weighted by the geometric solid angle of channel surface
    patches, so they describe what the detector effectively "sees".
    """
    num_measurements = len(translations)
    num_channels = len(geometry["diode_positions_local"])
    distance_power = float(distance_power)
    if distance_power < 0.0:
        raise ValueError("distance_power must be >= 0.")
    detector_intrinsic_efficiency = float(detector_intrinsic_efficiency)
    if detector_intrinsic_efficiency < 0.0:
        raise ValueError("detector_intrinsic_efficiency must be >= 0.")
    response_scale = float(response_scale)
    if response_scale < 0.0:
        raise ValueError("response_scale must be >= 0.")

    surface_offsets_local, patch_area = channel_surface_offsets(geometry)
    shape = (num_measurements, num_channels)
    response = np.zeros(shape, dtype=float)
    solid_angle = np.zeros(shape, dtype=float)
    angular = np.zeros(shape, dtype=float)
    attenuation = np.ones(shape, dtype=float)
    distance_factor = np.ones(shape, dtype=float)
    distance_cm = np.full(shape, np.nan, dtype=float)
    theta_deg = np.full(shape, np.nan, dtype=float)
    cos_theta_map = np.zeros(shape, dtype=float)

    for measurement_index in range(num_measurements):
        rotation = rotations[measurement_index]
        detector_position = translations[measurement_index]

        for channel_index in range(num_channels):
            diode_position_local = geometry["diode_positions_local"][channel_index]
            diode_normal_local = geometry["diode_normals_local"][channel_index]
            diode_normal_world = rotation @ diode_normal_local

            response_sum = 0.0
            solid_sum = 0.0
            angular_weighted = 0.0
            attenuation_weighted = 0.0
            distance_factor_weighted = 0.0
            distance_weighted = 0.0
            theta_weighted = 0.0
            cos_weighted = 0.0

            for surface_offset_local in surface_offsets_local:
                surface_point_local = diode_position_local + surface_offset_local
                surface_point_world = rotation @ surface_point_local + detector_position

                vector_to_source = source_position - surface_point_world
                distance = np.linalg.norm(vector_to_source)
                direction_to_source = vector_to_source / max(distance, EPS)

                cos_theta = float(np.dot(direction_to_source, diode_normal_world))
                if cos_theta <= 0.0:
                    continue

                solid_angle_patch = solid_angle_fraction(
                    patch_area=patch_area,
                    cos_theta=cos_theta,
                    distance=distance,
                )
                angle_gain = angular_response(
                    cos_theta=cos_theta,
                    k=k,
                    angular_peak_angle_deg=angular_peak_angle_deg,
                    angular_width_deg=angular_width_deg,
                )
                attenuation_gain = skin_attenuation(
                    cos_theta=cos_theta,
                    skin_thickness_cm=skin_thickness_cm,
                    skin_mu_per_cm=skin_mu_per_cm,
                )
                empirical_distance = empirical_distance_factor(
                    distance=distance,
                    distance_power=distance_power,
                )
                theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

                response_sum += (
                    solid_angle_patch
                    * angle_gain
                    * attenuation_gain
                    * empirical_distance
                )
                solid_sum += solid_angle_patch
                angular_weighted += solid_angle_patch * angle_gain
                attenuation_weighted += solid_angle_patch * attenuation_gain
                distance_factor_weighted += solid_angle_patch * empirical_distance
                distance_weighted += solid_angle_patch * distance
                theta_weighted += solid_angle_patch * theta
                cos_weighted += solid_angle_patch * cos_theta

            response[measurement_index, channel_index] = (
                channel_scales[channel_index]
                * detector_intrinsic_efficiency
                * response_scale
                * response_sum
            )
            solid_angle[measurement_index, channel_index] = solid_sum
            if solid_sum > EPS:
                angular[measurement_index, channel_index] = angular_weighted / solid_sum
                attenuation[measurement_index, channel_index] = attenuation_weighted / solid_sum
                distance_factor[measurement_index, channel_index] = distance_factor_weighted / solid_sum
                distance_cm[measurement_index, channel_index] = distance_weighted / solid_sum
                theta_deg[measurement_index, channel_index] = theta_weighted / solid_sum
                cos_theta_map[measurement_index, channel_index] = cos_weighted / solid_sum

    return {
        "response": response,
        "solid_angle": solid_angle,
        "angular_response": angular,
        "skin_attenuation": attenuation,
        "distance_factor": distance_factor,
        "distance_cm": distance_cm,
        "theta_deg": theta_deg,
        "cos_theta": cos_theta_map,
    }


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
    angular_peak_angle_deg=0.0,
    angular_width_deg=None,
    skin_thickness_cm=0.0,
    skin_mu_per_cm=0.0,
    detector_intrinsic_efficiency=1.0,
    response_scale=1.0,
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
        angular_peak_angle_deg=angular_peak_angle_deg,
        angular_width_deg=angular_width_deg,
        skin_thickness_cm=skin_thickness_cm,
        skin_mu_per_cm=skin_mu_per_cm,
        detector_intrinsic_efficiency=detector_intrinsic_efficiency,
        response_scale=response_scale,
    )
    return backgrounds[None, :] + source_strength * response
