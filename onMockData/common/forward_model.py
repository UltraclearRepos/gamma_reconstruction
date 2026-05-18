"""
Forward measurement model for the moving gamma detector.
"""

import math

import numpy as np
import torch

from common.detector_response_components import (
    angular_response,
    channel_surface_offsets,
    collimator_response,
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
    collimator_half_angle_deg=None,
    collimator_edge_width_deg=3.0,
    collimator_leakage=0.0,
    skin_thickness_cm=0.0,
    skin_mu_per_cm=0.0,
    detector_intrinsic_efficiency=1.0,
    response_scale=1.0,
):
    """
    Compute the geometric part of the measurement model.

    Returned array has shape (num_measurements, num_channels) and contains:
        alpha_c * solid_angle * angular_response * collimator_response * skin_attenuation

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
                collimator_gain = collimator_response(
                    cos_theta=cos_theta,
                    collimator_half_angle_deg=collimator_half_angle_deg,
                    collimator_edge_width_deg=collimator_edge_width_deg,
                    collimator_leakage=collimator_leakage,
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
                    * collimator_gain
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


def compute_detector_response_torch(
    source_position,
    geometry,
    rotations,
    translations,
    channel_scales,
    k,
    distance_power=2.0,
    angular_peak_angle_deg=0.0,
    angular_width_deg=None,
    collimator_half_angle_deg=None,
    collimator_edge_width_deg=3.0,
    collimator_leakage=0.0,
    skin_thickness_cm=0.0,
    skin_mu_per_cm=0.0,
    detector_intrinsic_efficiency=1.0,
    response_scale=1.0,
    device=None,
    dtype=torch.float32,
):
    """
    Torch/GPU version of compute_detector_response.

    source_position can be one position with shape (3,) or a chunk of positions
    with shape (num_sources, 3). The returned tensor has shape
    (num_measurements, num_channels) for one source, or
    (num_sources, num_measurements, num_channels) for a source chunk.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    distance_power = float(distance_power)
    if distance_power < 0.0:
        raise ValueError("distance_power must be >= 0.")
    detector_intrinsic_efficiency = float(detector_intrinsic_efficiency)
    if detector_intrinsic_efficiency < 0.0:
        raise ValueError("detector_intrinsic_efficiency must be >= 0.")
    response_scale = float(response_scale)
    if response_scale < 0.0:
        raise ValueError("response_scale must be >= 0.")

    source_positions = torch.as_tensor(source_position, dtype=dtype, device=device)
    single_source = source_positions.ndim == 1
    if single_source:
        source_positions = source_positions[None, :]

    rotations = torch.as_tensor(rotations, dtype=dtype, device=device)
    translations = torch.as_tensor(translations, dtype=dtype, device=device)
    channel_scales = torch.as_tensor(channel_scales, dtype=dtype, device=device)
    diode_positions_local = torch.as_tensor(
        geometry["diode_positions_local"],
        dtype=dtype,
        device=device,
    )
    diode_normals_local = torch.as_tensor(
        geometry["diode_normals_local"],
        dtype=dtype,
        device=device,
    )
    surface_offsets_local, patch_area = channel_surface_offsets(geometry)
    surface_offsets_local = torch.as_tensor(surface_offsets_local, dtype=dtype, device=device)
    patch_area = torch.as_tensor(float(patch_area), dtype=dtype, device=device)

    surface_points_local = diode_positions_local[:, None, :] + surface_offsets_local[None, :, :]
    surface_points_world = (
        torch.einsum("mij,csj->mcsi", rotations, surface_points_local)
        + translations[:, None, None, :]
    )
    diode_normals_world = torch.einsum("mij,cj->mci", rotations, diode_normals_local)

    vector_to_source = (
        source_positions[:, None, None, None, :]
        - surface_points_world[None, :, :, :, :]
    )
    distance = torch.linalg.norm(vector_to_source, dim=-1)
    direction_to_source = vector_to_source / torch.clamp(distance[..., None], min=EPS)
    cos_theta = torch.sum(
        direction_to_source * diode_normals_world[None, :, :, None, :],
        dim=-1,
    )
    positive_cos = torch.clamp(cos_theta, min=0.0)

    solid_angle_patch = (
        patch_area
        * positive_cos
        / torch.clamp(4.0 * math.pi * distance * distance, min=EPS)
    )

    if angular_width_deg is None:
        angular_gain = positive_cos ** float(k)
    else:
        theta_deg = torch.rad2deg(torch.acos(torch.clamp(cos_theta, min=-1.0, max=1.0)))
        peak = float(angular_peak_angle_deg)
        width = max(float(angular_width_deg), EPS)
        angular_gain = torch.exp(-0.5 * ((theta_deg - peak) / width) ** 2)

    if collimator_half_angle_deg is None:
        collimator_gain = 1.0
    else:
        if "theta_deg" not in locals():
            theta_deg = torch.rad2deg(torch.acos(torch.clamp(cos_theta, min=-1.0, max=1.0)))
        half_angle = float(collimator_half_angle_deg)
        edge_width = max(float(collimator_edge_width_deg), EPS)
        leakage = float(collimator_leakage)
        if half_angle < 0.0:
            raise ValueError("collimator_half_angle_deg must be >= 0.")
        if not 0.0 <= leakage < 1.0:
            raise ValueError("collimator_leakage must be in the range [0, 1).")
        raw_collimator = torch.sigmoid((half_angle - theta_deg) / edge_width)
        on_axis = float(1.0 / (1.0 + math.exp(-half_angle / edge_width)))
        normalized_collimator = torch.clamp(raw_collimator / max(on_axis, EPS), max=1.0)
        collimator_gain = leakage + (1.0 - leakage) * normalized_collimator

    thickness = float(skin_thickness_cm)
    mu = float(skin_mu_per_cm)
    if thickness > 0.0 and mu > 0.0:
        skin_path_length = thickness / torch.clamp(cos_theta, min=0.05)
        attenuation = torch.exp(-mu * skin_path_length)
    else:
        attenuation = 1.0

    if distance_power > 0.0:
        empirical_distance = 1.0 / torch.clamp(distance ** distance_power, min=EPS)
    else:
        empirical_distance = 1.0

    response = torch.sum(
        solid_angle_patch * angular_gain * collimator_gain * attenuation * empirical_distance,
        dim=3,
    )
    response = (
        response
        * channel_scales[None, None, :]
        * detector_intrinsic_efficiency
        * response_scale
    )

    if single_source:
        return response[0]
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
    collimator_half_angle_deg=None,
    collimator_edge_width_deg=3.0,
    collimator_leakage=0.0,
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
    collimator = np.ones(shape, dtype=float)
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
            collimator_weighted = 0.0
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
                collimator_gain = collimator_response(
                    cos_theta=cos_theta,
                    collimator_half_angle_deg=collimator_half_angle_deg,
                    collimator_edge_width_deg=collimator_edge_width_deg,
                    collimator_leakage=collimator_leakage,
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
                    * collimator_gain
                    * attenuation_gain
                    * empirical_distance
                )
                solid_sum += solid_angle_patch
                angular_weighted += solid_angle_patch * angle_gain
                collimator_weighted += solid_angle_patch * collimator_gain
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
                collimator[measurement_index, channel_index] = collimator_weighted / solid_sum
                attenuation[measurement_index, channel_index] = attenuation_weighted / solid_sum
                distance_factor[measurement_index, channel_index] = distance_factor_weighted / solid_sum
                distance_cm[measurement_index, channel_index] = distance_weighted / solid_sum
                theta_deg[measurement_index, channel_index] = theta_weighted / solid_sum
                cos_theta_map[measurement_index, channel_index] = cos_weighted / solid_sum

    return {
        "response": response,
        "solid_angle": solid_angle,
        "angular_response": angular,
        "collimator_response": collimator,
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
    collimator_half_angle_deg=None,
    collimator_edge_width_deg=3.0,
    collimator_leakage=0.0,
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
        collimator_half_angle_deg=collimator_half_angle_deg,
        collimator_edge_width_deg=collimator_edge_width_deg,
        collimator_leakage=collimator_leakage,
        skin_thickness_cm=skin_thickness_cm,
        skin_mu_per_cm=skin_mu_per_cm,
        detector_intrinsic_efficiency=detector_intrinsic_efficiency,
        response_scale=response_scale,
    )
    return backgrounds[None, :] + source_strength * response
