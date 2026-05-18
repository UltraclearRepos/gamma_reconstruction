"""
Objective/model helpers for multi-point source reconstruction.
"""

import numpy as np
import torch

from common.forward_model import compute_detector_response, compute_detector_response_torch
from common.geometry import EPS


def poisson_negative_log_likelihood(counts, mean_counts):
    """
    Poisson negative log-likelihood without the constant log(y!) term.
    """
    mean_counts = np.maximum(mean_counts, EPS)
    return float(np.sum(mean_counts - counts * np.log(mean_counts)))


def build_system_matrix(
    candidate_positions,
    geometry,
    rotations,
    translations,
    assumed_channel_scales,
    assumed_k,
    assumed_distance_power,
    assumed_angular_peak_angle_deg=0.0,
    assumed_angular_width_deg=None,
    assumed_skin_thickness_cm=0.0,
    assumed_skin_mu_per_cm=0.0,
    assumed_detector_intrinsic_efficiency=1.0,
    assumed_response_scale=1.0,
):
    """
    Build H matrix for voxel reconstruction.

    H[row, voxel] says how much one unit of activity in a candidate voxel
    contributes to one flattened detector count.
    """
    num_rows = len(translations) * len(assumed_channel_scales)
    num_voxels = len(candidate_positions)
    # Rows are flattened detector observations: measurement 0/channel 0, measurement 0/channel 1, ...
    # Columns are candidate voxels. Each column is the detector signature of one unit source.
    system_matrix = np.zeros((num_rows, num_voxels), dtype=float)

    for voxel_index, candidate_position in enumerate(candidate_positions):
        # Simulate what all detector positions/channels would measure if this voxel had activity 1.
        response = compute_detector_response(
            source_position=candidate_position,
            geometry=geometry,
            rotations=rotations,
            translations=translations,
            channel_scales=assumed_channel_scales,
            k=assumed_k,
            distance_power=assumed_distance_power,
            angular_peak_angle_deg=assumed_angular_peak_angle_deg,
            angular_width_deg=assumed_angular_width_deg,
            skin_thickness_cm=assumed_skin_thickness_cm,
            skin_mu_per_cm=assumed_skin_mu_per_cm,
            detector_intrinsic_efficiency=assumed_detector_intrinsic_efficiency,
            response_scale=assumed_response_scale,
        )
        system_matrix[:, voxel_index] = response.ravel()

    return system_matrix


def build_system_matrix_torch(
    candidate_positions,
    geometry,
    rotations,
    translations,
    assumed_channel_scales,
    assumed_k,
    assumed_distance_power,
    assumed_angular_peak_angle_deg=0.0,
    assumed_angular_width_deg=None,
    assumed_skin_thickness_cm=0.0,
    assumed_skin_mu_per_cm=0.0,
    assumed_detector_intrinsic_efficiency=1.0,
    assumed_response_scale=1.0,
    device=None,
    dtype=torch.float32,
    voxel_chunk_size=4096,
):
    """
    Build H matrix with Torch tensor operations.

    The CPU build_system_matrix above is kept as the simple reference version.
    This function mirrors the CPU structure and delegates the physical response
    model to compute_detector_response_torch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate_positions = torch.as_tensor(candidate_positions, dtype=dtype, device=device)
    num_measurements = len(translations)
    num_channels = len(assumed_channel_scales)
    num_voxels = int(candidate_positions.shape[0])
    system_matrix = torch.empty(
        (num_measurements * num_channels, num_voxels),
        dtype=dtype,
        device=device,
    )

    for start in range(0, num_voxels, int(voxel_chunk_size)):
        stop = min(start + int(voxel_chunk_size), num_voxels)
        response = compute_detector_response_torch(
            source_position=candidate_positions[start:stop],
            geometry=geometry,
            rotations=rotations,
            translations=translations,
            channel_scales=assumed_channel_scales,
            k=assumed_k,
            distance_power=assumed_distance_power,
            angular_peak_angle_deg=assumed_angular_peak_angle_deg,
            angular_width_deg=assumed_angular_width_deg,
            skin_thickness_cm=assumed_skin_thickness_cm,
            skin_mu_per_cm=assumed_skin_mu_per_cm,
            detector_intrinsic_efficiency=assumed_detector_intrinsic_efficiency,
            response_scale=assumed_response_scale,
            device=device,
            dtype=dtype,
        )
        system_matrix[:, start:stop] = response.permute(1, 2, 0).reshape(
            num_measurements * num_channels,
            stop - start,
        )

    return system_matrix


def predict_counts_from_activity(backgrounds_flat, system_matrix, activity):
    """Predict flattened detector counts from background and voxel activity."""
    # H @ activity sums contributions from all voxels for every measurement/channel row.
    return backgrounds_flat + system_matrix @ activity
