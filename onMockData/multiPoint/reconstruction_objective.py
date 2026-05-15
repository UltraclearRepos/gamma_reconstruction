"""
Objective/model helpers for multi-point source reconstruction.
"""

import numpy as np

from common.forward_model import compute_detector_response
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
        )
        system_matrix[:, voxel_index] = response.ravel()

    return system_matrix


def predict_counts_from_activity(backgrounds_flat, system_matrix, activity):
    """Predict flattened detector counts from background and voxel activity."""
    # H @ activity sums contributions from all voxels for every measurement/channel row.
    return backgrounds_flat + system_matrix @ activity
