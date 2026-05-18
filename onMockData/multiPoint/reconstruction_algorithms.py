"""
Reconstruction algorithms for multi-point source experiments.
"""

import numpy as np

from common.geometry import EPS, points_in_detector_box
from reconstruction_objective import (
    build_system_matrix,
    poisson_negative_log_likelihood,
    predict_counts_from_activity,
)


def reconstruct_sources(
    counts,
    geometry,
    rotations,
    translations,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_k,
    assumed_distance_power,
    sparsity_lambda,
    search_min,
    search_max,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
    assumed_angular_peak_angle_deg=0.0,
    assumed_angular_width_deg=None,
    assumed_skin_thickness_cm=0.0,
    assumed_skin_mu_per_cm=0.0,
    assumed_detector_intrinsic_efficiency=1.0,
    assumed_response_scale=1.0,
    method="mlem",
    grid_size=11,
    num_iterations=40,
    history_interval=5,
):
    """
    Run one multi-source reconstruction method using a plain if/elif dispatcher.
    """
    method = method.strip().lower().replace("-", "_")

    if method in ("mlem", "ml_em"):
        return mlem_reconstruction(
            counts=counts,
            geometry=geometry,
            rotations=rotations,
            translations=translations,
            assumed_backgrounds=assumed_backgrounds,
            assumed_channel_scales=assumed_channel_scales,
            assumed_k=assumed_k,
            assumed_distance_power=assumed_distance_power,
            assumed_angular_peak_angle_deg=assumed_angular_peak_angle_deg,
            assumed_angular_width_deg=assumed_angular_width_deg,
            assumed_skin_thickness_cm=assumed_skin_thickness_cm,
            assumed_skin_mu_per_cm=assumed_skin_mu_per_cm,
            assumed_detector_intrinsic_efficiency=assumed_detector_intrinsic_efficiency,
            assumed_response_scale=assumed_response_scale,
            sparsity_lambda=sparsity_lambda,
            search_min=search_min,
            search_max=search_max,
            search_start_distance=search_start_distance,
            search_stop_distance=search_stop_distance,
            search_width=search_width,
            search_height=search_height,
            grid_size=grid_size,
            num_iterations=num_iterations,
            history_interval=history_interval,
        )

    raise ValueError(
        f"Unknown multi-source reconstruction method '{method}'. "
        "Use: mlem."
    )


def make_candidate_grid(
    search_min,
    search_max,
    grid_size,
    geometry,
    rotations,
    translations,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
):
    """Create candidate source positions inside the detector-facing rectangular boxes."""
    search_min = np.array(search_min, dtype=float)
    search_max = np.array(search_max, dtype=float)

    xs = np.linspace(search_min[0], search_max[0], grid_size)
    ys = np.linspace(search_min[1], search_max[1], grid_size)
    zs = np.linspace(search_min[2], search_max[2], grid_size)

    candidate_positions = []
    for x in xs:
        for y in ys:
            for z in zs:
                candidate_positions.append(np.array([x, y, z], dtype=float))

    candidate_positions = np.array(candidate_positions, dtype=float)
    inside = points_in_detector_box(
        points=candidate_positions,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        search_start_distance=search_start_distance,
        search_stop_distance=search_stop_distance,
        search_width=search_width,
        search_height=search_height,
    )
    candidate_positions = candidate_positions[inside]
    if len(candidate_positions) == 0:
        raise ValueError("No candidate voxels are inside the detector search boxes.")
    return candidate_positions


def mlem_reconstruction(
    counts,
    geometry,
    rotations,
    translations,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_k,
    assumed_distance_power,
    sparsity_lambda,
    search_min,
    search_max,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
    assumed_angular_peak_angle_deg=0.0,
    assumed_angular_width_deg=None,
    assumed_skin_thickness_cm=0.0,
    assumed_skin_mu_per_cm=0.0,
    assumed_detector_intrinsic_efficiency=1.0,
    assumed_response_scale=1.0,
    grid_size=11,
    num_iterations=40,
    history_interval=5,
):
    """
    Reconstruct a voxel activity map with ML-EM.

    Model:
        counts ~= background + H @ activity

    The main ML-EM result is the full activity map.
    """
    candidate_positions = make_candidate_grid(
        search_min=search_min,
        search_max=search_max,
        grid_size=grid_size,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        search_start_distance=search_start_distance,
        search_stop_distance=search_stop_distance,
        search_width=search_width,
        search_height=search_height,
    )
    system_matrix = build_system_matrix(         # system_matrix is H: rows are flattened measurement/channel pairs, columns are voxels.
        candidate_positions=candidate_positions, # H[row, voxel] says how much one unit of activity in that voxel would add to that detector count for specific measurement and specific channel.
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        assumed_channel_scales=assumed_channel_scales,
        assumed_k=assumed_k,
        assumed_distance_power=assumed_distance_power,
        assumed_angular_peak_angle_deg=assumed_angular_peak_angle_deg,
        assumed_angular_width_deg=assumed_angular_width_deg,
        assumed_skin_thickness_cm=assumed_skin_thickness_cm,
        assumed_skin_mu_per_cm=assumed_skin_mu_per_cm,
        assumed_detector_intrinsic_efficiency=assumed_detector_intrinsic_efficiency,
        assumed_response_scale=assumed_response_scale,
    )
    
    counts_flat = counts.ravel()
    backgrounds_flat = np.broadcast_to(assumed_backgrounds, counts.shape).ravel()
    
    sparsity_lambda = float(sparsity_lambda)
    if sparsity_lambda < 0.0:
        raise ValueError("sparsity_lambda must be >= 0.")

    sensitivity = np.sum(system_matrix, axis=0) # Sensitivity tells us how much each voxel contribute to detector response, not for each measurement, but in total

    initial_signal = np.maximum(counts_flat - backgrounds_flat, 0.0)
    initial_activity = np.sum(initial_signal) / max(float(np.sum(sensitivity)), EPS)
    
    activity = np.full(len(candidate_positions), max(initial_activity, EPS), dtype=float) # Start from a flat activity map: every candidate voxel gets the same positive activity. 
                                                                                          # ML-EM will then move activity toward voxels whose H columns best explain the counts.

    history = []
    activity_history = []
    for iteration in range(1, num_iterations + 1):
        
        predicted_counts = predict_counts_from_activity( # BACKGROUND + H @ ACTIVITY -> Based on each voxel's constibution to each measurement-channel pair and activity -> predicts counts for each measurement and each channel.
            backgrounds_flat=backgrounds_flat,
            system_matrix=system_matrix,
            activity=activity,
        )
        ratio = counts_flat / np.maximum(predicted_counts, EPS) # Ratio tells how much error we have in each measurement-channel pair.
        correction = (system_matrix.T @ ratio) / np.maximum(sensitivity + sparsity_lambda, EPS) # So crucial step. H tells us how much each voxel contribute to measurement-channel pair prediction and ratio tells us how bad prediction was.
                                                                              # H.T @ ratio gives each voxel a response-weighted sum of measurement errors. Dividing by sensitivity normalizes it by total voxel response,
                                                                              # so correction[voxel] becomes the H-weighted average ratio for that voxel, not just a larger sum for better-covered voxels.
                                                                              # sparsity_lambda adds an L1-like penalty that makes weak diffuse activity shrink faster.
        activity *= correction

        if iteration == 1 or iteration % history_interval == 0 or iteration == num_iterations:
            predicted_counts = predict_counts_from_activity(
                backgrounds_flat=backgrounds_flat,
                system_matrix=system_matrix,
                activity=activity,
            )
            history.append(
                {
                    "iteration": float(iteration),
                    "score": poisson_negative_log_likelihood(counts_flat, predicted_counts),
                    "total_activity": float(np.sum(activity)),
                    "max_activity": float(np.max(activity)),
                }
            )
            activity_history.append(activity.copy())

    return {
        "candidate_positions": candidate_positions,
        "activity": activity,
        "activity_history": [snapshot.copy() for snapshot in activity_history],
        "history": history,
    }
