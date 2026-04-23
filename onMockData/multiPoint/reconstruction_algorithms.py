"""
Reconstruction algorithms for multi-point source experiments.
"""

import numpy as np

from common.geometry import EPS
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
    search_min,
    search_max,
    method="mlem",
    grid_size=11,
    num_iterations=40,
    history_interval=5,
    max_sources=3,
    min_source_distance=0.08,
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
            search_min=search_min,
            search_max=search_max,
            grid_size=grid_size,
            num_iterations=num_iterations,
            history_interval=history_interval,
            max_sources=max_sources,
            min_source_distance=min_source_distance,
        )

    raise ValueError(f"Unknown multi-source reconstruction method '{method}'. Use: mlem.")


def make_candidate_grid(search_min, search_max, grid_size):
    """Create candidate source positions on a regular XYZ grid."""
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

    return np.array(candidate_positions, dtype=float)


def pick_activity_peaks(candidate_positions, activity, max_sources=3, min_distance=0.08):
    """
    Pick strongest separated activity peaks from a voxel activity map.

    This is only a simple peak picker for summarizing ML-EM output.
    """
    estimated_sources = []
    sorted_indices = np.argsort(activity)[::-1]

    for voxel_index in sorted_indices:
        if activity[voxel_index] <= 0.0:
            break

        position = candidate_positions[voxel_index]
        far_enough = True
        for source in estimated_sources:
            distance = np.linalg.norm(position - source["position"])
            if distance < min_distance:
                far_enough = False
                break

        if far_enough:
            estimated_sources.append(
                {
                    "position": position.copy(),
                    "activity": float(activity[voxel_index]),
                    "voxel_index": int(voxel_index),
                }
            )

        if len(estimated_sources) >= max_sources:
            break

    return estimated_sources


def mlem_reconstruction(
    counts,
    geometry,
    rotations,
    translations,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_k,
    search_min,
    search_max,
    grid_size=11,
    num_iterations=40,
    history_interval=5,
    max_sources=3,
    min_source_distance=0.08,
):
    """
    Reconstruct a voxel activity map with ML-EM.

    Model:
        counts ~= background + H @ activity

    The returned estimated_sources are only a compact summary of the strongest
    reconstructed voxels. The main ML-EM result is the full activity map.
    """
    candidate_positions = make_candidate_grid(
        search_min=search_min,
        search_max=search_max,
        grid_size=grid_size,
    )
    system_matrix = build_system_matrix(
        candidate_positions=candidate_positions,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        assumed_channel_scales=assumed_channel_scales,
        assumed_k=assumed_k,
    )

    counts_flat = counts.ravel()
    backgrounds_flat = np.broadcast_to(assumed_backgrounds, counts.shape).ravel()
    sensitivity = np.sum(system_matrix, axis=0)

    initial_signal = np.maximum(counts_flat - backgrounds_flat, 0.0)
    initial_activity = np.sum(initial_signal) / max(float(np.sum(sensitivity)), EPS)
    activity = np.full(len(candidate_positions), max(initial_activity, EPS), dtype=float)

    history = []
    for iteration in range(1, num_iterations + 1):
        predicted_counts = predict_counts_from_activity(
            backgrounds_flat=backgrounds_flat,
            system_matrix=system_matrix,
            activity=activity,
        )
        ratio = counts_flat / np.maximum(predicted_counts, EPS)
        correction = (system_matrix.T @ ratio) / np.maximum(sensitivity, EPS)
        activity *= correction

        if iteration == 1 or iteration % history_interval == 0 or iteration == num_iterations:
            predicted_counts = predict_counts_from_activity(
                backgrounds_flat=backgrounds_flat,
                system_matrix=system_matrix,
                activity=activity,
            )
            best_index = int(np.argmax(activity))
            history.append(
                {
                    "iteration": float(iteration),
                    "score": poisson_negative_log_likelihood(counts_flat, predicted_counts),
                    "total_activity": float(np.sum(activity)),
                    "max_activity": float(activity[best_index]),
                    "best_position": candidate_positions[best_index].copy(),
                }
            )

    estimated_sources = pick_activity_peaks(
        candidate_positions=candidate_positions,
        activity=activity,
        max_sources=max_sources,
        min_distance=min_source_distance,
    )

    return {
        "candidate_positions": candidate_positions,
        "activity": activity,
        "estimated_sources": estimated_sources,
        "history": history,
    }
