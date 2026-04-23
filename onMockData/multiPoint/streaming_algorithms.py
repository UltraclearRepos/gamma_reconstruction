"""
Streaming reconstruction algorithms for multi-point source experiments.
"""

import numpy as np

from common.geometry import EPS
from reconstruction_algorithms import make_candidate_grid, pick_activity_peaks
from reconstruction_objective import (
    build_system_matrix,
    poisson_negative_log_likelihood,
    predict_counts_from_activity,
)


def streaming_mlem_reconstruction(
    counts,
    geometry,
    rotations,
    translations,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_k,
    search_min,
    search_max,
    grid_size=9,
    inner_iterations_per_update=4,
    measurements_per_update=1,
    max_sources=3,
    min_source_distance=0.08,
    store_activity_history=True,
):
    """
    Run ML-EM in a streaming workflow.

    New measurements are added in chunks, and after each chunk we run a small
    number of ML-EM iterations starting from the previous activity estimate.
    """
    num_measurements, num_channels = counts.shape
    chunk_size = max(1, int(measurements_per_update))

    candidate_positions = make_candidate_grid(
        search_min=search_min,
        search_max=search_max,
        grid_size=grid_size,
    )

    response_blocks = []
    counts_blocks = []
    background_blocks = []
    activity = np.full(len(candidate_positions), EPS, dtype=float)
    stream_history = []
    activity_history = []

    for start_index in range(0, num_measurements, chunk_size):
        end_index = min(start_index + chunk_size, num_measurements)

        response_block = build_system_matrix(
            candidate_positions=candidate_positions,
            geometry=geometry,
            rotations=rotations[start_index:end_index],
            translations=translations[start_index:end_index],
            assumed_channel_scales=assumed_channel_scales,
            assumed_k=assumed_k,
        )
        counts_block = counts[start_index:end_index].ravel()
        backgrounds_block = np.broadcast_to(
            assumed_backgrounds,
            (end_index - start_index, num_channels),
        ).ravel()

        response_blocks.append(response_block)
        counts_blocks.append(counts_block)
        background_blocks.append(backgrounds_block)

        system_matrix = np.vstack(response_blocks)
        counts_flat = np.concatenate(counts_blocks)
        backgrounds_flat = np.concatenate(background_blocks)
        sensitivity = np.sum(system_matrix, axis=0)

        if start_index == 0:
            initial_signal = np.maximum(counts_flat - backgrounds_flat, 0.0)
            initial_activity = np.sum(initial_signal) / max(float(np.sum(sensitivity)), EPS)
            activity[:] = max(initial_activity, EPS)

        for _ in range(max(1, int(inner_iterations_per_update))):
            predicted_counts = predict_counts_from_activity(
                backgrounds_flat=backgrounds_flat,
                system_matrix=system_matrix,
                activity=activity,
            )
            ratio = counts_flat / np.maximum(predicted_counts, EPS)
            correction = (system_matrix.T @ ratio) / np.maximum(sensitivity, EPS)
            activity *= correction

        predicted_counts = predict_counts_from_activity(
            backgrounds_flat=backgrounds_flat,
            system_matrix=system_matrix,
            activity=activity,
        )
        estimated_sources = pick_activity_peaks(
            candidate_positions=candidate_positions,
            activity=activity,
            max_sources=max_sources,
            min_distance=min_source_distance,
        )
        best_position = (
            estimated_sources[0]["position"].copy()
            if estimated_sources
            else np.array([np.nan, np.nan, np.nan], dtype=float)
        )
        best_activity = estimated_sources[0]["activity"] if estimated_sources else 0.0

        stream_history.append(
            {
                "update_index": float(len(stream_history) + 1),
                "used_measurements": float(end_index),
                "score": poisson_negative_log_likelihood(counts_flat, predicted_counts),
                "total_activity": float(np.sum(activity)),
                "max_activity": float(np.max(activity)),
                "best_position": best_position,
                "best_activity": float(best_activity),
            }
        )
        if store_activity_history:
            activity_history.append(activity.copy())

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
        "stream_history": stream_history,
        "activity_history": activity_history,
    }
