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


def compute_detector_avoidance_weights(
    candidate_positions,
    detector_positions,
    avoid_distance,
    power=2.0,
):
    """
    Return per-voxel weights in (0, 1] that penalize voxels too close to the detector.

    weight ~= 0 when voxel is at distance 0 from any detector position
    weight == 1 when voxel is farther than avoid_distance from all detector positions
    """
    avoid_distance = float(avoid_distance)
    power = float(power)

    if avoid_distance <= 0.0:
        return np.ones(len(candidate_positions), dtype=float)
    if detector_positions is None or len(detector_positions) == 0:
        return np.ones(len(candidate_positions), dtype=float)

    candidate_positions = np.asarray(candidate_positions, dtype=float)
    detector_positions = np.asarray(detector_positions, dtype=float)

    diffs = candidate_positions[:, None, :] - detector_positions[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    min_distance = np.min(distances, axis=1)

    fraction = np.minimum(1.0, np.maximum(min_distance / avoid_distance, 0.0))
    weights = fraction**power
    return np.maximum(weights, EPS)


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
    method="prefix_mlem",
    grid_size=9,
    inner_iterations_per_update=4,
    measurements_per_update=1,
    max_sources=3,
    min_source_distance=0.08,
    relaxation=1.0,
    avoid_detector_distance=0.0,
    avoid_detector_power=2.0,
    store_activity_history=True,
):
    """
    Run one streaming reconstruction method using a plain if/elif dispatcher.

    Methods:
        - "prefix_mlem": after each update, run ML-EM iterations on all data seen so far
        - "incremental_mlem": update activity using only the newly arrived chunk (OSEM-like)
    """
    method = method.strip().lower().replace("-", "_")

    if method == "prefix_mlem":
        return streaming_mlem_reconstruction(
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
            inner_iterations_per_update=inner_iterations_per_update,
            measurements_per_update=measurements_per_update,
            max_sources=max_sources,
            min_source_distance=min_source_distance,
            avoid_detector_distance=avoid_detector_distance,
            avoid_detector_power=avoid_detector_power,
            store_activity_history=store_activity_history,
        )

    if method == "incremental_mlem":
        return streaming_incremental_mlem_reconstruction(
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
            inner_iterations_per_update=inner_iterations_per_update,
            measurements_per_update=measurements_per_update,
            max_sources=max_sources,
            min_source_distance=min_source_distance,
            relaxation=relaxation,
            avoid_detector_distance=avoid_detector_distance,
            avoid_detector_power=avoid_detector_power,
            store_activity_history=store_activity_history,
        )

    raise ValueError(
        "Unknown streaming reconstruction method "
        f"'{method}'. Use: prefix_mlem, incremental_mlem."
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
    avoid_detector_distance=0.0,
    avoid_detector_power=2.0,
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

        avoidance_weights = compute_detector_avoidance_weights(
            candidate_positions=candidate_positions,
            detector_positions=translations[:end_index],
            avoid_distance=avoid_detector_distance,
            power=avoid_detector_power,
        )

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
            activity *= avoidance_weights
            activity = np.maximum(activity, EPS)

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


def streaming_incremental_mlem_reconstruction(
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
    inner_iterations_per_update=1,
    measurements_per_update=1,
    max_sources=3,
    min_source_distance=0.08,
    relaxation=1.0,
    avoid_detector_distance=0.0,
    avoid_detector_power=2.0,
    store_activity_history=True,
):
    """
    Streaming-friendly ML-EM variant (OSEM-like / incremental EM).

    Each update uses only the *newly arrived* measurement chunk to update the
    activity map. This avoids reprocessing the entire prefix of data at every
    update, which is closer to a true streaming workflow.
    """
    num_measurements, num_channels = counts.shape
    chunk_size = max(1, int(measurements_per_update))
    relaxation = float(relaxation)

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

        sensitivity_block = np.sum(response_block, axis=0)
        avoidance_weights = compute_detector_avoidance_weights(
            candidate_positions=candidate_positions,
            detector_positions=translations[:end_index],
            avoid_distance=avoid_detector_distance,
            power=avoid_detector_power,
        )

        if start_index == 0:
            initial_signal = np.maximum(counts_block - backgrounds_block, 0.0)
            initial_activity = np.sum(initial_signal) / max(float(np.sum(sensitivity_block)), EPS)
            activity[:] = max(initial_activity, EPS)

        for _ in range(max(1, int(inner_iterations_per_update))):
            predicted_counts = predict_counts_from_activity(
                backgrounds_flat=backgrounds_block,
                system_matrix=response_block,
                activity=activity,
            )
            ratio = counts_block / np.maximum(predicted_counts, EPS)
            correction = (response_block.T @ ratio) / np.maximum(sensitivity_block, EPS)
            correction = np.maximum(correction, EPS)
            if abs(relaxation - 1.0) > 1e-12:
                correction = correction ** relaxation
            activity *= correction
            activity *= avoidance_weights
            activity = np.maximum(activity, EPS)

        system_matrix_prefix = np.vstack(response_blocks)
        counts_prefix = np.concatenate(counts_blocks)
        backgrounds_prefix = np.concatenate(background_blocks)
        predicted_prefix = predict_counts_from_activity(
            backgrounds_flat=backgrounds_prefix,
            system_matrix=system_matrix_prefix,
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
                "score": poisson_negative_log_likelihood(counts_prefix, predicted_prefix),
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
