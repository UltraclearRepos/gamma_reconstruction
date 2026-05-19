"""
Streaming reconstruction algorithms for multi-point source experiments.
"""

import numpy as np

from common.geometry import EPS
from reconstruction_algorithms import make_candidate_grid
from reconstruction_objective import (
    build_system_matrix,
    poisson_negative_log_likelihood,
    predict_counts_from_activity,
)


def compute_detector_distance_penalty(
    candidate_positions,
    detector_positions,
):
    """
    Return per-voxel penalty in [0, 1] based only on distance to detector path.

    penalty == 1 for voxels closest to the detector trajectory
    penalty == 0 for voxels farthest from the detector trajectory seen so far
    """
    if detector_positions is None or len(detector_positions) == 0:
        raise ValueError("Detector positions must be provided for detector distance penalty.")

    candidate_positions = np.asarray(candidate_positions, dtype=float)
    detector_positions = np.asarray(detector_positions, dtype=float)

    diffs = candidate_positions[:, None, :] - detector_positions[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    min_distance = np.min(distances, axis=1)

    farthest_distance = max(float(np.max(min_distance)), EPS)
    fraction = np.maximum(min_distance / farthest_distance, 0.0)
    return np.clip(1.0 - fraction, 0.0, 1.0)


def reconstruct_sources(
    counts,
    geometry,
    rotations,
    translations,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_distance_power,
    sparsity_lambda,
    search_min,
    search_max,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
    true_sources,
    method="prefix_mlem",
    voxel_size_cm=1.0,
    inner_iterations_per_update=4,
    measurements_per_update=1,
    relaxation=1.0,
    avoid_detector_strength=0.05,
    avoid_detector_decay_per_update=0.9,
    activity_decay=1.0,
    normalize_activity_to_signal=False,
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
            assumed_distance_power=assumed_distance_power,
            sparsity_lambda=sparsity_lambda,
            search_min=search_min,
            search_max=search_max,
            search_start_distance=search_start_distance,
            search_stop_distance=search_stop_distance,
            search_width=search_width,
            search_height=search_height,
            voxel_size_cm=voxel_size_cm,
            inner_iterations_per_update=inner_iterations_per_update,
            measurements_per_update=measurements_per_update,
            avoid_detector_strength=avoid_detector_strength,
            avoid_detector_decay_per_update=avoid_detector_decay_per_update,
            activity_decay=activity_decay,
            normalize_activity_to_signal=normalize_activity_to_signal,
            store_activity_history=store_activity_history,
            true_sources=true_sources,
        )

    if method == "incremental_mlem":
        return streaming_incremental_mlem_reconstruction(
            counts=counts,
            geometry=geometry,
            rotations=rotations,
            translations=translations,
            assumed_backgrounds=assumed_backgrounds,
            assumed_channel_scales=assumed_channel_scales,
            assumed_distance_power=assumed_distance_power,
            sparsity_lambda=sparsity_lambda,
            search_min=search_min,
            search_max=search_max,
            search_start_distance=search_start_distance,
            search_stop_distance=search_stop_distance,
            search_width=search_width,
            search_height=search_height,
            voxel_size_cm=voxel_size_cm,
            inner_iterations_per_update=inner_iterations_per_update,
            measurements_per_update=measurements_per_update,
            relaxation=relaxation,
            avoid_detector_strength=avoid_detector_strength,
            avoid_detector_decay_per_update=avoid_detector_decay_per_update,
            activity_decay=activity_decay,
            normalize_activity_to_signal=normalize_activity_to_signal,
            store_activity_history=store_activity_history,
            true_sources=true_sources,
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
    assumed_distance_power,
    sparsity_lambda,
    search_min,
    search_max,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
    true_sources,
    voxel_size_cm=1.0,
    inner_iterations_per_update=4,
    measurements_per_update=1,
    avoid_detector_strength=0.05,
    avoid_detector_decay_per_update=0.9,
    activity_decay=1.0,
    normalize_activity_to_signal=False,
    store_activity_history=True,
):
    """
    Run ML-EM in a streaming workflow.

    New measurements are added in chunks, and after each chunk we run a small
    number of ML-EM iterations starting from the previous activity estimate.
    """
    num_measurements, num_channels = counts.shape
    chunk_size = max(1, int(measurements_per_update))
    sparsity_lambda = float(sparsity_lambda)
    if sparsity_lambda < 0.0:
        raise ValueError("sparsity_lambda must be >= 0.")
    avoid_detector_strength = float(avoid_detector_strength)
    avoid_detector_decay_per_update = float(avoid_detector_decay_per_update)
    if not 0.0 <= avoid_detector_strength <= 1.0:
        raise ValueError("avoid_detector_strength must be in the range [0, 1].")
    if not 0.0 <= avoid_detector_decay_per_update <= 1.0:
        raise ValueError("avoid_detector_decay_per_update must be in the range [0, 1].")

    candidate_positions = make_candidate_grid(
        search_min=search_min,
        search_max=search_max,
        voxel_size_cm=voxel_size_cm,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        search_start_distance=search_start_distance,
        search_stop_distance=search_stop_distance,
        search_width=search_width,
        search_height=search_height,
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
            assumed_distance_power=assumed_distance_power,
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

        distance_penalty = compute_detector_distance_penalty(
            candidate_positions=candidate_positions,
            detector_positions=translations[:end_index],
        )
        avoidance_decay = avoid_detector_decay_per_update ** len(stream_history)
        current_avoidance_strength = avoid_detector_strength * avoidance_decay
        avoidance_weights = 1.0 - current_avoidance_strength * distance_penalty
        avoidance_weights = np.maximum(avoidance_weights, EPS)

        if start_index > 0 and float(activity_decay) < 1.0:
            activity *= float(activity_decay)
            activity = np.maximum(activity, EPS)

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
            correction = (system_matrix.T @ ratio) / np.maximum(sensitivity + sparsity_lambda, EPS)
            activity *= correction
            activity = np.maximum(activity, EPS)

        activity *= avoidance_weights
        activity = np.maximum(activity, EPS)

        observed_signal = float(np.sum(np.maximum(counts_flat - backgrounds_flat, 0.0)))
        predicted_signal = float(np.sum(system_matrix @ activity))
        signal_normalization_scale = 1.0
        if normalize_activity_to_signal and predicted_signal > EPS:
            signal_normalization_scale = observed_signal / predicted_signal
            activity *= signal_normalization_scale
            activity = np.maximum(activity, EPS)
            predicted_signal = float(np.sum(system_matrix @ activity))

        predicted_counts = predict_counts_from_activity(
            backgrounds_flat=backgrounds_flat,
            system_matrix=system_matrix,
            activity=activity,
        )
        history_item = {
            "update_index": float(len(stream_history) + 1),
            "used_measurements": float(end_index),
            "score": poisson_negative_log_likelihood(counts_flat, predicted_counts),
            "total_activity": float(np.sum(activity)),
            "max_activity": float(np.max(activity)),
            "avoid_detector_decay": float(avoidance_decay),
            "avoid_detector_strength": float(current_avoidance_strength),
            "observed_signal": float(observed_signal),
            "predicted_signal": float(predicted_signal),
            "signal_normalization_scale": float(signal_normalization_scale),
        }

        stream_history.append(history_item)
        if store_activity_history:
            activity_history.append(activity.copy())

    return {
        "candidate_positions": candidate_positions,
        "activity": activity,
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
    assumed_distance_power,
    sparsity_lambda,
    search_min,
    search_max,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
    true_sources,
    voxel_size_cm=1.0,
    inner_iterations_per_update=1,
    measurements_per_update=1,
    relaxation=1.0,
    avoid_detector_strength=0.05,
    avoid_detector_decay_per_update=0.9,
    activity_decay=1.0,
    normalize_activity_to_signal=False,
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
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must be in the range (0, 1].")
    sparsity_lambda = float(sparsity_lambda)
    if sparsity_lambda < 0.0:
        raise ValueError("sparsity_lambda must be >= 0.")
    avoid_detector_strength = float(avoid_detector_strength)
    avoid_detector_decay_per_update = float(avoid_detector_decay_per_update)
    if not 0.0 <= avoid_detector_strength <= 1.0:
        raise ValueError("avoid_detector_strength must be in the range [0, 1].")
    if not 0.0 <= avoid_detector_decay_per_update <= 1.0:
        raise ValueError("avoid_detector_decay_per_update must be in the range [0, 1].")

    candidate_positions = make_candidate_grid(
        search_min=search_min,
        search_max=search_max,
        voxel_size_cm=voxel_size_cm,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        search_start_distance=search_start_distance,
        search_stop_distance=search_stop_distance,
        search_width=search_width,
        search_height=search_height,
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
            assumed_distance_power=assumed_distance_power,
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
        distance_penalty = compute_detector_distance_penalty(
            candidate_positions=candidate_positions,
            detector_positions=translations[:end_index],
        )
        avoidance_decay = avoid_detector_decay_per_update ** len(stream_history)
        current_avoidance_strength = avoid_detector_strength * avoidance_decay
        avoidance_weights = 1.0 - current_avoidance_strength * distance_penalty
        avoidance_weights = np.maximum(avoidance_weights, EPS)

        if start_index > 0 and float(activity_decay) < 1.0:
            activity *= float(activity_decay)
            activity = np.maximum(activity, EPS)

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
            correction = (response_block.T @ ratio) / np.maximum(sensitivity_block + sparsity_lambda, EPS)
            correction = np.maximum(correction, EPS)
            correction = correction ** relaxation
            activity *= correction
            activity = np.maximum(activity, EPS)

        activity *= avoidance_weights
        activity = np.maximum(activity, EPS)

        system_matrix_prefix = np.vstack(response_blocks)
        counts_prefix = np.concatenate(counts_blocks)
        backgrounds_prefix = np.concatenate(background_blocks)
        observed_signal = float(np.sum(np.maximum(counts_prefix - backgrounds_prefix, 0.0)))
        predicted_signal = float(np.sum(system_matrix_prefix @ activity))
        signal_normalization_scale = 1.0
        if normalize_activity_to_signal and predicted_signal > EPS:
            signal_normalization_scale = observed_signal / predicted_signal
            activity *= signal_normalization_scale
            activity = np.maximum(activity, EPS)
            predicted_signal = float(np.sum(system_matrix_prefix @ activity))

        predicted_prefix = predict_counts_from_activity(
            backgrounds_flat=backgrounds_prefix,
            system_matrix=system_matrix_prefix,
            activity=activity,
        )

        history_item = {
            "update_index": float(len(stream_history) + 1),
            "used_measurements": float(end_index),
            "score": poisson_negative_log_likelihood(counts_prefix, predicted_prefix),
            "total_activity": float(np.sum(activity)),
            "max_activity": float(np.max(activity)),
            "avoid_detector_decay": float(avoidance_decay),
            "avoid_detector_strength": float(current_avoidance_strength),
            "observed_signal": float(observed_signal),
            "predicted_signal": float(predicted_signal),
            "signal_normalization_scale": float(signal_normalization_scale),
        }

        stream_history.append(history_item)
        if store_activity_history:
            activity_history.append(activity.copy())

    return {
        "candidate_positions": candidate_positions,
        "activity": activity,
        "stream_history": stream_history,
        "activity_history": activity_history,
    }
