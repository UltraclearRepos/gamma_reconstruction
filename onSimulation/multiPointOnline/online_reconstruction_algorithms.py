"""
Online reconstruction algorithms for multi-point source experiments.

This module keeps online variants separate from the batch reconstruction code.
The BSREM implementation below receives measurements in small batches and never
needs the full counts/trajectory arrays at function entry.
"""

from time import perf_counter

import numpy as np
import torch

from common.geometry import EPS, points_in_search_cylinder
from reconstruction_objective import (
    build_system_matrix,
    predict_counts_from_activity,
)


TORCH_DTYPE = torch.float32


def to_tensor(array, device):
    """Move numeric data to the Torch device used by reconstruction."""
    return torch.as_tensor(array, dtype=TORCH_DTYPE, device=device)


def tensor_to_float(value):
    """Convert a scalar Torch tensor to a plain Python float."""
    return float(value.detach().cpu().item())


def tensor_to_numpy(array):
    """Copy a Torch tensor back to NumPy for plotting and metrics."""
    return array.detach().cpu().numpy()


def synchronize_if_cuda(device):
    """Wait for queued CUDA work so timing reflects finished GPU operations."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def poisson_negative_log_likelihood_torch(counts, mean_counts):
    """Poisson negative log-likelihood computed in Torch."""
    mean_counts = torch.clamp(mean_counts, min=EPS)
    return tensor_to_float(torch.sum(mean_counts - counts * torch.log(mean_counts)))


def get_torch_device():
    """Return CUDA when available, otherwise run the same Torch code on CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_candidate_grid_data(
    trajectory_center,
    search_radius,
    cylinder_height,
    voxel_size_cm,
):
    """Create candidate source positions and regular-grid metadata for a cylinder."""
    trajectory_center = np.asarray(trajectory_center, dtype=float)
    search_radius = float(search_radius)
    cylinder_height = float(cylinder_height)
    voxel_size_cm = float(voxel_size_cm)
    if search_radius <= 0.0:
        raise ValueError("search_radius must be > 0.")
    if cylinder_height <= 0.0:
        raise ValueError("cylinder_height must be > 0.")
    if voxel_size_cm <= 0.0:
        raise ValueError("voxel_size_cm must be > 0.")

    half_height = 0.5 * cylinder_height
    search_min = trajectory_center + np.array([-search_radius, -search_radius, -half_height])
    search_max = trajectory_center + np.array([search_radius, search_radius, half_height])
    xs = np.arange(search_min[0], search_max[0] + 0.5 * voxel_size_cm, voxel_size_cm)
    ys = np.arange(search_min[1], search_max[1] + 0.5 * voxel_size_cm, voxel_size_cm)
    zs = np.arange(search_min[2], search_max[2] + 0.5 * voxel_size_cm, voxel_size_cm)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    all_positions = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    inside = points_in_search_cylinder(
        points=all_positions,
        trajectory_center=trajectory_center,
        search_radius=search_radius,
        cylinder_height=cylinder_height,
    )
    candidate_positions = all_positions[inside]
    if len(candidate_positions) == 0:
        raise ValueError("No candidate voxels are inside the cylindrical search volume.")
    return {
        "candidate_positions": candidate_positions,
        "candidate_flat_indices": np.flatnonzero(inside).astype(np.int64),
        "grid_shape": (len(xs), len(ys), len(zs)),
        "voxel_size_cm": voxel_size_cm,
    }


def resolve_candidate_grid(
    trajectory_center,
    search_radius,
    cylinder_height,
    voxel_size_cm,
    candidate_grid_override=None,
):
    """Return either a fresh full candidate grid or a validated caller-provided subset."""
    if candidate_grid_override is None:
        return make_candidate_grid_data(
            trajectory_center=trajectory_center,
            search_radius=search_radius,
            cylinder_height=cylinder_height,
            voxel_size_cm=voxel_size_cm,
        )

    required_keys = (
        "candidate_positions",
        "candidate_flat_indices",
        "grid_shape",
        "voxel_size_cm",
    )
    missing = [key for key in required_keys if key not in candidate_grid_override]
    if missing:
        raise KeyError(f"candidate_grid_override missing keys: {missing}")

    candidate_positions = np.asarray(
        candidate_grid_override["candidate_positions"],
        dtype=float,
    )
    candidate_flat_indices = np.asarray(
        candidate_grid_override["candidate_flat_indices"],
        dtype=np.int64,
    )
    if candidate_positions.ndim != 2 or candidate_positions.shape[1] != 3:
        raise ValueError("candidate_grid_override candidate_positions must have shape (N, 3).")
    if candidate_flat_indices.shape != (len(candidate_positions),):
        raise ValueError("candidate_grid_override candidate_flat_indices must match candidate_positions.")
    if len(candidate_positions) == 0:
        raise ValueError("candidate_grid_override must contain at least one candidate voxel.")

    return {
        **dict(candidate_grid_override),
        "candidate_positions": candidate_positions,
        "candidate_flat_indices": candidate_flat_indices,
        "grid_shape": tuple(int(value) for value in candidate_grid_override["grid_shape"]),
        "voxel_size_cm": float(candidate_grid_override["voxel_size_cm"]),
    }


def initialize_activity_tensor(initial_activity, candidate_positions, initial_activity_value, device):
    """Create the reconstruction activity tensor, optionally from a caller-provided start."""
    if initial_activity is None:
        return torch.full(
            (len(candidate_positions),),
            max(float(initial_activity_value), EPS),
            dtype=TORCH_DTYPE,
            device=device,
        )

    initial_activity = torch.as_tensor(initial_activity, dtype=TORCH_DTYPE, device=device)
    if initial_activity.shape != (len(candidate_positions),):
        raise ValueError("initial_activity must match the candidate voxel count.")
    return torch.clamp(initial_activity, min=EPS).clone()


def candidate_activity_volume(
    activity,
    candidate_flat_indices,
    grid_shape,
    device,
):
    """Return dense activity and candidate mask volumes for sparse candidate voxels."""
    flat_indices = torch.as_tensor(candidate_flat_indices, dtype=torch.long, device=device)
    num_grid_points = int(np.prod(grid_shape))
    activity_flat = torch.zeros(num_grid_points, dtype=TORCH_DTYPE, device=device)
    mask_flat = torch.zeros(num_grid_points, dtype=TORCH_DTYPE, device=device)
    activity_flat[flat_indices] = activity
    mask_flat[flat_indices] = 1.0

    return flat_indices, activity_flat.view(*grid_shape), mask_flat.view(*grid_shape)


def l1_prior_gradient(activity):
    """Return L1 sparsity prior gradient for candidate voxels."""
    return torch.ones_like(activity)


def tv_neighbor_prior_gradient(
    activity,
    candidate_flat_indices,
    grid_shape,
    device,
    epsilon=1e-3,
):
    """Return six-neighbor smoothed total-variation prior gradient for candidate voxels."""
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("TV prior epsilon must be > 0.")

    flat_indices, volume, mask = candidate_activity_volume(
        activity=activity,
        candidate_flat_indices=candidate_flat_indices,
        grid_shape=grid_shape,
        device=device,
    )
    gradient_volume = torch.zeros_like(volume)

    def add_tv_edge_gradient(left_slices, right_slices):
        left = volume[left_slices]
        right = volume[right_slices]
        difference = left - right
        denominator = torch.sqrt(difference * difference + epsilon * epsilon)
        left_gradient = difference / torch.clamp(denominator, min=EPS)
        valid = mask[left_slices] * mask[right_slices]

        gradient_volume[left_slices] = gradient_volume[left_slices] + left_gradient * valid
        gradient_volume[right_slices] = gradient_volume[right_slices] - left_gradient * valid

    add_tv_edge_gradient((slice(0, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None)))
    add_tv_edge_gradient((slice(None), slice(0, -1), slice(None)), (slice(None), slice(1, None), slice(None)))
    add_tv_edge_gradient((slice(None), slice(None), slice(0, -1)), (slice(None), slice(None), slice(1, None)))

    return gradient_volume.reshape(-1)[flat_indices]


def compute_reconstruction_prior_gradient(
    activity,
    candidate_grid,
    reconstruction_priors,
    device,
):
    """Return summed one-step-late reconstruction prior gradients from config dictionaries."""
    if not reconstruction_priors:
        return torch.zeros_like(activity)

    gradient = torch.zeros_like(activity)
    for prior in reconstruction_priors:
        prior_type = str(prior.get("type", "")).strip().lower()

        beta = float(prior.get("beta", 0.0))
        if beta < 0.0:
            raise ValueError("Reconstruction prior beta must be >= 0.")
        if beta == 0.0:
            continue

        if prior_type in ("l1", "sparsity"):
            gradient = gradient + beta * l1_prior_gradient(activity)
        elif prior_type in ("tv", "total_variation", "compactness"):
            gradient = gradient + beta * tv_neighbor_prior_gradient(
                activity=activity,
                candidate_flat_indices=candidate_grid["candidate_flat_indices"],
                grid_shape=candidate_grid["grid_shape"],
                device=device,
                epsilon=float(prior.get("epsilon", 1e-3)),
            )
        else:
            raise ValueError(
                f"Unknown reconstruction prior type {prior_type!r}. "
                "Use: l1, tv."
            )

    return gradient


def make_measurement_row_subsets(counts_shape, num_subsets, device):
    """Split flattened H rows into measurement-based subsets."""
    if len(counts_shape) != 2:
        raise ValueError("counts must be a 2D array: measurements x channels.")

    num_measurements, num_channels = counts_shape
    num_subsets = int(num_subsets)
    if num_subsets <= 0:
        raise ValueError("BSREM num_subsets must be > 0.")
    num_subsets = min(num_subsets, int(num_measurements))

    channel_offsets = np.arange(int(num_channels), dtype=np.int64)
    subsets = []
    for subset_index in range(num_subsets):
        measurement_indices = np.arange(subset_index, int(num_measurements), num_subsets)
        row_indices = (
            measurement_indices[:, np.newaxis] * int(num_channels)
            + channel_offsets[np.newaxis, :]
        ).reshape(-1)
        subsets.append(
            {
                "measurement_count": int(len(measurement_indices)),
                "row_indices": torch.as_tensor(row_indices, dtype=torch.long, device=device),
            }
        )
    return subsets


def iter_measurement_batches(counts, rotations, translations, batch_size):
    """Yield measurement dictionaries from full arrays for local experiments."""
    counts = np.asarray(counts, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if counts.ndim != 2:
        raise ValueError("counts must be a 2D array: measurements x channels.")
    if len(rotations) != counts.shape[0] or len(translations) != counts.shape[0]:
        raise ValueError("counts, rotations, and translations must have the same measurement count.")

    for start in range(0, counts.shape[0], batch_size):
        stop = min(start + batch_size, counts.shape[0])
        yield {
            "counts": counts[start:stop],
            "rotations": rotations[start:stop],
            "translations": translations[start:stop],
        }


def normalize_measurement_batch(batch):
    """Return counts, rotations, translations arrays from a batch dict or tuple."""
    if isinstance(batch, dict):
        counts = batch["counts"]
        rotations = batch["rotations"]
        translations = batch["translations"]
    else:
        if len(batch) != 3:
            raise ValueError("measurement batch tuples must be: counts, rotations, translations.")
        counts, rotations, translations = batch

    counts = np.asarray(counts, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    if counts.ndim != 2:
        raise ValueError("batch counts must be a 2D array: measurements x channels.")
    if counts.shape[0] == 0:
        raise ValueError("measurement batches must not be empty.")
    if len(rotations) != counts.shape[0] or len(translations) != counts.shape[0]:
        raise ValueError("batch counts, rotations, and translations must have the same measurement count.")
    return counts, rotations, translations


def bsrem_relaxation_value(update_index, relaxation_initial, relaxation_decay, relaxation_power):
    """Return the BSREM relaxation value for a one-based online update index."""
    return relaxation_initial / (
        (1.0 + relaxation_decay * float(update_index - 1)) ** relaxation_power
    )


def reconstruct_sources_online(method="online_bsrem", **kwargs):
    """Run an online reconstruction method. Currently supports online BSREM."""
    method = str(method).strip().lower().replace("-", "_")
    if method in ("online_bsrem", "bsrem_online"):
        return online_bsrem_reconstruction(**kwargs)
    raise ValueError(
        f"Unknown online reconstruction method '{method}'. "
        "Use: online_bsrem."
    )


def make_online_history_item(
    activity,
    known_system_matrix,
    known_counts_flat,
    known_backgrounds_flat,
    online_update_index,
    batch_index,
    batch_update,
    latest_batch_measurements,
    received_measurements,
    relaxation,
    reconstruction_priors,
    num_subsets,
    torch_device_name,
):
    """Return one history row for the current online BSREM state."""
    predicted_counts = predict_counts_from_activity(
        backgrounds_flat=known_backgrounds_flat,
        system_matrix=known_system_matrix,
        activity=activity,
    )
    return {
        "iteration": float(online_update_index),
        "online_update": float(online_update_index),
        "batch_index": int(batch_index),
        "batch_update": int(batch_update),
        "latest_batch_measurements": int(latest_batch_measurements),
        "received_measurements": int(received_measurements),
        "score": poisson_negative_log_likelihood_torch(
            known_counts_flat,
            predicted_counts,
        ),
        "total_activity": tensor_to_float(torch.sum(activity)),
        "max_activity": tensor_to_float(torch.max(activity)),
        "reconstruction_priors": [dict(prior) for prior in reconstruction_priors],
        "bsrem_num_subsets": int(min(num_subsets, received_measurements)),
        "bsrem_relaxation": float(relaxation),
        "torch_device": torch_device_name,
    }


def make_refinement_support_indices(
    activity,
    candidate_positions,
    threshold_fraction,
    min_component_voxels,
    min_total_activity_fraction,
    padding_cm,
    candidate_flat_indices,
    grid_shape,
):
    """Return full candidate indices selected for local refinement."""
    from metrics import find_activity_components

    activity = np.maximum(np.asarray(activity, dtype=float), 0.0)
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    if len(activity) == 0 or float(np.max(activity)) <= EPS:
        return np.array([], dtype=np.int64)

    components = find_activity_components(
        activity=activity,
        candidate_positions=candidate_positions,
        candidate_flat_indices=candidate_flat_indices,
        grid_shape=grid_shape,
        threshold_fraction=threshold_fraction,
        min_component_voxels=min_component_voxels,
        min_total_activity_fraction=min_total_activity_fraction,
    )
    support_mask = np.zeros(len(activity), dtype=bool)
    for component in components:
        support_mask[np.asarray(component["indices"], dtype=np.int64)] = True
    if not np.any(support_mask):
        return np.array([], dtype=np.int64)

    padding_cm = float(padding_cm)
    if padding_cm > 0.0:
        active_positions = candidate_positions[support_mask]
        padded_mask = np.zeros(len(candidate_positions), dtype=bool)
        padding_squared = padding_cm * padding_cm
        chunk_size = 2048
        for start in range(0, len(candidate_positions), chunk_size):
            stop = min(start + chunk_size, len(candidate_positions))
            squared_distances = np.sum(
                (
                    candidate_positions[start:stop, np.newaxis, :]
                    - active_positions[np.newaxis, :, :]
                )
                ** 2,
                axis=2,
            )
            padded_mask[start:stop] = np.any(squared_distances <= padding_squared, axis=1)
        support_mask = padded_mask

    return np.flatnonzero(support_mask).astype(np.int64)


def run_periodic_refinement(
    activity,
    known_system_matrix,
    known_counts_flat,
    known_backgrounds_flat,
    candidate_grid,
    support_indices,
    reconstruction_priors,
    refinement_updates,
    refinement_history_interval,
    num_channels,
    num_subsets,
    relaxation,
    max_relative_update,
    device,
):
    """Run local BSREM updates only on support voxels and return activity snapshots."""
    support_indices = np.asarray(support_indices, dtype=np.int64)
    support_index_tensor = torch.as_tensor(support_indices, dtype=torch.long, device=device)
    support_activity = torch.clamp(activity[support_index_tensor].clone(), min=EPS)
    support_matrix = known_system_matrix[:, support_index_tensor]
    support_sensitivity = torch.sum(support_matrix, dim=0)
    num_channels = int(num_channels)
    if num_channels <= 0:
        raise ValueError("num_channels must be > 0.")

    # known_system_matrix rows are flattened measurement/channel rows.
    known_measurements = int(known_system_matrix.shape[0] // num_channels)
    subsets = make_measurement_row_subsets(
        counts_shape=(known_measurements, num_channels),
        num_subsets=num_subsets,
        device=device,
    )
    support_candidate_grid = {
        "candidate_positions": candidate_grid["candidate_positions"][support_indices],
        "candidate_flat_indices": candidate_grid["candidate_flat_indices"][support_indices],
        "grid_shape": candidate_grid["grid_shape"],
        "voxel_size_cm": candidate_grid["voxel_size_cm"],
    }
    support_history = []
    support_history_steps = []

    refinement_updates = int(refinement_updates)
    refinement_history_interval = int(refinement_history_interval)

    for refinement_update_index in range(1, refinement_updates + 1):
        for subset in subsets:
            row_indices = subset["row_indices"]
            subset_matrix = support_matrix[row_indices, :]
            subset_counts = known_counts_flat[row_indices]
            subset_backgrounds = known_backgrounds_flat[row_indices]
            predicted_counts = predict_counts_from_activity(
                backgrounds_flat=subset_backgrounds,
                system_matrix=subset_matrix,
                activity=support_activity,
            )
            ratio = subset_counts / torch.clamp(predicted_counts, min=EPS)
            subset_scale = known_measurements / max(int(subset["measurement_count"]), 1)
            data_gradient = subset_scale * (subset_matrix.T @ (ratio - 1.0))
            prior_gradient = compute_reconstruction_prior_gradient(
                activity=support_activity,
                candidate_grid=support_candidate_grid,
                reconstruction_priors=reconstruction_priors,
                device=device,
            )
            relative_update = (
                relaxation
                * (data_gradient - prior_gradient)
                / torch.clamp(support_sensitivity, min=EPS)
            )
            relative_update = torch.clamp(
                relative_update,
                min=-0.95,
                max=max_relative_update,
            )
            support_activity = torch.clamp(support_activity * (1.0 + relative_update), min=EPS)
        if (
            refinement_update_index == 1
            or refinement_update_index % refinement_history_interval == 0
            or refinement_update_index == refinement_updates
        ):
            support_history.append(tensor_to_numpy(support_activity).copy())
            support_history_steps.append(int(refinement_update_index))

    return support_activity, support_history, support_history_steps


def online_bsrem_reconstruction(
    measurement_batches,
    background_counts,
    relative_response_table_path,
    trajectory_center,
    search_radius,
    cylinder_height,
    response_column="response_weighted_probability",
    activity_to_emission_scale=1.0,
    voxel_size_cm=1.0,
    updates_per_batch=1,
    history_interval=1,
    reconstruction_priors=None,
    num_subsets=8,
    relaxation_initial=1.0,
    relaxation_decay=0.01,
    relaxation_power=1.0,
    max_relative_update=2.0,
    candidate_grid_override=None,
    initial_activity=None,
    enable_periodic_refinement=False,
    refinement_interval_batches=5,
    refinement_updates=10,
    refinement_history_interval=1,
    refinement_threshold_fraction=0.1,
    refinement_min_component_voxels=3,
    refinement_min_total_activity_fraction=0.0,
    refinement_padding_cm=0.75,
    refinement_outside_support_decay=0.5,
    refinement_priors=None,
):
    """
    Reconstruct a voxel activity map with online BSREM.

    measurement_batches is an iterable of dictionaries or tuples containing
    only the newly available measurements:
        {"counts": ..., "rotations": ..., "translations": ...}
    or:
        (counts, rotations, translations)

    After each incoming batch, the algorithm performs updates_per_batch BSREM
    passes over all measurements received so far. Optional periodic refinement
    locally updates thresholded support voxels and decays activity outside that
    support instead of zeroing it.
    """
    updates_per_batch = int(updates_per_batch)
    history_interval = int(history_interval)
    if updates_per_batch <= 0:
        raise ValueError("updates_per_batch must be > 0.")
    if history_interval <= 0:
        raise ValueError("history_interval must be > 0.")

    relaxation_initial = float(relaxation_initial)
    relaxation_decay = float(relaxation_decay)
    relaxation_power = float(relaxation_power)
    max_relative_update = float(max_relative_update)
    if relaxation_initial <= 0.0:
        raise ValueError("BSREM relaxation_initial must be > 0.")
    if relaxation_decay < 0.0:
        raise ValueError("BSREM relaxation_decay must be >= 0.")
    if relaxation_power < 0.0:
        raise ValueError("BSREM relaxation_power must be >= 0.")
    if max_relative_update <= 0.0:
        raise ValueError("BSREM max_relative_update must be > 0.")
    refinement_interval_batches = int(refinement_interval_batches)
    refinement_updates = int(refinement_updates)
    refinement_history_interval = int(refinement_history_interval)
    refinement_threshold_fraction = float(refinement_threshold_fraction)
    refinement_min_component_voxels = int(refinement_min_component_voxels)
    refinement_min_total_activity_fraction = float(refinement_min_total_activity_fraction)
    refinement_padding_cm = float(refinement_padding_cm)
    refinement_outside_support_decay = float(refinement_outside_support_decay)
    if enable_periodic_refinement:
        if refinement_interval_batches <= 0:
            raise ValueError("refinement_interval_batches must be > 0.")
        if refinement_updates <= 0:
            raise ValueError("refinement_updates must be > 0.")
        if refinement_history_interval <= 0:
            raise ValueError("refinement_history_interval must be > 0.")
        if not 0.0 <= refinement_threshold_fraction <= 1.0:
            raise ValueError("refinement_threshold_fraction must be between 0 and 1.")
        if refinement_min_component_voxels <= 0:
            raise ValueError("refinement_min_component_voxels must be > 0.")
        if not 0.0 <= refinement_min_total_activity_fraction <= 1.0:
            raise ValueError(
                "refinement_min_total_activity_fraction must be between 0 and 1."
            )
        if refinement_padding_cm < 0.0:
            raise ValueError("refinement_padding_cm must be >= 0.")
        if not 0.0 <= refinement_outside_support_decay <= 1.0:
            raise ValueError("refinement_outside_support_decay must be between 0 and 1.")

    candidate_grid = resolve_candidate_grid(
        trajectory_center=trajectory_center,
        search_radius=search_radius,
        cylinder_height=cylinder_height,
        voxel_size_cm=voxel_size_cm,
        candidate_grid_override=candidate_grid_override,
    )
    candidate_positions = candidate_grid["candidate_positions"]
    device = get_torch_device()
    torch_device_name = str(device)

    background_counts = np.asarray(background_counts, dtype=float)
    if background_counts.ndim != 1:
        raise ValueError("background_counts must be a 1D array with one value per channel.")

    reconstruction_priors = list(reconstruction_priors or [])
    refinement_priors = list(refinement_priors if refinement_priors is not None else reconstruction_priors)

    known_system_matrices = []
    known_counts = []
    known_backgrounds = []
    received_measurements = 0
    batch_count = 0
    online_update_index = 0
    last_history_update = 0
    activity = None
    history = []
    activity_history = []
    refinement_events = []
    refinement_count = 0
    system_matrix_elapsed_seconds = 0.0
    transfer_elapsed_seconds = 0.0
    iteration_start_time = perf_counter()

    with torch.no_grad():
        for batch in measurement_batches:
            batch_count += 1
            counts_array, rotations_array, translations_array = normalize_measurement_batch(batch)
            batch_measurements, num_channels = counts_array.shape
            if background_counts.shape != (num_channels,):
                raise ValueError("background_counts length must match batch channel count.")

            system_matrix_start_time = perf_counter()
            batch_system_matrix = build_system_matrix(
                candidate_positions=candidate_positions,
                rotations=rotations_array,
                translations=translations_array,
                relative_response_table_path=relative_response_table_path,
                response_column=response_column,
                activity_to_emission_scale=activity_to_emission_scale,
                device=device,
                dtype=TORCH_DTYPE,
            )
            synchronize_if_cuda(device)
            system_matrix_elapsed_seconds += perf_counter() - system_matrix_start_time

            transfer_start_time = perf_counter()
            batch_counts_flat = to_tensor(counts_array.ravel(), device)
            batch_backgrounds_flat = to_tensor(
                np.broadcast_to(background_counts, counts_array.shape).astype(float).ravel(),
                device,
            )
            known_system_matrices.append(batch_system_matrix)
            known_counts.append(batch_counts_flat)
            known_backgrounds.append(batch_backgrounds_flat)
            received_measurements += int(batch_measurements)
            known_system_matrix = torch.cat(known_system_matrices, dim=0)
            known_counts_flat = torch.cat(known_counts, dim=0)
            known_backgrounds_flat = torch.cat(known_backgrounds, dim=0)
            known_counts_shape = (received_measurements, num_channels)
            subsets = make_measurement_row_subsets(
                counts_shape=known_counts_shape,
                num_subsets=num_subsets,
                device=device,
            )
            sensitivity = torch.sum(known_system_matrix, dim=0)

            if activity is None:
                initial_signal = torch.clamp(known_counts_flat - known_backgrounds_flat, min=0.0)
                initial_activity_value = tensor_to_float(
                    torch.sum(initial_signal) / torch.clamp(torch.sum(sensitivity), min=EPS)
                )
                activity = initialize_activity_tensor(
                    initial_activity=initial_activity,
                    candidate_positions=candidate_positions,
                    initial_activity_value=initial_activity_value,
                    device=device,
                )
            synchronize_if_cuda(device)
            transfer_elapsed_seconds += perf_counter() - transfer_start_time

            for batch_update in range(1, updates_per_batch + 1):
                online_update_index += 1
                relaxation = bsrem_relaxation_value(
                    update_index=online_update_index,
                    relaxation_initial=relaxation_initial,
                    relaxation_decay=relaxation_decay,
                    relaxation_power=relaxation_power,
                )

                for subset in subsets:
                    row_indices = subset["row_indices"]
                    subset_matrix = known_system_matrix[row_indices, :]
                    subset_counts = known_counts_flat[row_indices]
                    subset_backgrounds = known_backgrounds_flat[row_indices]
                    predicted_counts = predict_counts_from_activity(
                        backgrounds_flat=subset_backgrounds,
                        system_matrix=subset_matrix,
                        activity=activity,
                    )
                    ratio = subset_counts / torch.clamp(predicted_counts, min=EPS)
                    subset_scale = received_measurements / max(
                        int(subset["measurement_count"]),
                        1,
                    )
                    data_gradient = subset_scale * (subset_matrix.T @ (ratio - 1.0))
                    prior_gradient = compute_reconstruction_prior_gradient(
                        activity=activity,
                        candidate_grid=candidate_grid,
                        reconstruction_priors=reconstruction_priors,
                        device=device,
                    )
                    relative_update = (
                        relaxation
                        * (data_gradient - prior_gradient)
                        / torch.clamp(sensitivity, min=EPS)
                    )
                    relative_update = torch.clamp(
                        relative_update,
                        min=-0.95,
                        max=max_relative_update,
                    )
                    activity = activity * (1.0 + relative_update)
                    activity = torch.clamp(activity, min=EPS)

                if online_update_index == 1 or online_update_index % history_interval == 0:
                    history.append(
                        make_online_history_item(
                            activity=activity,
                            known_system_matrix=known_system_matrix,
                            known_counts_flat=known_counts_flat,
                            known_backgrounds_flat=known_backgrounds_flat,
                            online_update_index=online_update_index,
                            batch_index=batch_count,
                            batch_update=batch_update,
                            latest_batch_measurements=batch_measurements,
                            received_measurements=received_measurements,
                            relaxation=relaxation,
                            reconstruction_priors=reconstruction_priors,
                            num_subsets=num_subsets,
                            torch_device_name=torch_device_name,
                        )
                    )
                    activity_history.append(tensor_to_numpy(activity).copy())
                    last_history_update = online_update_index

            if enable_periodic_refinement and batch_count % refinement_interval_batches == 0:
                full_activity_before_refinement = tensor_to_numpy(activity)
                support_indices = make_refinement_support_indices(
                    activity=full_activity_before_refinement,
                    candidate_positions=candidate_positions,
                    threshold_fraction=refinement_threshold_fraction,
                    min_component_voxels=refinement_min_component_voxels,
                    min_total_activity_fraction=refinement_min_total_activity_fraction,
                    padding_cm=refinement_padding_cm,
                    candidate_flat_indices=candidate_grid["candidate_flat_indices"],
                    grid_shape=candidate_grid["grid_shape"],
                )
                if len(support_indices) > 0:
                    refinement_count += 1
                    refinement_relaxation = bsrem_relaxation_value(
                        update_index=online_update_index,
                        relaxation_initial=relaxation_initial,
                        relaxation_decay=relaxation_decay,
                        relaxation_power=relaxation_power,
                    )
                    refined_support_activity, refinement_activity_history, refinement_activity_history_steps = run_periodic_refinement(
                        activity=activity,
                        known_system_matrix=known_system_matrix,
                        known_counts_flat=known_counts_flat,
                        known_backgrounds_flat=known_backgrounds_flat,
                        candidate_grid=candidate_grid,
                        support_indices=support_indices,
                        reconstruction_priors=refinement_priors,
                        refinement_updates=refinement_updates,
                        refinement_history_interval=refinement_history_interval,
                        num_channels=num_channels,
                        num_subsets=num_subsets,
                        relaxation=refinement_relaxation,
                        max_relative_update=max_relative_update,
                        device=device,
                    )
                    support_index_tensor = torch.as_tensor(support_indices, dtype=torch.long, device=device)
                    support_mask = torch.zeros_like(activity, dtype=torch.bool)
                    support_mask[support_index_tensor] = True
                    activity = torch.where(
                        support_mask,
                        activity,
                        torch.clamp(activity * refinement_outside_support_decay, min=EPS),
                    )
                    activity[support_index_tensor] = torch.clamp(refined_support_activity, min=EPS)
                    refinement_events.append(
                        {
                            "after_online_update": int(online_update_index),
                            "batch_index": int(batch_count),
                            "refinement_index": int(refinement_count),
                            "support_indices": support_indices.copy(),
                            "support_candidate_positions": candidate_positions[support_indices].copy(),
                            "support_activity": tensor_to_numpy(refined_support_activity).copy(),
                            "activity_history": [
                                np.asarray(snapshot, dtype=float).copy()
                                for snapshot in refinement_activity_history
                            ],
                            "activity_history_steps": [
                                int(refinement_step)
                                for refinement_step in refinement_activity_history_steps
                            ],
                            "refinement_updates": int(refinement_updates),
                            "refinement_history_interval": int(refinement_history_interval),
                            "outside_support_decay": float(refinement_outside_support_decay),
                            "threshold_fraction": float(refinement_threshold_fraction),
                            "min_component_voxels": int(refinement_min_component_voxels),
                            "min_total_activity_fraction": float(
                                refinement_min_total_activity_fraction
                            ),
                            "padding_cm": float(refinement_padding_cm),
                        }
                    )
                    last_history_update = -1

    if batch_count == 0:
        raise ValueError("measurement_batches must yield at least one batch.")

    synchronize_if_cuda(device)
    iteration_elapsed_seconds = perf_counter() - iteration_start_time

    final_system_matrix = torch.cat(known_system_matrices, dim=0)
    if last_history_update != online_update_index:
        final_counts_flat = torch.cat(known_counts, dim=0)
        final_backgrounds_flat = torch.cat(known_backgrounds, dim=0)
        history.append(
            make_online_history_item(
                activity=activity,
                known_system_matrix=final_system_matrix,
                known_counts_flat=final_counts_flat,
                known_backgrounds_flat=final_backgrounds_flat,
                online_update_index=online_update_index,
                batch_index=batch_count,
                batch_update=updates_per_batch,
                latest_batch_measurements=0,
                received_measurements=received_measurements,
                relaxation=bsrem_relaxation_value(
                    update_index=online_update_index,
                    relaxation_initial=relaxation_initial,
                    relaxation_decay=relaxation_decay,
                    relaxation_power=relaxation_power,
                ),
                reconstruction_priors=reconstruction_priors,
                num_subsets=num_subsets,
                torch_device_name=torch_device_name,
            )
        )
        activity_history.append(tensor_to_numpy(activity).copy())

    return {
        "candidate_positions": candidate_positions,
        "candidate_flat_indices": candidate_grid["candidate_flat_indices"].copy(),
        "grid_shape": tuple(candidate_grid["grid_shape"]),
        "voxel_size_cm": float(candidate_grid["voxel_size_cm"]),
        "activity": tensor_to_numpy(activity).copy(),
        "activity_history": [snapshot.copy() for snapshot in activity_history],
        "history": history,
        "system_matrix": tensor_to_numpy(final_system_matrix).copy(),
        "torch_device": torch_device_name,
        "method": "online_bsrem",
        "online": {
            "measurement_batches": int(batch_count),
            "received_measurements": int(received_measurements),
            "updates_per_batch": int(updates_per_batch),
            "periodic_refinement_enabled": bool(enable_periodic_refinement),
            "periodic_refinement_count": int(refinement_count),
            "periodic_refinement_history_interval": int(refinement_history_interval),
            "periodic_refinement_min_total_activity_fraction": float(
                refinement_min_total_activity_fraction
            ),
        },
        "refinement_events": refinement_events,
        "timings": {
            "system_matrix_seconds": system_matrix_elapsed_seconds,
            "transfer_seconds": transfer_elapsed_seconds,
            "online_bsrem_iterations_seconds": iteration_elapsed_seconds,
            "bsrem_iterations_seconds": iteration_elapsed_seconds,
        },
    }
