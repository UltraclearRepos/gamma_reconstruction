"""
Online reconstruction algorithms for multi-point source experiments.

This module keeps online variants separate from the batch reconstruction code.
The BSREM implementation below receives measurements in small batches and never
needs the full counts/trajectory arrays at function entry.
"""

from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from common.geometry import EPS, points_in_search_cylinder
from reconstruction_objective import (
    build_system_matrix_torch,
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


def get_cuda_device():
    """Return the CUDA device used by reconstruction, or fail loudly if unavailable."""
    if not torch.cuda.is_available():
        cuda_build = torch.version.cuda
        raise RuntimeError(
            "Online BSREM is configured to run on CUDA, but PyTorch cannot see a CUDA device. "
            f"Installed torch version: {torch.__version__}; torch CUDA build: {cuda_build}. "
            "Install a CUDA-enabled PyTorch build and make sure the NVIDIA driver/GPU is visible."
        )
    return torch.device("cuda")


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


def gaussian_kernel_1d(sigma_voxels, device):
    """Return a normalized 1D Gaussian kernel for separable 3D smoothing."""
    sigma_voxels = float(sigma_voxels)
    if sigma_voxels <= 0.0:
        raise ValueError("sigma_voxels must be > 0.")
    radius = max(1, int(np.ceil(3.0 * sigma_voxels)))
    offsets = torch.arange(-radius, radius + 1, dtype=TORCH_DTYPE, device=device)
    kernel = torch.exp(-0.5 * (offsets / sigma_voxels) ** 2)
    kernel /= torch.sum(kernel)
    return kernel


def gaussian_smooth_volume(volume, kernel):
    """Apply a separable 3D Gaussian filter to a 5D Torch volume."""
    radius = kernel.numel() // 2
    kernel_x = kernel.view(1, 1, -1, 1, 1)
    kernel_y = kernel.view(1, 1, 1, -1, 1)
    kernel_z = kernel.view(1, 1, 1, 1, -1)
    volume = F.conv3d(volume, kernel_x, padding=(radius, 0, 0))
    volume = F.conv3d(volume, kernel_y, padding=(0, radius, 0))
    volume = F.conv3d(volume, kernel_z, padding=(0, 0, radius))
    return volume


def apply_sieve_filter(
    activity,
    candidate_flat_indices,
    grid_shape,
    voxel_size_cm,
    sieve_sigma_cm,
    device,
):
    """Smooth activity on the 3D voxel grid and return values at candidate voxels."""
    sieve_sigma_cm = float(sieve_sigma_cm)
    if sieve_sigma_cm <= 0.0:
        return activity

    sigma_voxels = sieve_sigma_cm / float(voxel_size_cm)
    kernel = gaussian_kernel_1d(sigma_voxels=sigma_voxels, device=device)
    flat_indices = torch.as_tensor(candidate_flat_indices, dtype=torch.long, device=device)
    num_grid_points = int(np.prod(grid_shape))

    total_before = torch.sum(activity)
    activity_flat = torch.zeros(num_grid_points, dtype=TORCH_DTYPE, device=device)
    mask_flat = torch.zeros(num_grid_points, dtype=TORCH_DTYPE, device=device)
    activity_flat[flat_indices] = activity
    mask_flat[flat_indices] = 1.0

    volume = activity_flat.view(1, 1, *grid_shape)
    mask = mask_flat.view(1, 1, *grid_shape)
    smoothed_volume = gaussian_smooth_volume(volume, kernel)
    smoothed_mask = gaussian_smooth_volume(mask, kernel)
    normalized_volume = smoothed_volume / torch.clamp(smoothed_mask, min=EPS)

    filtered = normalized_volume.reshape(-1)[flat_indices]
    filtered = torch.clamp(filtered, min=EPS)
    total_after = torch.sum(filtered)
    if tensor_to_float(total_after) > EPS:
        filtered = filtered * (total_before / total_after)
    return filtered


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


def online_bsrem_reconstruction(
    measurement_batches,
    geometry,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_distance_power,
    trajectory_center,
    search_radius,
    cylinder_height,
    assumed_angular_peak_angle_deg=0.0,
    assumed_angular_width_deg=None,
    assumed_collimator_half_angle_deg=None,
    assumed_collimator_edge_width_deg=3.0,
    assumed_collimator_leakage=0.0,
    assumed_skin_thickness_cm=0.0,
    assumed_skin_mu_per_cm=0.0,
    assumed_detector_intrinsic_efficiency=1.0,
    assumed_response_scale=1.0,
    voxel_size_cm=1.0,
    updates_per_batch=1,
    history_interval=1,
    reconstruction_priors=None,
    reconstruction_sieve_sigma_cm=0.0,
    num_subsets=8,
    relaxation_initial=1.0,
    relaxation_decay=0.01,
    relaxation_power=1.0,
    max_relative_update=2.0,
    candidate_grid_override=None,
    initial_activity=None,
):
    """
    Reconstruct a voxel activity map with online BSREM.

    measurement_batches is an iterable of dictionaries or tuples containing
    only the newly available measurements:
        {"counts": ..., "rotations": ..., "translations": ...}
    or:
        (counts, rotations, translations)

    After each incoming batch, the algorithm performs updates_per_batch BSREM
    passes over all measurements received so far. Refinement is intentionally
    not included here.
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

    candidate_grid = resolve_candidate_grid(
        trajectory_center=trajectory_center,
        search_radius=search_radius,
        cylinder_height=cylinder_height,
        voxel_size_cm=voxel_size_cm,
        candidate_grid_override=candidate_grid_override,
    )
    candidate_positions = candidate_grid["candidate_positions"]
    device = get_cuda_device()
    torch_device_name = str(device)

    assumed_backgrounds = np.asarray(assumed_backgrounds, dtype=float)
    if assumed_backgrounds.ndim != 1:
        raise ValueError("assumed_backgrounds must be a 1D array with one value per channel.")

    reconstruction_priors = list(reconstruction_priors or [])
    sieve_sigma_cm = float(reconstruction_sieve_sigma_cm)

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
    system_matrix_elapsed_seconds = 0.0
    transfer_elapsed_seconds = 0.0
    iteration_start_time = perf_counter()

    with torch.no_grad():
        for batch in measurement_batches:
            batch_count += 1
            counts_array, rotations_array, translations_array = normalize_measurement_batch(batch)
            batch_measurements, num_channels = counts_array.shape
            if assumed_backgrounds.shape != (num_channels,):
                raise ValueError("assumed_backgrounds length must match batch channel count.")

            system_matrix_start_time = perf_counter()
            batch_system_matrix = build_system_matrix_torch(
                candidate_positions=candidate_positions,
                geometry=geometry,
                rotations=rotations_array,
                translations=translations_array,
                assumed_channel_scales=assumed_channel_scales,
                assumed_distance_power=assumed_distance_power,
                assumed_angular_peak_angle_deg=assumed_angular_peak_angle_deg,
                assumed_angular_width_deg=assumed_angular_width_deg,
                assumed_collimator_half_angle_deg=assumed_collimator_half_angle_deg,
                assumed_collimator_edge_width_deg=assumed_collimator_edge_width_deg,
                assumed_collimator_leakage=assumed_collimator_leakage,
                assumed_skin_thickness_cm=assumed_skin_thickness_cm,
                assumed_skin_mu_per_cm=assumed_skin_mu_per_cm,
                assumed_detector_intrinsic_efficiency=assumed_detector_intrinsic_efficiency,
                assumed_response_scale=assumed_response_scale,
                device=device,
                dtype=TORCH_DTYPE,
            )
            synchronize_if_cuda(device)
            system_matrix_elapsed_seconds += perf_counter() - system_matrix_start_time

            transfer_start_time = perf_counter()
            batch_counts_flat = to_tensor(counts_array.ravel(), device)
            batch_backgrounds_flat = to_tensor(
                np.broadcast_to(assumed_backgrounds, counts_array.shape).astype(float).ravel(),
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
                    activity = apply_sieve_filter(
                        activity=activity,
                        candidate_flat_indices=candidate_grid["candidate_flat_indices"],
                        grid_shape=candidate_grid["grid_shape"],
                        voxel_size_cm=candidate_grid["voxel_size_cm"],
                        sieve_sigma_cm=sieve_sigma_cm,
                        device=device,
                    )

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
        },
        "timings": {
            "system_matrix_seconds": system_matrix_elapsed_seconds,
            "transfer_seconds": transfer_elapsed_seconds,
            "online_bsrem_iterations_seconds": iteration_elapsed_seconds,
            "bsrem_iterations_seconds": iteration_elapsed_seconds,
        },
    }
