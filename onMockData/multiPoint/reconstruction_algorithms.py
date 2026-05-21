"""
Reconstruction algorithms for multi-point source experiments.
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
    """Move numeric data to the Torch device used by ML-EM."""
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
    """Return the CUDA device used by ML-EM, or fail loudly if Torch cannot use CUDA."""
    if not torch.cuda.is_available():
        cuda_build = torch.version.cuda
        raise RuntimeError(
            "ML-EM is configured to run on CUDA, but PyTorch cannot see a CUDA device. "
            f"Installed torch version: {torch.__version__}; torch CUDA build: {cuda_build}. "
            "Install a CUDA-enabled PyTorch build and make sure the NVIDIA driver/GPU is visible."
        )
    return torch.device("cuda")


def reconstruct_sources(
    counts,
    geometry,
    rotations,
    translations,
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
    method="mlem",
    voxel_size_cm=1.0,
    num_iterations=40,
    history_interval=5,
    reconstruction_priors=None,
    reconstruction_sieve_sigma_cm=0.0,
    bsrem_num_subsets=8,
    bsrem_relaxation_initial=1.0,
    bsrem_relaxation_decay=0.01,
    bsrem_relaxation_power=1.0,
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
            trajectory_center=trajectory_center,
            search_radius=search_radius,
            cylinder_height=cylinder_height,
            voxel_size_cm=voxel_size_cm,
            num_iterations=num_iterations,
            history_interval=history_interval,
        )

    if method in ("map_em", "map"):
        return mapem_reconstruction(
            counts=counts,
            geometry=geometry,
            rotations=rotations,
            translations=translations,
            assumed_backgrounds=assumed_backgrounds,
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
            trajectory_center=trajectory_center,
            search_radius=search_radius,
            cylinder_height=cylinder_height,
            voxel_size_cm=voxel_size_cm,
            num_iterations=num_iterations,
            history_interval=history_interval,
            reconstruction_priors=reconstruction_priors,
            reconstruction_sieve_sigma_cm=reconstruction_sieve_sigma_cm,
        )

    if method == "bsrem":
        return bsrem_reconstruction(
            counts=counts,
            geometry=geometry,
            rotations=rotations,
            translations=translations,
            assumed_backgrounds=assumed_backgrounds,
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
            trajectory_center=trajectory_center,
            search_radius=search_radius,
            cylinder_height=cylinder_height,
            voxel_size_cm=voxel_size_cm,
            num_iterations=num_iterations,
            history_interval=history_interval,
            reconstruction_priors=reconstruction_priors,
            reconstruction_sieve_sigma_cm=reconstruction_sieve_sigma_cm,
            num_subsets=bsrem_num_subsets,
            relaxation_initial=bsrem_relaxation_initial,
            relaxation_decay=bsrem_relaxation_decay,
            relaxation_power=bsrem_relaxation_power,
        )

    raise ValueError(
        f"Unknown multi-source reconstruction method '{method}'. "
        "Use: mlem, map_em, bsrem."
    )


def make_candidate_grid(
    trajectory_center,
    search_radius,
    cylinder_height,
    voxel_size_cm,
):
    """Create candidate source positions inside the cylindrical search volume."""
    return make_candidate_grid_data(
        trajectory_center=trajectory_center,
        search_radius=search_radius,
        cylinder_height=cylinder_height,
        voxel_size_cm=voxel_size_cm,
    )["candidate_positions"]


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


def neighborhood_mean_activity(
    activity,
    candidate_flat_indices,
    grid_shape,
    device,
):
    """Return the six-neighbor mean activity for each candidate voxel."""
    flat_indices = torch.as_tensor(candidate_flat_indices, dtype=torch.long, device=device)
    num_grid_points = int(np.prod(grid_shape))
    activity_flat = torch.zeros(num_grid_points, dtype=TORCH_DTYPE, device=device)
    mask_flat = torch.zeros(num_grid_points, dtype=TORCH_DTYPE, device=device)
    activity_flat[flat_indices] = activity
    mask_flat[flat_indices] = 1.0

    volume = activity_flat.view(1, 1, *grid_shape)
    mask = mask_flat.view(1, 1, *grid_shape)
    kernel = torch.zeros((1, 1, 3, 3, 3), dtype=TORCH_DTYPE, device=device)
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0

    neighbor_sum = F.conv3d(volume, kernel, padding=1)
    neighbor_count = F.conv3d(mask, kernel, padding=1)
    neighbor_mean = neighbor_sum / torch.clamp(neighbor_count, min=1.0)
    mean_flat = neighbor_mean.reshape(-1)[flat_indices]
    has_neighbors = neighbor_count.reshape(-1)[flat_indices] > 0.0
    return torch.where(has_neighbors, mean_flat, activity)


def huber_pair_gradient(left, right, delta):
    """Return Huber prior gradients for one pair of neighboring voxels."""
    delta = float(delta)
    if delta <= 0.0:
        raise ValueError("Huber prior delta must be > 0.")

    difference = left - right
    left_gradient = difference / torch.sqrt(1.0 + (difference / delta) ** 2)
    return left_gradient, -left_gradient


def rdp_pair_gradient(left, right, gamma):
    """Return Relative Difference Penalty gradients for one neighboring voxel pair."""
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError("RDP prior gamma must be >= 0.")

    difference = left - right
    abs_difference = torch.abs(difference)
    sign_difference = torch.sign(difference)
    denominator = torch.clamp(left + right + gamma * abs_difference, min=EPS)
    squared_difference = difference * difference
    left_gradient = (
        2.0 * difference * denominator
        - squared_difference * (1.0 + gamma * sign_difference)
    ) / (denominator * denominator)
    right_gradient = (
        -2.0 * difference * denominator
        - squared_difference * (1.0 - gamma * sign_difference)
    ) / (denominator * denominator)
    return left_gradient, right_gradient


def pairwise_neighbor_prior_gradient(
    activity,
    candidate_flat_indices,
    grid_shape,
    device,
    pair_gradient_fn,
    **pair_gradient_kwargs,
):
    """Return pairwise six-neighbor prior gradient for candidate voxels."""
    flat_indices = torch.as_tensor(candidate_flat_indices, dtype=torch.long, device=device)
    num_grid_points = int(np.prod(grid_shape))
    activity_flat = torch.zeros(num_grid_points, dtype=TORCH_DTYPE, device=device)
    mask_flat = torch.zeros(num_grid_points, dtype=torch.bool, device=device)
    activity_flat[flat_indices] = activity
    mask_flat[flat_indices] = True

    volume = activity_flat.view(*grid_shape)
    mask = mask_flat.view(*grid_shape)
    gradient_volume = torch.zeros_like(volume)

    def add_edge_gradient(left_slices, right_slices):
        left_mask = mask[left_slices]
        right_mask = mask[right_slices]
        valid = left_mask & right_mask
        if not torch.any(valid):
            return

        left = volume[left_slices]
        right = volume[right_slices]
        left_gradient, right_gradient = pair_gradient_fn(
            left,
            right,
            **pair_gradient_kwargs,
        )

        gradient_left = gradient_volume[left_slices]
        gradient_right = gradient_volume[right_slices]
        gradient_left[valid] = gradient_left[valid] + left_gradient[valid]
        gradient_right[valid] = gradient_right[valid] + right_gradient[valid]

    add_edge_gradient((slice(0, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None)))
    add_edge_gradient((slice(None), slice(0, -1), slice(None)), (slice(None), slice(1, None), slice(None)))
    add_edge_gradient((slice(None), slice(None), slice(0, -1)), (slice(None), slice(None), slice(1, None)))

    return gradient_volume.reshape(-1)[flat_indices]


def huber_neighbor_prior_gradient(
    activity,
    candidate_flat_indices,
    grid_shape,
    device,
    delta=1.0,
):
    """Return six-neighbor Huber prior gradient for candidate voxels."""
    return pairwise_neighbor_prior_gradient(
        activity=activity,
        candidate_flat_indices=candidate_flat_indices,
        grid_shape=grid_shape,
        device=device,
        pair_gradient_fn=huber_pair_gradient,
        delta=delta,
    )


def rdp_neighbor_prior_gradient(
    activity,
    candidate_flat_indices,
    grid_shape,
    device,
    gamma=1.0,
):
    """Return six-neighbor RDP prior gradient for candidate voxels."""
    return pairwise_neighbor_prior_gradient(
        activity=activity,
        candidate_flat_indices=candidate_flat_indices,
        grid_shape=grid_shape,
        device=device,
        pair_gradient_fn=rdp_pair_gradient,
        gamma=gamma,
    )


def compute_reconstruction_prior_gradient(
    activity,
    candidate_positions,
    candidate_grid,
    reconstruction_priors,
    device,
):
    """Return summed one-step-late reconstruction prior gradients from config dictionaries."""
    if not reconstruction_priors:
        return torch.zeros_like(activity)

    candidate_positions = torch.as_tensor(candidate_positions, dtype=TORCH_DTYPE, device=device)
    gradient = torch.zeros_like(activity)
    for prior in reconstruction_priors:
        prior_type = str(prior.get("type", "")).strip().lower()

        beta = float(prior.get("beta", 0.0))
        if beta < 0.0:
            raise ValueError("Reconstruction prior beta must be >= 0.")
        if beta == 0.0:
            continue

        if prior_type in ("smoothness", "neighbor_smoothness"):
            neighbor_mean = neighborhood_mean_activity(
                activity=activity,
                candidate_flat_indices=candidate_grid["candidate_flat_indices"],
                grid_shape=candidate_grid["grid_shape"],
                device=device,
            )
            gradient = gradient + beta * (activity - neighbor_mean)
        elif prior_type in ("l1", "sparsity"):
            gradient = gradient + beta
        elif prior_type in ("huber", "huber_penalty"):
            gradient = gradient + beta * huber_neighbor_prior_gradient(
                activity=activity,
                candidate_flat_indices=candidate_grid["candidate_flat_indices"],
                grid_shape=candidate_grid["grid_shape"],
                device=device,
                delta=float(prior.get("delta", 1.0)),
            )
        elif prior_type in ("rdp", "relative_difference", "relative_difference_penalty"):
            gradient = gradient + beta * rdp_neighbor_prior_gradient(
                activity=activity,
                candidate_flat_indices=candidate_grid["candidate_flat_indices"],
                grid_shape=candidate_grid["grid_shape"],
                device=device,
                gamma=float(prior.get("gamma", 1.0)),
            )
        else:
            raise ValueError(
                f"Unknown reconstruction prior type {prior_type!r}. "
                "Use: smoothness, l1, huber, rdp."
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


def mlem_reconstruction(
    counts,
    geometry,
    rotations,
    translations,
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
    num_iterations=40,
    history_interval=5,
):
    """
    Reconstruct a voxel activity map with ML-EM.

    Model:
        counts ~= background + H @ activity

    The main ML-EM result is the full activity map.
    """
    candidate_grid = make_candidate_grid_data(
        trajectory_center=trajectory_center,
        search_radius=search_radius,
        cylinder_height=cylinder_height,
        voxel_size_cm=voxel_size_cm,
    )
    candidate_positions = candidate_grid["candidate_positions"]
    device = get_cuda_device()
    system_matrix_start_time = perf_counter()
    system_matrix_tensor = build_system_matrix_torch(  # system_matrix is H: rows are flattened measurement/channel pairs, columns are voxels.
        candidate_positions=candidate_positions,       # H[row, voxel] says how much one unit of activity in that voxel would add to that detector count for specific measurement and specific channel.
        geometry=geometry,
        rotations=rotations,
        translations=translations,
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
    system_matrix_elapsed_seconds = perf_counter() - system_matrix_start_time

    transfer_start_time = perf_counter()
    torch_device_name = str(device)
    counts_flat = to_tensor(np.asarray(counts, dtype=float).ravel(), device)
    backgrounds_flat = to_tensor(
        np.broadcast_to(assumed_backgrounds, counts.shape).astype(float).ravel(),
        device,
    )
    synchronize_if_cuda(device)
    transfer_elapsed_seconds = perf_counter() - transfer_start_time
    
    sensitivity = torch.sum(system_matrix_tensor, dim=0)
    # Sensitivity tells us how much each voxel contribute to detector response, not for each measurement, but in total

    initial_signal = torch.clamp(counts_flat - backgrounds_flat, min=0.0)
    initial_activity = tensor_to_float(
        torch.sum(initial_signal) / torch.clamp(torch.sum(sensitivity), min=EPS)
    )
    
    activity = torch.full(
        (len(candidate_positions),),
        max(initial_activity, EPS),
        dtype=TORCH_DTYPE,
        device=device,
    ) # Start from a flat activity map: every candidate voxel gets the same positive activity.
      # ML-EM will then move activity toward voxels whose H columns best explain the counts.

    history = []
    activity_history = []
    iteration_start_time = perf_counter()
    with torch.no_grad():
        for iteration in range(1, num_iterations + 1):
            
            predicted_counts = predict_counts_from_activity( # BACKGROUND + H @ ACTIVITY -> Based on each voxel's constibution to each measurement-channel pair and activity -> predicts counts for each measurement and each channel.
                backgrounds_flat=backgrounds_flat,
                system_matrix=system_matrix_tensor,
                activity=activity,
            )
            ratio = counts_flat / torch.clamp(predicted_counts, min=EPS) # Ratio tells how much error we have in each measurement-channel pair.
            denominator = sensitivity
            correction = (system_matrix_tensor.T @ ratio) / torch.clamp(denominator, min=EPS) # So crucial step. H tells us how much each voxel contribute to measurement-channel pair prediction and ratio tells us how bad prediction was.
                                                                                  # H.T @ ratio gives each voxel a response-weighted sum of measurement errors. Dividing by sensitivity normalizes it by total voxel response,
                                                                                  # so correction[voxel] becomes the H-weighted average ratio for that voxel, not just a larger sum for better-covered voxels.
            activity *= correction

            if iteration == 1 or iteration % history_interval == 0 or iteration == num_iterations:
                predicted_counts = predict_counts_from_activity(
                    backgrounds_flat=backgrounds_flat,
                    system_matrix=system_matrix_tensor,
                    activity=activity,
                )
                history.append(
                    {
                        "iteration": float(iteration),
                        "score": poisson_negative_log_likelihood_torch(
                            counts_flat,
                            predicted_counts,
                        ),
                        "total_activity": tensor_to_float(torch.sum(activity)),
                        "max_activity": tensor_to_float(torch.max(activity)),
                        "torch_device": torch_device_name,
                    }
                )
                activity_history.append(tensor_to_numpy(activity).copy())
    synchronize_if_cuda(device)
    iteration_elapsed_seconds = perf_counter() - iteration_start_time

    return {
        "candidate_positions": candidate_positions,
        "candidate_flat_indices": candidate_grid["candidate_flat_indices"].copy(),
        "grid_shape": tuple(candidate_grid["grid_shape"]),
        "voxel_size_cm": float(candidate_grid["voxel_size_cm"]),
        "activity": tensor_to_numpy(activity).copy(),
        "activity_history": [snapshot.copy() for snapshot in activity_history],
        "history": history,
        "system_matrix": tensor_to_numpy(system_matrix_tensor).copy(),
        "torch_device": torch_device_name,
        "timings": {
            "system_matrix_seconds": system_matrix_elapsed_seconds,
            "transfer_seconds": transfer_elapsed_seconds,
            "mlem_iterations_seconds": iteration_elapsed_seconds,
        },
    }


def mapem_reconstruction(
    counts,
    geometry,
    rotations,
    translations,
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
    num_iterations=40,
    history_interval=5,
    reconstruction_priors=None,
    reconstruction_sieve_sigma_cm=0.0,
):
    """
    Reconstruct a voxel activity map with MAP-EM.

    The likelihood term is the same Poisson model as ML-EM. Reconstruction
    priors are supplied as config dictionaries and enter the one-step-late
    denominator.
    """
    candidate_grid = make_candidate_grid_data(
        trajectory_center=trajectory_center,
        search_radius=search_radius,
        cylinder_height=cylinder_height,
        voxel_size_cm=voxel_size_cm,
    )
    candidate_positions = candidate_grid["candidate_positions"]
    device = get_cuda_device()
    system_matrix_start_time = perf_counter()
    system_matrix_tensor = build_system_matrix_torch(
        candidate_positions=candidate_positions,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
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
    system_matrix_elapsed_seconds = perf_counter() - system_matrix_start_time

    transfer_start_time = perf_counter()
    torch_device_name = str(device)
    counts_flat = to_tensor(np.asarray(counts, dtype=float).ravel(), device)
    backgrounds_flat = to_tensor(
        np.broadcast_to(assumed_backgrounds, counts.shape).astype(float).ravel(),
        device,
    )
    synchronize_if_cuda(device)
    transfer_elapsed_seconds = perf_counter() - transfer_start_time

    reconstruction_priors = list(reconstruction_priors or [])
    sieve_sigma_cm = float(reconstruction_sieve_sigma_cm)

    sensitivity = torch.sum(system_matrix_tensor, dim=0)
    initial_signal = torch.clamp(counts_flat - backgrounds_flat, min=0.0)
    initial_activity = tensor_to_float(
        torch.sum(initial_signal) / torch.clamp(torch.sum(sensitivity), min=EPS)
    )
    activity = torch.full(
        (len(candidate_positions),),
        max(initial_activity, EPS),
        dtype=TORCH_DTYPE,
        device=device,
    )

    history = []
    activity_history = []
    iteration_start_time = perf_counter()
    with torch.no_grad():
        for iteration in range(1, num_iterations + 1):
            predicted_counts = predict_counts_from_activity(
                backgrounds_flat=backgrounds_flat,
                system_matrix=system_matrix_tensor,
                activity=activity,
            )
            ratio = counts_flat / torch.clamp(predicted_counts, min=EPS)
            prior_gradient = compute_reconstruction_prior_gradient(
                activity=activity,
                candidate_positions=candidate_positions,
                candidate_grid=candidate_grid,
                reconstruction_priors=reconstruction_priors,
                device=device,
            )
            denominator = sensitivity + prior_gradient
            # Smoothness priors can have a negative local gradient. If that
            # nearly cancels the tiny H sensitivity, the EM denominator collapses
            # and activity explodes. Keep the prior influence bounded relative
            # to the data sensitivity.
            denominator_floor = torch.clamp(0.25 * sensitivity, min=EPS)
            denominator = torch.maximum(denominator, denominator_floor)
            correction = (system_matrix_tensor.T @ ratio) / denominator
            activity *= correction
            activity = apply_sieve_filter(
                activity=activity,
                candidate_flat_indices=candidate_grid["candidate_flat_indices"],
                grid_shape=candidate_grid["grid_shape"],
                voxel_size_cm=candidate_grid["voxel_size_cm"],
                sieve_sigma_cm=sieve_sigma_cm,
                device=device,
            )

            if iteration == 1 or iteration % history_interval == 0 or iteration == num_iterations:
                predicted_counts = predict_counts_from_activity(
                    backgrounds_flat=backgrounds_flat,
                    system_matrix=system_matrix_tensor,
                    activity=activity,
                )
                history.append(
                    {
                        "iteration": float(iteration),
                        "score": poisson_negative_log_likelihood_torch(
                            counts_flat,
                            predicted_counts,
                        ),
                        "total_activity": tensor_to_float(torch.sum(activity)),
                        "max_activity": tensor_to_float(torch.max(activity)),
                        "reconstruction_priors": [dict(prior) for prior in reconstruction_priors],
                        "torch_device": torch_device_name,
                    }
                )
                activity_history.append(tensor_to_numpy(activity).copy())
    synchronize_if_cuda(device)
    iteration_elapsed_seconds = perf_counter() - iteration_start_time

    return {
        "candidate_positions": candidate_positions,
        "candidate_flat_indices": candidate_grid["candidate_flat_indices"].copy(),
        "grid_shape": tuple(candidate_grid["grid_shape"]),
        "voxel_size_cm": float(candidate_grid["voxel_size_cm"]),
        "activity": tensor_to_numpy(activity).copy(),
        "activity_history": [snapshot.copy() for snapshot in activity_history],
        "history": history,
        "system_matrix": tensor_to_numpy(system_matrix_tensor).copy(),
        "torch_device": torch_device_name,
        "timings": {
            "system_matrix_seconds": system_matrix_elapsed_seconds,
            "transfer_seconds": transfer_elapsed_seconds,
            "mapem_iterations_seconds": iteration_elapsed_seconds,
        },
    }


def bsrem_reconstruction(
    counts,
    geometry,
    rotations,
    translations,
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
    num_iterations=40,
    history_interval=5,
    reconstruction_priors=None,
    reconstruction_sieve_sigma_cm=0.0,
    num_subsets=8,
    relaxation_initial=1.0,
    relaxation_decay=0.01,
    relaxation_power=1.0,
    max_relative_update=2.0,
):
    """
    Reconstruct a voxel activity map with BSREM.

    BSREM uses ordered subsets like OSEM, but applies a relaxation schedule and
    the same penalized-likelihood prior gradient used by MAP-EM.
    """
    candidate_grid = make_candidate_grid_data(
        trajectory_center=trajectory_center,
        search_radius=search_radius,
        cylinder_height=cylinder_height,
        voxel_size_cm=voxel_size_cm,
    )
    candidate_positions = candidate_grid["candidate_positions"]
    device = get_cuda_device()
    system_matrix_start_time = perf_counter()
    system_matrix_tensor = build_system_matrix_torch(
        candidate_positions=candidate_positions,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
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
    system_matrix_elapsed_seconds = perf_counter() - system_matrix_start_time

    transfer_start_time = perf_counter()
    torch_device_name = str(device)
    counts_array = np.asarray(counts, dtype=float)
    counts_flat = to_tensor(counts_array.ravel(), device)
    backgrounds_flat = to_tensor(
        np.broadcast_to(assumed_backgrounds, counts_array.shape).astype(float).ravel(),
        device,
    )
    subsets = make_measurement_row_subsets(
        counts_shape=counts_array.shape,
        num_subsets=num_subsets,
        device=device,
    )
    synchronize_if_cuda(device)
    transfer_elapsed_seconds = perf_counter() - transfer_start_time

    reconstruction_priors = list(reconstruction_priors or [])
    sieve_sigma_cm = float(reconstruction_sieve_sigma_cm)

    sensitivity = torch.sum(system_matrix_tensor, dim=0)
    initial_signal = torch.clamp(counts_flat - backgrounds_flat, min=0.0)
    initial_activity = tensor_to_float(
        torch.sum(initial_signal) / torch.clamp(torch.sum(sensitivity), min=EPS)
    )
    activity = torch.full(
        (len(candidate_positions),),
        max(initial_activity, EPS),
        dtype=TORCH_DTYPE,
        device=device,
    )

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

    full_measurement_count = int(counts_array.shape[0])
    history = []
    activity_history = []
    iteration_start_time = perf_counter()
    with torch.no_grad():
        for iteration in range(1, num_iterations + 1):
            relaxation = relaxation_initial / (
                (1.0 + relaxation_decay * float(iteration - 1)) ** relaxation_power
            )
            for subset in subsets:
                row_indices = subset["row_indices"]
                subset_matrix = system_matrix_tensor[row_indices, :]
                subset_counts = counts_flat[row_indices]
                subset_backgrounds = backgrounds_flat[row_indices]
                predicted_counts = predict_counts_from_activity(
                    backgrounds_flat=subset_backgrounds,
                    system_matrix=subset_matrix,
                    activity=activity,
                )
                ratio = subset_counts / torch.clamp(predicted_counts, min=EPS)
                subset_scale = full_measurement_count / max(
                    int(subset["measurement_count"]),
                    1,
                )
                data_gradient = subset_scale * (subset_matrix.T @ (ratio - 1.0))
                prior_gradient = compute_reconstruction_prior_gradient(
                    activity=activity,
                    candidate_positions=candidate_positions,
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

            if iteration == 1 or iteration % history_interval == 0 or iteration == num_iterations:
                predicted_counts = predict_counts_from_activity(
                    backgrounds_flat=backgrounds_flat,
                    system_matrix=system_matrix_tensor,
                    activity=activity,
                )
                history.append(
                    {
                        "iteration": float(iteration),
                        "score": poisson_negative_log_likelihood_torch(
                            counts_flat,
                            predicted_counts,
                        ),
                        "total_activity": tensor_to_float(torch.sum(activity)),
                        "max_activity": tensor_to_float(torch.max(activity)),
                        "reconstruction_priors": [dict(prior) for prior in reconstruction_priors],
                        "bsrem_num_subsets": int(len(subsets)),
                        "bsrem_relaxation": float(relaxation),
                        "torch_device": torch_device_name,
                    }
                )
                activity_history.append(tensor_to_numpy(activity).copy())
    synchronize_if_cuda(device)
    iteration_elapsed_seconds = perf_counter() - iteration_start_time

    return {
        "candidate_positions": candidate_positions,
        "candidate_flat_indices": candidate_grid["candidate_flat_indices"].copy(),
        "grid_shape": tuple(candidate_grid["grid_shape"]),
        "voxel_size_cm": float(candidate_grid["voxel_size_cm"]),
        "activity": tensor_to_numpy(activity).copy(),
        "activity_history": [snapshot.copy() for snapshot in activity_history],
        "history": history,
        "system_matrix": tensor_to_numpy(system_matrix_tensor).copy(),
        "torch_device": torch_device_name,
        "timings": {
            "system_matrix_seconds": system_matrix_elapsed_seconds,
            "transfer_seconds": transfer_elapsed_seconds,
            "bsrem_iterations_seconds": iteration_elapsed_seconds,
        },
    }
