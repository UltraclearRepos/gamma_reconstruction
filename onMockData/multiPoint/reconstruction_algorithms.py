"""
Reconstruction algorithms for multi-point source experiments.
"""

from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from common.geometry import EPS, points_in_detector_box
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


def compute_measurement_coverage_mask(
    system_matrix,
    num_measurements,
    num_channels,
    min_measurements,
    relative_threshold,
):
    """
    Return voxels seen by enough distinct detector positions.

    A measurement is counted for a voxel when its summed channel response is at
    least relative_threshold * max_measurement_response_for_that_voxel.
    """
    min_measurements = int(min_measurements)
    if min_measurements <= 1:
        keep = torch.ones(system_matrix.shape[1], dtype=torch.bool, device=system_matrix.device)
        counts = torch.ones(system_matrix.shape[1], dtype=torch.int64, device=system_matrix.device)
        return keep, counts

    relative_threshold = float(relative_threshold)
    if not 0.0 <= relative_threshold <= 1.0:
        raise ValueError("coverage_relative_threshold must be in the range [0, 1].")

    measurement_response = system_matrix.reshape(num_measurements, num_channels, -1).sum(dim=1)
    max_response = torch.max(measurement_response, dim=0).values
    threshold = relative_threshold * max_response
    covered = (measurement_response >= threshold[None, :]) & (max_response[None, :] > EPS)
    coverage_counts = torch.sum(covered, dim=0)
    return coverage_counts >= min_measurements, coverage_counts


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
    sieve_sigma_cm=0.0,
    min_coverage_measurements=1,
    coverage_relative_threshold=0.05,
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
            assumed_collimator_half_angle_deg=assumed_collimator_half_angle_deg,
            assumed_collimator_edge_width_deg=assumed_collimator_edge_width_deg,
            assumed_collimator_leakage=assumed_collimator_leakage,
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
            voxel_size_cm=voxel_size_cm,
            num_iterations=num_iterations,
            history_interval=history_interval,
            sieve_sigma_cm=sieve_sigma_cm,
            min_coverage_measurements=min_coverage_measurements,
            coverage_relative_threshold=coverage_relative_threshold,
        )

    raise ValueError(
        f"Unknown multi-source reconstruction method '{method}'. "
        "Use: mlem."
    )


def make_candidate_grid(
    search_min,
    search_max,
    voxel_size_cm,
    geometry,
    rotations,
    translations,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
):
    """Create candidate source positions inside the detector-facing rectangular boxes."""
    return make_candidate_grid_data(
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
    )["candidate_positions"]


def make_candidate_grid_data(
    search_min,
    search_max,
    voxel_size_cm,
    geometry,
    rotations,
    translations,
    search_start_distance,
    search_stop_distance,
    search_width,
    search_height,
):
    """Create candidate source positions and regular-grid metadata for 3D filters."""
    search_min = np.array(search_min, dtype=float)
    search_max = np.array(search_max, dtype=float)
    voxel_size_cm = float(voxel_size_cm)
    if voxel_size_cm <= 0.0:
        raise ValueError("voxel_size_cm must be > 0.")

    xs = np.arange(search_min[0], search_max[0] + 0.5 * voxel_size_cm, voxel_size_cm)
    ys = np.arange(search_min[1], search_max[1] + 0.5 * voxel_size_cm, voxel_size_cm)
    zs = np.arange(search_min[2], search_max[2] + 0.5 * voxel_size_cm, voxel_size_cm)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    all_positions = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    inside = points_in_detector_box(
        points=all_positions,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        search_start_distance=search_start_distance,
        search_stop_distance=search_stop_distance,
        search_width=search_width,
        search_height=search_height,
    )
    candidate_positions = all_positions[inside]
    if len(candidate_positions) == 0:
        raise ValueError("No candidate voxels are inside the detector search boxes.")
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
    sieve_sigma_cm=0.0,
    min_coverage_measurements=1,
    coverage_relative_threshold=0.05,
):
    """
    Reconstruct a voxel activity map with ML-EM.

    Model:
        counts ~= background + H @ activity

    The main ML-EM result is the full activity map.
    """
    candidate_grid = make_candidate_grid_data(
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
    candidate_positions = candidate_grid["candidate_positions"]
    device = get_cuda_device()
    system_matrix_start_time = perf_counter()
    system_matrix_tensor = build_system_matrix_torch(  # system_matrix is H: rows are flattened measurement/channel pairs, columns are voxels.
        candidate_positions=candidate_positions,       # H[row, voxel] says how much one unit of activity in that voxel would add to that detector count for specific measurement and specific channel.
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        assumed_channel_scales=assumed_channel_scales,
        assumed_k=assumed_k,
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

    num_measurements, num_channels = counts.shape
    coverage_mask, coverage_counts = compute_measurement_coverage_mask(
        system_matrix=system_matrix_tensor,
        num_measurements=num_measurements,
        num_channels=num_channels,
        min_measurements=min_coverage_measurements,
        relative_threshold=coverage_relative_threshold,
    )
    kept_voxels = int(torch.sum(coverage_mask).detach().cpu().item())
    total_voxels_before_coverage = int(system_matrix_tensor.shape[1])
    if kept_voxels == 0:
        raise ValueError(
            "Coverage mask removed all candidate voxels. Lower "
            "min_coverage_measurements or coverage_relative_threshold."
        )
    coverage_mask_numpy = tensor_to_numpy(coverage_mask).astype(bool)
    candidate_positions = candidate_positions[coverage_mask_numpy]
    candidate_grid["candidate_flat_indices"] = candidate_grid["candidate_flat_indices"][
        coverage_mask_numpy
    ]
    coverage_counts = coverage_counts[coverage_mask]
    system_matrix_tensor = system_matrix_tensor[:, coverage_mask]
    coverage_counts_numpy = tensor_to_numpy(coverage_counts).astype(int)
    
    transfer_start_time = perf_counter()
    torch_device_name = str(device)
    counts_flat = to_tensor(np.asarray(counts, dtype=float).ravel(), device)
    backgrounds_flat = to_tensor(
        np.broadcast_to(assumed_backgrounds, counts.shape).astype(float).ravel(),
        device,
    )
    synchronize_if_cuda(device)
    transfer_elapsed_seconds = perf_counter() - transfer_start_time
    
    sparsity_lambda = float(sparsity_lambda)
    if sparsity_lambda < 0.0:
        raise ValueError("sparsity_lambda must be >= 0.")

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
            correction = (system_matrix_tensor.T @ ratio) / torch.clamp(sensitivity + sparsity_lambda, min=EPS) # So crucial step. H tells us how much each voxel contribute to measurement-channel pair prediction and ratio tells us how bad prediction was.
                                                                                  # H.T @ ratio gives each voxel a response-weighted sum of measurement errors. Dividing by sensitivity normalizes it by total voxel response,
                                                                                  # so correction[voxel] becomes the H-weighted average ratio for that voxel, not just a larger sum for better-covered voxels.
                                                                                  # sparsity_lambda adds an L1-like penalty that makes weak diffuse activity shrink faster.
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
        "torch_device": torch_device_name,
        "coverage": {
            "min_measurements": int(min_coverage_measurements),
            "relative_threshold": float(coverage_relative_threshold),
            "total_voxels_before": total_voxels_before_coverage,
            "kept_voxels": int(kept_voxels),
            "removed_voxels": int(total_voxels_before_coverage - kept_voxels),
            "removed_voxel_fraction": (
                float(total_voxels_before_coverage - kept_voxels)
                / max(float(total_voxels_before_coverage), 1.0)
            ),
            "coverage_count_min": int(np.min(coverage_counts_numpy)),
            "coverage_count_mean": float(np.mean(coverage_counts_numpy)),
            "coverage_count_max": int(np.max(coverage_counts_numpy)),
        },
        "timings": {
            "system_matrix_seconds": system_matrix_elapsed_seconds,
            "transfer_seconds": transfer_elapsed_seconds,
            "mlem_iterations_seconds": iteration_elapsed_seconds,
        },
    }
