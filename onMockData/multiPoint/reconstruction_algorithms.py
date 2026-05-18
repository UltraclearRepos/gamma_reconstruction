"""
Reconstruction algorithms for multi-point source experiments.
"""

from time import perf_counter

import numpy as np
import torch

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
        "activity": tensor_to_numpy(activity).copy(),
        "activity_history": [snapshot.copy() for snapshot in activity_history],
        "history": history,
        "torch_device": torch_device_name,
        "timings": {
            "system_matrix_seconds": system_matrix_elapsed_seconds,
            "transfer_seconds": transfer_elapsed_seconds,
            "mlem_iterations_seconds": iteration_elapsed_seconds,
        },
    }
