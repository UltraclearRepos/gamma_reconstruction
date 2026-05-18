"""
Metrics for evaluating multi-point activity reconstructions.
"""

import numpy as np

from common.geometry import EPS


def summarize_activity_map_similarity(true_activity, reconstructed_activity, threshold_fraction=0.75):
    """
    Compare true and reconstructed activity as voxel activity distributions.

    Both vectors must live on the same candidate voxel grid. Metrics that compare
    shape are computed after normalizing each map to unit total activity, so they
    are not dominated by an arbitrary global activity scale.
    """
    true_activity = np.asarray(true_activity, dtype=float)
    reconstructed_activity = np.asarray(reconstructed_activity, dtype=float)
    if true_activity.shape != reconstructed_activity.shape:
        raise ValueError("true_activity and reconstructed_activity must have the same shape.")
    if true_activity.ndim != 1:
        raise ValueError("activity maps must be one-dimensional vectors.")
    if len(true_activity) == 0:
        raise ValueError("activity maps must not be empty.")

    true_activity = np.maximum(true_activity, 0.0)
    reconstructed_activity = np.maximum(reconstructed_activity, 0.0)
    true_total = float(np.sum(true_activity))
    reconstructed_total = float(np.sum(reconstructed_activity))
    threshold_fraction = float(threshold_fraction)
    if not 0.0 <= threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be between 0 and 1.")

    if true_total <= EPS or reconstructed_total <= EPS:
        return {
            "true_total_activity": true_total,
            "reconstructed_total_activity": reconstructed_total,
            "scale_to_true_total": np.nan,
            "cosine_similarity": np.nan,
            "pearson_correlation": np.nan,
            "normalized_l1_distance": np.nan,
            "threshold_fraction": threshold_fraction,
            "true_active_voxel_count": 0,
            "reconstructed_active_voxel_count": 0,
            "active_mask_overlap": np.nan,
            "active_voxel_recall": np.nan,
            "active_voxel_precision": np.nan,
        }

    true_distribution = true_activity / true_total
    reconstructed_distribution = reconstructed_activity / reconstructed_total
    dot_product = float(np.dot(true_activity, reconstructed_activity))
    norm_product = float(np.linalg.norm(true_activity) * np.linalg.norm(reconstructed_activity))
    cosine_similarity = dot_product / max(norm_product, EPS)

    true_centered = true_activity - float(np.mean(true_activity))
    reconstructed_centered = reconstructed_activity - float(np.mean(reconstructed_activity))
    centered_norm_product = float(
        np.linalg.norm(true_centered) * np.linalg.norm(reconstructed_centered)
    )
    pearson_correlation = float(
        np.dot(true_centered, reconstructed_centered) / max(centered_norm_product, EPS)
    )

    normalized_difference = true_distribution - reconstructed_distribution
    normalized_l1_distance = float(np.sum(np.abs(normalized_difference)))

    true_threshold = threshold_fraction * float(np.max(true_activity))
    reconstructed_threshold = threshold_fraction * float(np.max(reconstructed_activity))
    true_mask = true_activity >= true_threshold
    reconstructed_mask = reconstructed_activity >= reconstructed_threshold
    intersection = int(np.count_nonzero(true_mask & reconstructed_mask))
    union = int(np.count_nonzero(true_mask | reconstructed_mask))
    true_active_count = int(np.count_nonzero(true_mask))
    reconstructed_active_count = int(np.count_nonzero(reconstructed_mask))

    return {
        "true_total_activity": true_total,
        "reconstructed_total_activity": reconstructed_total,
        "scale_to_true_total": true_total / max(reconstructed_total, EPS),
        "cosine_similarity": float(cosine_similarity),
        "pearson_correlation": pearson_correlation,
        "normalized_l1_distance": normalized_l1_distance,
        "threshold_fraction": threshold_fraction,
        "true_active_voxel_count": true_active_count,
        "reconstructed_active_voxel_count": reconstructed_active_count,
        "active_mask_overlap": intersection / max(union, 1),
        "active_voxel_recall": intersection / max(true_active_count, 1),
        "active_voxel_precision": intersection / max(reconstructed_active_count, 1),
    }


def print_activity_map_similarity_summary(summary):
    """Print activity-map similarity metrics."""
    print()
    print("Activity map similarity metrics:")
    print(
        " total activity true/recon: "
        f"{summary['true_total_activity']:.4f} / "
        f"{summary['reconstructed_total_activity']:.4f}"
    )
    print(f" scale recon -> true total: {summary['scale_to_true_total']:.6f}")
    print(
        " shape cosine / pearson:    "
        f"{summary['cosine_similarity']:.4f} / "
        f"{summary['pearson_correlation']:.4f}"
    )
    print(
        " active mask voxels true/recon: "
        f"{summary['true_active_voxel_count']} / "
        f"{summary['reconstructed_active_voxel_count']} "
        f"(threshold {100.0 * summary['threshold_fraction']:.1f}% of each max)"
    )
    print(
        " active mask overlap/recall/precision: "
        f"{summary['active_mask_overlap']:.4f} / "
        f"{summary['active_voxel_recall']:.4f} / "
        f"{summary['active_voxel_precision']:.4f}"
    )
