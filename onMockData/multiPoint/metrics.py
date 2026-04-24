"""
Metrics for evaluating multi-point activity reconstructions.
"""

import numpy as np

from common.geometry import EPS


def _true_source_positions(true_sources):
    return np.array([source["position"] for source in true_sources], dtype=float)


def summarize_activity_localization(
    voxel_positions,
    activity,
    true_sources,
    max_relative_drop=0.25,
):
    """
    Summarize how close the strongest activity voxels are to true sources.

    max_relative_drop=0.25 selects voxels with activity >= 75% of the best voxel.
    """
    voxel_positions = np.asarray(voxel_positions, dtype=float)
    activity = np.asarray(activity, dtype=float)
    true_positions = _true_source_positions(true_sources)

    if len(voxel_positions) != len(activity):
        raise ValueError("voxel_positions and activity must have the same length.")
    if len(activity) == 0:
        raise ValueError("activity must not be empty.")
    if len(true_positions) == 0:
        raise ValueError("true_sources must not be empty.")

    max_relative_drop = float(max_relative_drop)
    if not 0.0 <= max_relative_drop <= 1.0:
        raise ValueError("max_relative_drop must be between 0 and 1.")

    max_activity = float(np.max(activity))
    if max_activity <= EPS:
        return {
            "selection": f"within {100.0 * max_relative_drop:.1f}% of max",
            "selected_voxel_count": 0,
            "total_voxel_count": int(len(activity)),
            "max_activity": max_activity,
            "activity_threshold": 0.0,
            "selected_activity_fraction": 0.0,
            "selected_voxel_positions": np.empty((0, 3), dtype=float),
            "selected_voxel_activity": np.empty(0, dtype=float),
            "true_source_positions": true_positions,
            "voxel_to_source_distances": np.empty((0, len(true_positions)), dtype=float),
            "all_source_distance_min": np.nan,
            "all_source_distance_mean": np.nan,
            "all_source_distance_weighted_mean": np.nan,
            "all_source_distance_median": np.nan,
            "all_source_distance_p90": np.nan,
            "all_source_distance_max": np.nan,
            "per_source_distances": [],
        }

    activity_threshold = (1.0 - max_relative_drop) * max_activity
    selected_mask = activity >= activity_threshold
    selected_positions = voxel_positions[selected_mask]
    selected_activity = activity[selected_mask]

    deltas = selected_positions[:, None, :] - true_positions[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    flat_distances = distances.ravel()

    selected_activity_sum = float(np.sum(selected_activity))
    total_activity_sum = float(np.sum(activity))
    distance_weights = np.repeat(np.maximum(selected_activity, EPS), len(true_positions))
    weighted_mean = float(
        np.average(flat_distances, weights=distance_weights)
    )

    per_source_distances = []
    for source_index in range(len(true_positions)):
        source_distances = distances[:, source_index]
        per_source_distances.append(
            {
                "source_index": int(source_index),
                "distance_min": float(np.min(source_distances)),
                "distance_mean": float(np.mean(source_distances)),
                "distance_weighted_mean": float(
                    np.average(source_distances, weights=np.maximum(selected_activity, EPS))
                ),
                "distance_median": float(np.median(source_distances)),
                "distance_p90": float(np.percentile(source_distances, 90.0)),
                "distance_max": float(np.max(source_distances)),
            }
        )

    return {
        "selection": f"within {100.0 * max_relative_drop:.1f}% of max",
        "selected_voxel_count": int(len(selected_positions)),
        "total_voxel_count": int(len(activity)),
        "max_activity": max_activity,
        "activity_threshold": float(activity_threshold),
        "selected_activity_fraction": selected_activity_sum / max(total_activity_sum, EPS),
        "selected_voxel_positions": selected_positions.copy(),
        "selected_voxel_activity": selected_activity.copy(),
        "true_source_positions": true_positions.copy(),
        "voxel_to_source_distances": distances.copy(),
        "all_source_distance_min": float(np.min(flat_distances)),
        "all_source_distance_mean": float(np.mean(flat_distances)),
        "all_source_distance_weighted_mean": weighted_mean,
        "all_source_distance_median": float(np.median(flat_distances)),
        "all_source_distance_p90": float(np.percentile(flat_distances, 90.0)),
        "all_source_distance_max": float(np.max(flat_distances)),
        "per_source_distances": per_source_distances,
    }


def print_activity_localization_summary(summary):
    """Print a compact activity localization metric summary."""
    print()
    print("Activity localization metrics:")
    print(f" selection:                 {summary['selection']}")
    print(
        " selected voxels:           "
        f"{summary['selected_voxel_count']} / {summary['total_voxel_count']}"
    )
    print(
        " max/threshold activity:    "
        f"{summary['max_activity']:.4f} / {summary['activity_threshold']:.4f}"
    )
    print(
        " selected activity fraction:"
        f" {100.0 * summary['selected_activity_fraction']:.2f}%"
    )

    if summary["selected_voxel_count"] <= 0:
        print(" no active voxels selected")
        return

    print(" selected voxel -> true source distances [m]:")
    print(
        "  min/mean/wmean/median/p90/max: "
        f"{summary['all_source_distance_min']:.4f} / "
        f"{summary['all_source_distance_mean']:.4f} / "
        f"{summary['all_source_distance_weighted_mean']:.4f} / "
        f"{summary['all_source_distance_median']:.4f} / "
        f"{summary['all_source_distance_p90']:.4f} / "
        f"{summary['all_source_distance_max']:.4f}"
    )

    print(" per true source distances from selected voxels [m]:")
    for source_distances in summary["per_source_distances"]:
        source_number = source_distances["source_index"] + 1
        print(
            f"  S{source_number}: "
            f"min={source_distances['distance_min']:.4f}, "
            f"mean={source_distances['distance_mean']:.4f}, "
            f"wmean={source_distances['distance_weighted_mean']:.4f}, "
            f"p90={source_distances['distance_p90']:.4f}, "
            f"max={source_distances['distance_max']:.4f}"
        )
