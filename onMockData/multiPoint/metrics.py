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

    print(" selected voxel -> true source distances [cm]:")
    print(
        "  min/mean/wmean/median/p90/max: "
        f"{summary['all_source_distance_min']:.4f} / "
        f"{summary['all_source_distance_mean']:.4f} / "
        f"{summary['all_source_distance_weighted_mean']:.4f} / "
        f"{summary['all_source_distance_median']:.4f} / "
        f"{summary['all_source_distance_p90']:.4f} / "
        f"{summary['all_source_distance_max']:.4f}"
    )

    print(" per true source distances from selected voxels [cm]:")
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


def summarize_estimated_source_distances(true_sources, estimated_sources):
    """Greedily match every true source to the nearest unused estimated peak."""
    true_positions = _true_source_positions(true_sources)
    estimated_positions = np.array(
        [source["position"] for source in estimated_sources],
        dtype=float,
    )

    if len(true_positions) == 0:
        raise ValueError("true_sources must not be empty.")

    if len(estimated_positions) == 0:
        return {
            "true_source_count": int(len(true_positions)),
            "estimated_source_count": 0,
            "matched_count": 0,
            "per_true_source": [],
            "unmatched_true_indices": list(range(len(true_positions))),
            "unmatched_estimated_indices": [],
            "total_true_to_estimated_distance": np.nan,
            "distance_min": np.nan,
            "distance_mean": np.nan,
            "distance_max": np.nan,
        }

    deltas = true_positions[:, None, :] - estimated_positions[None, :, :]
    distance_matrix = np.sqrt(np.sum(deltas * deltas, axis=2))
    true_count, estimated_count = distance_matrix.shape
    per_true_source = []
    matched_true_indices = set()
    matched_estimated_indices = set()
    available_estimated_indices = list(range(estimated_count))

    for true_index in range(true_count):
        if not available_estimated_indices:
            break

        available_distances = distance_matrix[true_index, available_estimated_indices]
        nearest_available_index = int(np.argmin(available_distances))
        estimated_index = available_estimated_indices.pop(nearest_available_index)

        matched_true_indices.add(true_index)
        matched_estimated_indices.add(estimated_index)
        estimated_source = estimated_sources[estimated_index]
        per_true_source.append(
            {
                "true_index": int(true_index),
                "estimated_index": int(estimated_index),
                "true_position": true_positions[true_index].copy(),
                "estimated_position": estimated_positions[estimated_index].copy(),
                "estimated_activity": float(estimated_source["activity"]),
                "distance": float(distance_matrix[true_index, estimated_index]),
            }
        )

    distances = np.array([item["distance"] for item in per_true_source], dtype=float)
    return {
        "true_source_count": int(true_count),
        "estimated_source_count": int(estimated_count),
        "matched_count": int(len(per_true_source)),
        "per_true_source": per_true_source,
        "unmatched_true_indices": [
            index for index in range(true_count) if index not in matched_true_indices
        ],
        "unmatched_estimated_indices": [
            index for index in range(estimated_count) if index not in matched_estimated_indices
        ],
        "total_true_to_estimated_distance": float(np.sum(distances)),
        "distance_min": float(np.min(distances)),
        "distance_mean": float(np.mean(distances)),
        "distance_max": float(np.max(distances)),
    }


def print_estimated_source_distance_summary(summary):
    """Print nearest estimated peak distance for every true source."""
    print()
    print("Estimated peak -> true source distances:")
    print(
        " compared sources:          "
        f"{summary['true_source_count']} true, "
        f"{summary['estimated_source_count']} estimated"
    )
    print(f" unique matched pairs:       {summary['matched_count']}")

    if not summary["per_true_source"]:
        print(" no estimated peaks to compare")
        return

    if summary["true_source_count"] == 1:
        item = summary["per_true_source"][0]
        estimated_number = item["estimated_index"] + 1
        print(
            f"  S1 -> nearest P{estimated_number}: "
            f"distance={item['distance']:.4f} cm, "
            f"true={np.round(item['true_position'], 4).tolist()}, "
            f"estimated={np.round(item['estimated_position'], 4).tolist()}, "
            f"activity={item['estimated_activity']:.2f}"
        )

    print(
        " total true->estimate distance [cm]: "
        f"{summary['total_true_to_estimated_distance']:.4f}"
    )
    if summary["true_source_count"] == 1:
        return

    print(" multiple true sources: only total unique-match distance is shown")
    if summary["unmatched_true_indices"]:
        labels = [f"S{index + 1}" for index in summary["unmatched_true_indices"]]
        print(f" unmatched true sources:    {', '.join(labels)}")
    if summary["unmatched_estimated_indices"]:
        labels = [f"P{index + 1}" for index in summary["unmatched_estimated_indices"]]
        print(f" unmatched estimated peaks: {', '.join(labels)}")
