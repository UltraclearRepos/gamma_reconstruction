"""
Metrics for evaluating multi-point activity reconstructions.
"""

from collections import deque

import numpy as np

from common.geometry import EPS


NEIGHBOR_OFFSETS_3D = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
)


def summarize_activity_map_similarity(true_activity, reconstructed_activity):
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

    if true_total <= EPS or reconstructed_total <= EPS:
        return {
            "true_total_activity": true_total,
            "reconstructed_total_activity": reconstructed_total,
            "scale_to_true_total": np.nan,
            "cosine_similarity": np.nan,
            "pearson_correlation": np.nan,
        }

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

    return {
        "true_total_activity": true_total,
        "reconstructed_total_activity": reconstructed_total,
        "scale_to_true_total": true_total / max(reconstructed_total, EPS),
        "cosine_similarity": float(cosine_similarity),
        "pearson_correlation": pearson_correlation,
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


def find_activity_components(
    activity,
    candidate_positions,
    candidate_flat_indices,
    grid_shape,
    threshold_fraction,
    min_component_voxels=1,
    min_total_activity_fraction=0.0,
):
    """
    Split high-activity voxels into connected 3D blobs.

    Voxels are active when they are above threshold_fraction * max(activity).
    Active voxels that touch by face, edge, or corner are treated as one blob.
    If min_total_activity_fraction is positive, blobs below that fraction of
    the strongest blob's total activity are discarded.
    """
    activity = np.maximum(np.asarray(activity, dtype=float), 0.0)
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    candidate_flat_indices = np.asarray(candidate_flat_indices, dtype=np.int64)
    grid_shape = tuple(int(value) for value in grid_shape)

    if activity.shape != (len(candidate_positions),):
        raise ValueError("activity and candidate_positions must describe the same voxels.")
    if candidate_flat_indices.shape != activity.shape:
        raise ValueError("candidate_flat_indices must have the same length as activity.")
    if len(activity) == 0 or float(np.max(activity)) <= EPS:
        return []

    threshold_fraction = float(threshold_fraction)
    if not 0.0 <= threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be between 0 and 1.")
    min_total_activity_fraction = float(min_total_activity_fraction)
    if not 0.0 <= min_total_activity_fraction <= 1.0:
        raise ValueError("min_total_activity_fraction must be between 0 and 1.")

    threshold = threshold_fraction * float(np.max(activity))
    active_candidate_indices = np.flatnonzero(activity >= threshold)
    if len(active_candidate_indices) == 0:
        return []

    active_flat_to_candidate = {
        int(candidate_flat_indices[candidate_index]): int(candidate_index)
        for candidate_index in active_candidate_indices
    }
    candidate_grid_indices = np.column_stack(
        np.unravel_index(candidate_flat_indices, grid_shape)
    ).astype(np.int64)

    visited = set()
    components = []

    for start_candidate_index in active_candidate_indices:
        start_candidate_index = int(start_candidate_index)
        if start_candidate_index in visited:
            continue

        queue = deque([start_candidate_index])
        visited.add(start_candidate_index)
        component_indices = []

        while queue:
            candidate_index = queue.popleft()
            component_indices.append(candidate_index)
            gx, gy, gz = candidate_grid_indices[candidate_index]

            for dx, dy, dz in NEIGHBOR_OFFSETS_3D:
                neighbor_grid_index = (int(gx + dx), int(gy + dy), int(gz + dz))
                if (
                    neighbor_grid_index[0] < 0
                    or neighbor_grid_index[1] < 0
                    or neighbor_grid_index[2] < 0
                    or neighbor_grid_index[0] >= grid_shape[0]
                    or neighbor_grid_index[1] >= grid_shape[1]
                    or neighbor_grid_index[2] >= grid_shape[2]
                ):
                    continue

                neighbor_flat = int(np.ravel_multi_index(neighbor_grid_index, grid_shape))
                neighbor_candidate_index = active_flat_to_candidate.get(neighbor_flat)
                if neighbor_candidate_index is None or neighbor_candidate_index in visited:
                    continue

                visited.add(neighbor_candidate_index)
                queue.append(neighbor_candidate_index)

        if len(component_indices) < int(min_component_voxels):
            continue

        indices = np.asarray(component_indices, dtype=np.int64)
        component_activity = activity[indices]
        component_positions = candidate_positions[indices]
        total_activity = float(np.sum(component_activity))
        peak_local_index = int(np.argmax(component_activity))
        if total_activity > EPS:
            centroid = np.average(
                component_positions,
                axis=0,
                weights=component_activity,
            )
        else:
            centroid = np.mean(component_positions, axis=0)

        components.append(
            {
                "indices": indices,
                "voxel_count": int(len(indices)),
                "total_activity": total_activity,
                "max_activity": float(component_activity[peak_local_index]),
                "centroid": centroid,
                "peak_position": component_positions[peak_local_index],
            }
        )

    components.sort(key=lambda component: component["total_activity"], reverse=True)
    if components and min_total_activity_fraction > 0.0:
        strongest_total_activity = float(components[0]["total_activity"])
        min_total_activity = min_total_activity_fraction * strongest_total_activity
        components = [
            component
            for component in components
            if float(component["total_activity"]) >= min_total_activity
        ]
    for label, component in enumerate(components, start=1):
        component["label"] = label
    return components


def make_true_source_components(
    true_sources,
    candidate_positions,
    min_component_voxels=1,
):
    """
    Build true blob components directly from generated source definitions.

    Unlike reconstructed blobs, true blobs are not thresholded. Each source keeps
    all of its generated components because the simulation knows the full blob.
    """
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    components = []
    if len(candidate_positions) == 0:
        return components

    for source in true_sources:
        source_components = source.get("components")
        if source_components is None:
            source_components = [
                {
                    "position": source["position"],
                    "strength": source.get("strength", source.get("strength_per_second", 0.0)),
                }
            ]

        component_positions = np.asarray(
            [component["position"] for component in source_components],
            dtype=float,
        )
        component_strengths = np.asarray(
            [component["strength"] for component in source_components],
            dtype=float,
        )
        if len(component_positions) == 0:
            continue

        squared_distances = np.sum(
            (component_positions[:, np.newaxis, :] - candidate_positions[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        nearest_indices = np.argmin(squared_distances, axis=1)
        source_activity_by_voxel = {}
        for candidate_index, strength in zip(nearest_indices, component_strengths):
            candidate_index = int(candidate_index)
            source_activity_by_voxel[candidate_index] = (
                source_activity_by_voxel.get(candidate_index, 0.0) + float(strength)
            )

        indices = np.array(sorted(source_activity_by_voxel), dtype=np.int64)
        if len(indices) < int(min_component_voxels):
            continue

        component_activity = np.array(
            [source_activity_by_voxel[int(index)] for index in indices],
            dtype=float,
        )
        component_positions = candidate_positions[indices]
        total_activity = float(np.sum(component_activity))
        peak_local_index = int(np.argmax(component_activity))
        if total_activity > EPS:
            centroid = np.average(
                component_positions,
                axis=0,
                weights=component_activity,
            )
        else:
            centroid = np.mean(component_positions, axis=0)

        components.append(
            {
                "indices": indices,
                "voxel_count": int(len(indices)),
                "total_activity": total_activity,
                "max_activity": float(component_activity[peak_local_index]),
                "centroid": centroid,
                "peak_position": component_positions[peak_local_index],
            }
        )

    components.sort(key=lambda component: component["total_activity"], reverse=True)
    for label, component in enumerate(components, start=1):
        component["label"] = label
    return components


def min_component_voxel_distance(component_a, component_b, candidate_positions, chunk_size=512):
    """Return the closest voxel-to-voxel distance between two components."""
    positions_a = candidate_positions[component_a["indices"]]
    positions_b = candidate_positions[component_b["indices"]]
    best_squared = np.inf

    for start in range(0, len(positions_a), int(chunk_size)):
        chunk = positions_a[start:start + int(chunk_size)]
        squared_distances = np.sum(
            (chunk[:, np.newaxis, :] - positions_b[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        best_squared = min(best_squared, float(np.min(squared_distances)))

    return float(np.sqrt(best_squared))


def summarize_activity_component_matching(
    true_activity,
    reconstructed_activity,
    candidate_positions,
    candidate_flat_indices,
    grid_shape,
    true_sources=None,
    threshold_fraction=0.5,
    min_component_voxels=1,
    min_total_activity_fraction=0.0,
):
    """
    Compare reconstructed activity blobs with true activity blobs.

    Components are matched one-to-one by nearest weighted centroid distance.
    Extra reconstructed components and missed true components are reported.
    """
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    if true_sources is None:
        true_components = find_activity_components(
            activity=true_activity,
            candidate_positions=candidate_positions,
            candidate_flat_indices=candidate_flat_indices,
            grid_shape=grid_shape,
            threshold_fraction=0.0,
            min_component_voxels=min_component_voxels,
        )
    else:
        true_components = make_true_source_components(
            true_sources=true_sources,
            candidate_positions=candidate_positions,
            min_component_voxels=min_component_voxels,
        )
    reconstructed_components = find_activity_components(
        activity=reconstructed_activity,
        candidate_positions=candidate_positions,
        candidate_flat_indices=candidate_flat_indices,
        grid_shape=grid_shape,
        threshold_fraction=threshold_fraction,
        min_component_voxels=min_component_voxels,
        min_total_activity_fraction=min_total_activity_fraction,
    )

    candidate_pairs = []
    for reconstructed_index, reconstructed_component in enumerate(reconstructed_components):
        for true_index, true_component in enumerate(true_components):
            centroid_distance = float(
                np.linalg.norm(
                    reconstructed_component["centroid"] - true_component["centroid"]
                )
            )
            candidate_pairs.append((centroid_distance, reconstructed_index, true_index))

    matched_reconstructed = set()
    matched_true = set()
    matches = []
    for centroid_distance, reconstructed_index, true_index in sorted(candidate_pairs):
        if reconstructed_index in matched_reconstructed or true_index in matched_true:
            continue

        reconstructed_component = reconstructed_components[reconstructed_index]
        true_component = true_components[true_index]
        reconstructed_total_activity = float(reconstructed_component["total_activity"])
        true_total_activity = float(true_component["total_activity"])
        activity_recovery_fraction = reconstructed_total_activity / max(
            true_total_activity,
            EPS,
        )
        matched_reconstructed.add(reconstructed_index)
        matched_true.add(true_index)
        matches.append(
            {
                "reconstructed_label": int(reconstructed_component["label"]),
                "true_label": int(true_component["label"]),
                "centroid_distance_cm": centroid_distance,
                "closest_voxel_distance_cm": min_component_voxel_distance(
                    reconstructed_component,
                    true_component,
                    candidate_positions,
                ),
                "peak_distance_cm": float(
                    np.linalg.norm(
                        reconstructed_component["peak_position"]
                        - true_component["peak_position"]
                    )
                ),
                "reconstructed_voxels": int(reconstructed_component["voxel_count"]),
                "true_voxels": int(true_component["voxel_count"]),
                "reconstructed_total_activity": reconstructed_total_activity,
                "true_total_activity": true_total_activity,
                "activity_recovery_fraction": activity_recovery_fraction,
                "activity_error_fraction": activity_recovery_fraction - 1.0,
                "reconstructed_centroid": reconstructed_component["centroid"],
                "true_centroid": true_component["centroid"],
            }
        )

    extra_reconstructed = [
        component
        for index, component in enumerate(reconstructed_components)
        if index not in matched_reconstructed
    ]
    missed_true = [
        component
        for index, component in enumerate(true_components)
        if index not in matched_true
    ]

    centroid_distances = [match["centroid_distance_cm"] for match in matches]
    closest_distances = [match["closest_voxel_distance_cm"] for match in matches]
    activity_recovery_fractions = [
        match["activity_recovery_fraction"] for match in matches
    ]

    return {
        "threshold_fraction": float(threshold_fraction),
        "min_component_voxels": int(min_component_voxels),
        "min_total_activity_fraction": float(min_total_activity_fraction),
        "true_uses_full_source_blobs": true_sources is not None,
        "true_components": true_components,
        "reconstructed_components": reconstructed_components,
        "true_component_count": int(len(true_components)),
        "reconstructed_component_count": int(len(reconstructed_components)),
        "matched_component_count": int(len(matches)),
        "extra_reconstructed_count": int(len(extra_reconstructed)),
        "missed_true_count": int(len(missed_true)),
        "mean_centroid_distance_cm": (
            float(np.mean(centroid_distances)) if centroid_distances else np.nan
        ),
        "max_centroid_distance_cm": (
            float(np.max(centroid_distances)) if centroid_distances else np.nan
        ),
        "mean_closest_voxel_distance_cm": (
            float(np.mean(closest_distances)) if closest_distances else np.nan
        ),
        "mean_activity_recovery_fraction": (
            float(np.mean(activity_recovery_fractions))
            if activity_recovery_fractions
            else np.nan
        ),
        "matches": matches,
        "extra_reconstructed": extra_reconstructed,
        "missed_true": missed_true,
    }


def print_activity_component_matching_summary(summary):
    """Print reconstructed-vs-true blob matching metrics."""
    print()
    print("Activity component matching metrics:")
    print(
        " recon threshold/min voxels/min blob:"
        f"{100.0 * summary['threshold_fraction']:.1f}% of each max / "
        f"{summary['min_component_voxels']} / "
        f"{100.0 * summary.get('min_total_activity_fraction', 0.0):.1f}% of strongest"
    )
    if summary.get("true_uses_full_source_blobs", False):
        print(" true blobs:               full generated source blobs, no threshold")
    print(
        " blobs true/recon/matched:  "
        f"{summary['true_component_count']} / "
        f"{summary['reconstructed_component_count']} / "
        f"{summary['matched_component_count']}"
    )
    print(
        " extra recon / missed true: "
        f"{summary['extra_reconstructed_count']} / "
        f"{summary['missed_true_count']}"
    )
    print(
        " centroid distance mean/max [cm]: "
        f"{summary['mean_centroid_distance_cm']:.4f} / "
        f"{summary['max_centroid_distance_cm']:.4f}"
    )
    print(
        " closest voxel distance mean [cm]: "
        f"{summary['mean_closest_voxel_distance_cm']:.4f}"
    )
    print(
        " activity recovery mean recon/true: "
        f"{summary['mean_activity_recovery_fraction']:.4f}"
    )

    for match in summary["matches"]:
        recon_center = np.round(match["reconstructed_centroid"], 3)
        true_center = np.round(match["true_centroid"], 3)
        print(
            "  "
            f"R{match['reconstructed_label']} -> T{match['true_label']}: "
            f"centroid={match['centroid_distance_cm']:.4f} cm, "
            f"closest_voxels={match['closest_voxel_distance_cm']:.4f} cm, "
            f"peak={match['peak_distance_cm']:.4f} cm, "
            f"voxels={match['reconstructed_voxels']}/{match['true_voxels']}, "
            f"activity={match['reconstructed_total_activity']:.4f}/"
            f"{match['true_total_activity']:.4f}, "
            f"recovery={match['activity_recovery_fraction']:.4f}, "
            f"error={100.0 * match['activity_error_fraction']:.1f}%, "
            f"centers recon/true={recon_center}/{true_center}"
        )

    for component in summary["extra_reconstructed"][:5]:
        center = np.round(component["centroid"], 3)
        print(
            "  "
            f"extra R{component['label']}: "
            f"voxels={component['voxel_count']}, "
            f"activity={component['total_activity']:.4f}, "
            f"center={center}"
        )

    for component in summary["missed_true"][:5]:
        center = np.round(component["centroid"], 3)
        print(
            "  "
            f"missed T{component['label']}: "
            f"voxels={component['voxel_count']}, "
            f"activity={component['total_activity']:.4f}, "
            f"center={center}"
        )
