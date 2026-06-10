"""Shared settings for article Pearson-only reconstruction runs."""

from pathlib import Path


ON_MOCK_DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ON_MOCK_DATA_DIR / "article_results"

TRUE_SOURCE_CASES = [
    {
        "name": "mine",
        "true_sources": [
            {"position": (-6.5, 2.6, -0.8), "strength_per_second": 6000.0, "spread_cm": 1.8},
            {"position": (4.2, -1.2, 1.3), "strength_per_second": 7000.0, "spread_cm": 1.2},
        ],
    },
    {
        "name": "1",
        "true_sources": [
            {"position": (-2.5, 1.5, -1.0), "strength_per_second": 5000.0, "spread_cm": 1.2},
            {"position": (2.5, 1.0, 1.0), "strength_per_second": 4500.0, "spread_cm": 1.0},
            {"position": (1.5, 5.0, -0.5), "strength_per_second": 8000.0, "spread_cm": 0.5},
        ],
    },
    {
        "name": "2",
        "true_sources": [
            {"position": (0.0, -2.0, 0.0), "strength_per_second": 12000.0, "spread_cm": 1.0},
            {"position": (0.0, 6.0, 0.0), "strength_per_second": 2000.0, "spread_cm": 0.5},
            {"position": (-4.0, 2.0, 2.0), "strength_per_second": 6000.0, "spread_cm": 0.8},
        ],
    },
    {
        "name": "3",
        "true_sources": [
            {"position": (-1.0, 0.0, 0.0), "strength_per_second": 20000.0, "spread_cm": 2.0},
            {"position": (4.0, 4.0, 3.0), "strength_per_second": 3500.0, "spread_cm": 0.5},
            {"position": (2.0, -3.0, -4.0), "strength_per_second": 5000.0, "spread_cm": 0.7},
        ],
    },
]

ARTICLE_PRIORS = [
    {"type": "tv", "beta": 0.01, "epsilon": 0.001},
    {"type": "l1", "beta": 0.01},
]

BASE_OVERRIDES = {
    "SEED": 11,
    "NUM_MEASUREMENTS": 100,
    "PRINT_GENERATED_COUNTS_TABLE": False,
    "VOXEL_SIZE_CM": 0.5,
    "ACTIVITY_COMPONENT_THRESHOLD_FRACTION": 0.1,
    "ACTIVITY_COMPONENT_MIN_VOXELS": 3,
    "ACTIVITY_COMPONENT_MIN_TOTAL_FRACTION": 0.05,
}

OFFLINE_BASE_OVERRIDES = {
    **BASE_OVERRIDES,
    "NUM_ITERATIONS": 1000,
    "HISTORY_INTERVAL": 1000,
    "THRESHOLD_REFINEMENT_METHOD": "same",
    "THRESHOLD_REFINEMENT_NUM_ITERATIONS": 10,
    "THRESHOLD_REFINEMENT_HISTORY_INTERVAL": 10,
    "THRESHOLD_REFINEMENT_PRIORS": [],
    "THRESHOLD_REFINEMENT_PADDING_CM": 0.75,
}

ONLINE_BASE_OVERRIDES = {
    **BASE_OVERRIDES,
    "ONLINE_MEASUREMENT_BATCH_SIZE": 5,
    "ONLINE_UPDATES_PER_BATCH": 50,
    "ONLINE_HISTORY_INTERVAL": 50,
    "THRESHOLD_REFINEMENT_INTERVAL_BATCHES": 5,
    "THRESHOLD_REFINEMENT_METHOD": "same",
    "THRESHOLD_REFINEMENT_NUM_ITERATIONS": 10,
    "THRESHOLD_REFINEMENT_HISTORY_INTERVAL": 10,
    "THRESHOLD_REFINEMENT_PRIORS": [],
    "THRESHOLD_REFINEMENT_PADDING_CM": 0.75,
    "THRESHOLD_REFINEMENT_OUTSIDE_SUPPORT_DECAY": 0.0,
    "GENERATE_SYSTEM_MATRIX_MEASUREMENT_GIF": False,
}


def pearson_from_experiment(experiment):
    """Return the Pearson activity-map similarity from a run_experiment result."""
    return float(experiment["activity_map_similarity"]["pearson_correlation"])


def component_metrics_from_experiment(experiment):
    """Return compact component-matching metrics for article comparison tables."""
    similarity = experiment["activity_map_similarity"]
    matching = experiment["activity_component_matching"]
    matches = list(matching.get("matches", []))
    activity_errors = [abs(float(match["activity_error_fraction"])) for match in matches]
    return {
        "true_component_count": int(matching["true_component_count"]),
        "reconstructed_component_count": int(matching["reconstructed_component_count"]),
        "matched_component_count": int(matching["matched_component_count"]),
        "missed_true_count": int(matching["missed_true_count"]),
        "extra_reconstructed_count": int(matching["extra_reconstructed_count"]),
        "centroid_distance_mean_cm": float(matching["mean_centroid_distance_cm"]),
        "closest_voxel_distance_mean_cm": float(matching["mean_closest_voxel_distance_cm"]),
        "activity_recovery_mean": float(matching["mean_activity_recovery_fraction"]),
        "activity_error_abs_mean": (
            sum(activity_errors) / len(activity_errors)
            if activity_errors
            else float("nan")
        ),
        "activity_error_abs_max": max(activity_errors) if activity_errors else float("nan"),
        "true_total_activity": float(similarity["true_total_activity"]),
        "reconstructed_total_activity": float(similarity["reconstructed_total_activity"]),
        "scale_to_true_total": float(similarity["scale_to_true_total"]),
    }


def write_rows_csv(path, fieldnames, rows):
    """Write a compact CSV file with one row per reconstruction case."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
