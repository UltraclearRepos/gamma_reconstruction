"""
Run reconstruction sweeps over method, priors, and true source scenarios.

Only these parameters are varied:
- reconstruction_method
- RECONSTRUCTION_PRIORS
- TRUE_SOURCE_SPECS
- NUM_ITERATIONS with HISTORY_INTERVAL
- NUM_MEASUREMENTS
"""

from itertools import product
import csv
import json
from pathlib import Path
import time

import numpy as np

from run_demo import ON_MOCK_DATA_DIR, make_default_config, run_experiment


RECONSTRUCTION_METHODS = [
    "map_em",
    "bsrem",
]


ITERATION_CASES = [
    {"name": "iter_50", "num_iterations": 50, "history_interval": 50},
    {"name": "iter_100", "num_iterations": 100, "history_interval": 100},
    {"name": "iter_250", "num_iterations": 250, "history_interval": 250},
    {"name": "iter_500", "num_iterations": 500, "history_interval": 500},
    {"name": "iter_1000", "num_iterations": 1000, "history_interval": 1000},
    {"name": "iter_2000", "num_iterations": 2000, "history_interval": 2000},
]


NUM_MEASUREMENT_VARIANTS = [
    25,
    50,
    100,
]

# Available priors:
# smoothness: beta range: 0.01 - 1.0: Penalizes differences from the six-neighbor local mean; smooths noisy voxel spikes.
# huber: delta range: 0.1 - 5.0: Smooths small neighbor differences while preserving sharper blob edges; delta is the activity-difference scale where the penalty changes from quadratic-like to edge-preserving.
# rdp: gamma range: 0.1 - 3.0: Relative Difference Penalty; smooths weak/noisy regions while being gentler near strong activity contrast; gamma controls edge preservation in the relative denominator, with larger values preserving stronger contrasts.
# l1: beta range: 0.001 - 0.1: Adds sparsity pressure; discourages diffuse low activity spread across many voxels.

RECONSTRUCTION_PRIOR_VARIANTS = [
    [
        {"type": "smoothness", "beta": 0.01},
    ],
        [
        {"type": "smoothness", "beta": 0.1},
    ],
    [
        {"type": "smoothness", "beta": 0.5},
    ],
    [
        {"type": "smoothness", "beta": 1.0},
    ],
    [
        {"type": "huber", "beta": 0.1, "delta": 0.1},
    ],
    [
        {"type": "huber", "beta": 0.1, "delta": 0.3},
    ],
    [
        {"type": "huber", "beta": 0.1, "delta": 0.5},
    ],
    [
        {"type": "huber", "beta": 0.1, "delta": 0.8},
    ],
    [
        {"type": "huber", "beta": 0.1, "delta": 1.0},
    ],
    [
        {"type": "huber", "beta": 0.1, "delta": 1.5},
    ],
    [
        {"type": "huber", "beta": 0.1, "delta": 2.5},
    ],
    [
        {"type": "huber", "beta": 0.1, "delta": 3.0},
    ],
    [
        {"type": "huber", "beta": 0.1, "delta": 5.0},
    ],
    [
        {"type": "rdp", "beta": 0.1, "gamma": 0.1},
    ],
    [
        {"type": "rdp", "beta": 0.1, "gamma": 0.3},
    ],
    [
        {"type": "rdp", "beta": 0.1, "gamma": 0.5},
    ],
    [
        {"type": "rdp", "beta": 0.1, "gamma": 0.8},
    ],
    [
        {"type": "rdp", "beta": 0.1, "gamma": 1.0},
    ],
    [
        {"type": "rdp", "beta": 0.1, "gamma": 1.5},
    ],
    [
        {"type": "rdp", "beta": 0.1, "gamma": 2.0},
    ],
    [
        {"type": "rdp", "beta": 0.1, "gamma": 3.0},
    ],
    [
        {"type": "l1", "beta": 0.001},
    ],
    [
        {"type": "l1", "beta": 0.01},
    ],
    [
        {"type": "l1", "beta": 0.1},
    ],
    [
        {"type": "l1", "beta": 0.5},
    ]
]


TRUE_SOURCE_CASES = [
    {
        "name": "mine",
        "true_sources": [
            {"position": (-6.5, 2.6, -0.8), "strength_per_second": 6000.0, "spread_cm": 1.8},
            {"position": (4.2, -1.2, 1.3), "strength_per_second": 7000.0, "spread_cm": 1.2}
        ],
    },
    {
        "name": "1",
        "true_sources": [
            {"position": (-2.5, 1.5, -1.0), "strength_per_second": 5000.0, "spread_cm": 1.2},
            {"position": (2.5, 1.0, 1.0), "strength_per_second": 4500.0, "spread_cm": 1.0},
            {"position": (1.5, 5.0, -0.5), "strength_per_second": 8000.0, "spread_cm": 0.4}
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
            {"position": (4.0, 4.0, 3.0), "strength_per_second": 1500.0, "spread_cm": 0.2},
            {"position": (2.0, -3.0, -4.0), "strength_per_second": 5000.0, "spread_cm": 0.6},
        ],
    },
]


FIXED_OVERRIDES = {
    "SEED": 11,
    "PRINT_GENERATED_COUNTS_TABLE": False,
}


OUTPUT_CSV = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations" / "reconstruction_method_prior_sources_sweep.csv"


def _csv_value(value):
    """Convert nested settings to compact CSV-safe values."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def prior_variant_name(priors):
    """Build a compact stable label from a list of prior dictionaries."""
    if not priors:
        return "no_priors"

    parts = []
    for prior in priors:
        prior_type = str(prior.get("type", "prior"))
        parameters = [
            f"{key}_{prior[key]}"
            for key in sorted(prior)
            if key != "type"
        ]
        parts.append("_".join([prior_type, *parameters]))
    return "__".join(parts).replace(".", "_")


def iter_sweep_cases():
    """Yield every method x prior x true-source x iteration x measurement combination."""
    for method, reconstruction_priors, source_case, iteration_case, num_measurements in product(
        RECONSTRUCTION_METHODS,
        RECONSTRUCTION_PRIOR_VARIANTS,
        TRUE_SOURCE_CASES,
        ITERATION_CASES,
        NUM_MEASUREMENT_VARIANTS,
    ):
        overrides = {
            **FIXED_OVERRIDES,
            "reconstruction_method": method,
            "RECONSTRUCTION_PRIORS": reconstruction_priors,
            "TRUE_SOURCE_SPECS": source_case["true_sources"],
            "NUM_ITERATIONS": iteration_case["num_iterations"],
            "HISTORY_INTERVAL": iteration_case["history_interval"],
            "NUM_MEASUREMENTS": num_measurements,
        }
        yield {
            "method": method,
            "prior_case": {
                "name": prior_variant_name(reconstruction_priors),
                "priors": reconstruction_priors,
            },
            "source_case": source_case,
            "iteration_case": iteration_case,
            "num_measurements": num_measurements,
            "overrides": overrides,
        }


def add_match_columns(row, component_matching):
    """Add matched, missed, and extra component metrics to one CSV row."""
    row.update(
        {
            "true_component_count": int(component_matching["true_component_count"]),
            "reconstructed_component_count": int(component_matching["reconstructed_component_count"]),
            "matched_component_count": int(component_matching["matched_component_count"]),
            "extra_reconstructed_count": int(component_matching["extra_reconstructed_count"]),
            "missed_true_count": int(component_matching["missed_true_count"]),
            "centroid_distance_mean_cm": float(component_matching["mean_centroid_distance_cm"]),
            "centroid_distance_max_cm": float(component_matching["max_centroid_distance_cm"]),
            "closest_voxel_distance_mean_cm": float(
                component_matching["mean_closest_voxel_distance_cm"]
            ),
            "activity_recovery_mean": float(component_matching["mean_activity_recovery_fraction"]),
        }
    )

    for match_index, match in enumerate(component_matching["matches"], start=1):
        prefix = f"match_{match_index}"
        row.update(
            {
                f"{prefix}_reconstructed_label": int(match["reconstructed_label"]),
                f"{prefix}_true_label": int(match["true_label"]),
                f"{prefix}_centroid_distance_cm": float(match["centroid_distance_cm"]),
                f"{prefix}_closest_voxel_distance_cm": float(match["closest_voxel_distance_cm"]),
                f"{prefix}_peak_distance_cm": float(match["peak_distance_cm"]),
                f"{prefix}_reconstructed_voxels": int(match["reconstructed_voxels"]),
                f"{prefix}_true_voxels": int(match["true_voxels"]),
                f"{prefix}_reconstructed_total_activity": float(
                    match["reconstructed_total_activity"]
                ),
                f"{prefix}_true_total_activity": float(match["true_total_activity"]),
                f"{prefix}_activity_recovery_fraction": float(
                    match["activity_recovery_fraction"]
                ),
                f"{prefix}_activity_error_fraction": float(match["activity_error_fraction"]),
                f"{prefix}_activity_error_percent": 100.0
                * float(match["activity_error_fraction"]),
                f"{prefix}_reconstructed_centroid": _csv_value(match["reconstructed_centroid"]),
                f"{prefix}_true_centroid": _csv_value(match["true_centroid"]),
            }
        )

    for missed_index, component in enumerate(component_matching["missed_true"], start=1):
        prefix = f"missed_true_{missed_index}"
        row.update(
            {
                f"{prefix}_label": int(component["label"]),
                f"{prefix}_voxels": int(component["voxel_count"]),
                f"{prefix}_activity": float(component["total_activity"]),
                f"{prefix}_centroid": _csv_value(component["centroid"]),
            }
        )

    for extra_index, component in enumerate(component_matching["extra_reconstructed"], start=1):
        prefix = f"extra_reconstructed_{extra_index}"
        row.update(
            {
                f"{prefix}_label": int(component["label"]),
                f"{prefix}_voxels": int(component["voxel_count"]),
                f"{prefix}_activity": float(component["total_activity"]),
                f"{prefix}_centroid": _csv_value(component["centroid"]),
            }
        )


def summarize_run(case_index, case, experiment, elapsed_seconds):
    """Flatten one run into one CSV row."""
    similarity = experiment["activity_map_similarity"]
    component_matching = experiment["activity_component_matching"]
    result = experiment["result"]
    timings = result.get("timings", {})

    row = {
        "case_index": int(case_index),
        "source_case": case["source_case"]["name"],
        "reconstruction_method": case["method"],
        "prior_case": case["prior_case"]["name"],
        "iteration_case": case["iteration_case"]["name"],
        "num_measurements": int(case["num_measurements"]),
        "num_iterations": int(case["iteration_case"]["num_iterations"]),
        "history_interval": int(case["iteration_case"]["history_interval"]),
        "true_sources": _csv_value(case["source_case"]["true_sources"]),
        "reconstruction_priors": _csv_value(case["prior_case"]["priors"]),
        "elapsed_seconds": float(elapsed_seconds),
        "system_matrix_seconds": float(timings.get("system_matrix_seconds", np.nan)),
        "transfer_seconds": float(timings.get("transfer_seconds", np.nan)),
        "iteration_seconds": float(
            timings.get("mlem_iterations_seconds")
            or timings.get("mapem_iterations_seconds")
            or timings.get("bsrem_iterations_seconds")
            or np.nan
        ),
        "true_total_activity": float(similarity["true_total_activity"]),
        "reconstructed_total_activity": float(similarity["reconstructed_total_activity"]),
        "scale_to_true_total": float(similarity["scale_to_true_total"]),
        "shape_cosine": float(similarity["cosine_similarity"]),
        "shape_pearson": float(similarity["pearson_correlation"]),
    }
    add_match_columns(row, component_matching)
    return row


def ordered_columns(rows):
    """Return stable CSV columns with dynamic match columns appended."""
    preferred = [
        "case_index",
        "source_case",
        "reconstruction_method",
        "prior_case",
        "iteration_case",
        "num_measurements",
        "num_iterations",
        "history_interval",
        "true_sources",
        "reconstruction_priors",
        "elapsed_seconds",
        "system_matrix_seconds",
        "transfer_seconds",
        "iteration_seconds",
        "true_total_activity",
        "reconstructed_total_activity",
        "scale_to_true_total",
        "shape_cosine",
        "shape_pearson",
        "true_component_count",
        "reconstructed_component_count",
        "matched_component_count",
        "extra_reconstructed_count",
        "missed_true_count",
        "centroid_distance_mean_cm",
        "centroid_distance_max_cm",
        "closest_voxel_distance_mean_cm",
        "activity_recovery_mean",
    ]
    seen = set(preferred)
    dynamic = sorted(key for row in rows for key in row if key not in seen)
    return preferred + dynamic


def write_rows(path, rows):
    """Write rows to CSV with stable column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ordered_columns(rows))
        writer.writeheader()
        writer.writerows(rows)


def main():
    default_config = make_default_config()
    rows = []
    cases = list(iter_sweep_cases())

    print(f"Running {len(cases)} reconstruction sweep cases")
    print(f"Output CSV: {OUTPUT_CSV}")

    for case_index, case in enumerate(cases, start=1):
        started_at = time.perf_counter()
        experiment = run_experiment(overrides=case["overrides"])
        elapsed_seconds = time.perf_counter() - started_at

        row = summarize_run(
            case_index=case_index,
            case=case,
            experiment=experiment,
            elapsed_seconds=elapsed_seconds,
        )
        rows.append(row)

        print(
            f"[{case_index}/{len(cases)}] "
            f"sources={row['source_case']}, "
            f"method={row['reconstruction_method']}, "
            f"measurements={row['num_measurements']}, "
            f"iterations={row['num_iterations']}/{row['history_interval']}, "
            f"priors={row['prior_case']} -> "
            f"cosine={row['shape_cosine']:.4f}, "
            f"pearson={row['shape_pearson']:.4f}, "
            f"centroid_mean={row['centroid_distance_mean_cm']:.4f} cm, "
            f"elapsed={elapsed_seconds:.1f}s"
        )

    write_rows(OUTPUT_CSV, rows)

    print()
    print("Done.")
    print(f"Rows written: {len(rows)}")
    print(f"CSV path: {OUTPUT_CSV}")
    print(f"Base SEED: {default_config['SEED']}")


if __name__ == "__main__":
    main()
