"""
Run reconstruction sweeps over method, priors, and true source scenarios.

Only these parameters are varied:
- reconstruction_method
- RECONSTRUCTION_PRIORS
- RECONSTRUCTION_SIEVE_SIGMA_CM
- ACTIVITY_COMPONENT_THRESHOLD_FRACTION
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
    # "mlem",
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


RECONSTRUCTION_SIEVE_SIGMA_CM_VARIANTS = [
    0.0,  # Disabled.
    0.01,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.3,
    1.5,
    2.0,
    3.0
]


ACTIVITY_COMPONENT_THRESHOLD_FRACTION_VARIANTS = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.8,
    0.9
]

# Available priors:
# smoothness: beta range: 0.01 - 1.0: Penalizes differences from the six-neighbor local mean; smooths noisy voxel spikes.
# huber: delta range: 0.1 - 5.0: Smooths small neighbor differences while preserving sharper blob edges; delta is the activity-difference scale where the penalty changes from quadratic-like to edge-preserving.
# rdp: gamma range: 0.1 - 3.0: Relative Difference Penalty; smooths weak/noisy regions while being gentler near strong activity contrast; gamma controls edge preservation in the relative denominator, with larger values preserving stronger contrasts.
# l1: beta range: 0.001 - 0.1: Adds sparsity pressure; discourages diffuse low activity spread across many voxels.

RECONSTRUCTION_PRIOR_VARIANTS = [
    [],  # No priors; useful for plain ML-EM / unregularized baseline.
    # [
    #     {"type": "smoothness", "beta": 0.01},
    # ],
    # [
    #     {"type": "smoothness", "beta": 0.1},
    # ],
    # [
    #     {"type": "smoothness", "beta": 0.5},
    # ],
    # [
    #     {"type": "smoothness", "beta": 1.0},
    # ],
    # [
    #     {"type": "huber", "beta": 0.01, "delta": 0.1},
    # ],
    # [
    #     {"type": "huber", "beta": 0.01, "delta": 0.3},
    # ],
    # [
    #     {"type": "huber", "beta": 0.01, "delta": 0.5},
    # ],
    # [
    #     {"type": "huber", "beta": 0.01, "delta": 0.8},
    # ],
    # [
    #     {"type": "huber", "beta": 0.01, "delta": 1.0},
    # ],
    # [
    #     {"type": "huber", "beta": 0.01, "delta": 1.5},
    # ],
    # [
    #     {"type": "huber", "beta": 0.01, "delta": 2.5},
    # ],
    # [
    #     {"type": "huber", "beta": 0.01, "delta": 3.0},
    # ],
    # [
    #     {"type": "huber", "beta": 0.01, "delta": 5.0},
    # ],
    # [
    #     {"type": "huber", "beta": 0.3, "delta": 0.1},
    # ],
    # [
    #     {"type": "huber", "beta": 0.3, "delta": 0.3},
    # ],
    # [
    #     {"type": "huber", "beta": 0.3, "delta": 0.5},
    # ],
    # [
    #     {"type": "huber", "beta": 0.3, "delta": 0.8},
    # ],
    # [
    #     {"type": "huber", "beta": 0.3, "delta": 1.0},
    # ],
    # [
    #     {"type": "huber", "beta": 0.3, "delta": 1.5},
    # ],
    # [
    #     {"type": "huber", "beta": 0.3, "delta": 2.5},
    # ],
    # [
    #     {"type": "huber", "beta": 0.3, "delta": 3.0},
    # ],
    # [
    #     {"type": "huber", "beta": 0.3, "delta": 5.0},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.01, "gamma": 0.1},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.01, "gamma": 0.3},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.01, "gamma": 0.5},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.01, "gamma": 0.8},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.01, "gamma": 1.0},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.01, "gamma": 1.5},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.01, "gamma": 2.0},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.01, "gamma": 3.0},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.3, "gamma": 0.1},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.3, "gamma": 0.3},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.3, "gamma": 0.5},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.3, "gamma": 0.8},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.3, "gamma": 1.0},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.3, "gamma": 1.5},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.3, "gamma": 2.0},
    # ],
    # [
    #     {"type": "rdp", "beta": 0.3, "gamma": 3.0},
    # ],
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


OUTPUT_DIR = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations"
OUTPUT_CSV = OUTPUT_DIR / "reconstruction_sieve_l1_sweep.csv"
PAUSE_TOGGLE_KEY = "0"

SWEEP_CSV_COLUMNS = [
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
    "reconstruction_sieve_sigma_cm",
    "activity_component_threshold_fraction",
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
    "matched_reconstructed_voxels_mean",
    "matched_true_voxels_mean",
    "matched_voxel_count_ratio_mean",
    "matched_reconstructed_voxels_total",
    "matched_true_voxels_total",
    "matched_voxel_count_ratio_total",
]


class ConsoleCasePauseToggle:
    """Pause the sweep between cases when the toggle key is pending."""

    def __init__(self, toggle_key):
        self.toggle_key = str(toggle_key).lower()
        self.paused = False
        try:
            import msvcrt
        except ImportError:
            self.console = None
        else:
            self.console = msvcrt

    def print_help(self):
        """Explain the between-case pause shortcut."""
        if self.console is None:
            print("Between-case pause key is available only on Windows terminals.")
            return
        print(
            f"Press '{self.toggle_key}' during or between cases to pause before "
            "the next case; press it again to resume."
        )

    def _toggle(self):
        self.paused = not self.paused
        if self.paused:
            print(f"\nSweep paused. Press '{self.toggle_key}' to resume.")
            return
        print("\nSweep resumed.")

    def _consume_toggle_keys(self):
        if self.console is None:
            return
        while self.console.kbhit():
            key = self.console.getwch().lower()
            if key == self.toggle_key:
                self._toggle()

    def checkpoint(self):
        """Read pending key presses and wait only between sweep cases."""
        self._consume_toggle_keys()
        time.sleep(1.0)


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
    """Yield every method x prior x sieve x threshold x true-source x iteration x measurement combination."""
    for (
        method,
        reconstruction_priors,
        reconstruction_sieve_sigma_cm,
        activity_component_threshold_fraction,
        source_case,
        iteration_case,
        num_measurements,
    ) in product(
        RECONSTRUCTION_METHODS,
        RECONSTRUCTION_PRIOR_VARIANTS,
        RECONSTRUCTION_SIEVE_SIGMA_CM_VARIANTS,
        ACTIVITY_COMPONENT_THRESHOLD_FRACTION_VARIANTS,
        TRUE_SOURCE_CASES,
        ITERATION_CASES,
        NUM_MEASUREMENT_VARIANTS,
    ):
        overrides = {
            **FIXED_OVERRIDES,
            "reconstruction_method": method,
            "RECONSTRUCTION_PRIORS": reconstruction_priors,
            "RECONSTRUCTION_SIEVE_SIGMA_CM": reconstruction_sieve_sigma_cm,
            "ACTIVITY_COMPONENT_THRESHOLD_FRACTION": activity_component_threshold_fraction,
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
            "reconstruction_sieve_sigma_cm": float(reconstruction_sieve_sigma_cm),
            "activity_component_threshold_fraction": float(
                activity_component_threshold_fraction
            ),
            "source_case": source_case,
            "iteration_case": iteration_case,
            "num_measurements": num_measurements,
            "overrides": overrides,
        }


def add_match_columns(row, component_matching):
    """Add only aggregate component metrics to one CSV row."""
    matches = list(component_matching.get("matches", []))
    reconstructed_voxels = np.asarray(
        [match["reconstructed_voxels"] for match in matches],
        dtype=float,
    )
    true_voxels = np.asarray(
        [match["true_voxels"] for match in matches],
        dtype=float,
    )
    voxel_ratios = reconstructed_voxels / np.maximum(true_voxels, 1.0)
    reconstructed_voxel_total = float(np.sum(reconstructed_voxels))
    true_voxel_total = float(np.sum(true_voxels))

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
            "matched_reconstructed_voxels_mean": (
                float(np.mean(reconstructed_voxels)) if len(reconstructed_voxels) else np.nan
            ),
            "matched_true_voxels_mean": (
                float(np.mean(true_voxels)) if len(true_voxels) else np.nan
            ),
            "matched_voxel_count_ratio_mean": (
                float(np.mean(voxel_ratios)) if len(voxel_ratios) else np.nan
            ),
            "matched_reconstructed_voxels_total": reconstructed_voxel_total,
            "matched_true_voxels_total": true_voxel_total,
            "matched_voxel_count_ratio_total": (
                reconstructed_voxel_total / max(true_voxel_total, 1.0)
                if len(matches)
                else np.nan
            ),
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
        "reconstruction_sieve_sigma_cm": float(case["reconstruction_sieve_sigma_cm"]),
        "activity_component_threshold_fraction": float(
            case["activity_component_threshold_fraction"]
        ),
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
    """Return the fixed compact sweep CSV schema."""
    return SWEEP_CSV_COLUMNS


def write_rows(path, rows):
    """Append rows to CSV, writing the header only for a new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=ordered_columns(rows),
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    default_config = make_default_config()
    rows = []
    cases = list(iter_sweep_cases())

    print(f"Running {len(cases)} reconstruction sweep cases")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Sieve sigma variants [cm]: {RECONSTRUCTION_SIEVE_SIGMA_CM_VARIANTS}")
    print(
        "Component threshold variants: "
        f"{ACTIVITY_COMPONENT_THRESHOLD_FRACTION_VARIANTS}"
    )
    pause_toggle = ConsoleCasePauseToggle(PAUSE_TOGGLE_KEY)
    pause_toggle.print_help()

    for case_index, case in enumerate(cases, start=1):
        pause_toggle.checkpoint()
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
            f"sieve_sigma={row['reconstruction_sieve_sigma_cm']:.3g} cm, "
            f"component_threshold={row['activity_component_threshold_fraction']:.3g}, "
            f"priors={row['prior_case']} -> "
            f"cosine={row['shape_cosine']:.4f}, "
            f"pearson={row['shape_pearson']:.4f}, "
            f"centroid_mean={row['centroid_distance_mean_cm']:.4f} cm, "
            f"elapsed={elapsed_seconds:.1f}s"
        )
        if case_index < len(cases):
            pause_toggle.checkpoint()

    write_rows(OUTPUT_CSV, rows)

    print()
    print("Done.")
    print(f"Rows written: {len(rows)}")
    print(f"CSV path: {OUTPUT_CSV}")
    print(f"Base SEED: {default_config['SEED']}")


if __name__ == "__main__":
    main()
