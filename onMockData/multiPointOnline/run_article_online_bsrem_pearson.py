"""Run online BSREM with/without refinement and save Pearson only."""

from pathlib import Path
import sys
from time import perf_counter

ON_MOCK_DATA_DIR = Path(__file__).resolve().parents[1]
if str(ON_MOCK_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(ON_MOCK_DATA_DIR))

from article_pearson_settings import (  # noqa: E402
    ARTICLE_PRIORS,
    ONLINE_BASE_OVERRIDES,
    OUTPUT_DIR,
    TRUE_SOURCE_CASES,
    component_metrics_from_experiment,
    pearson_from_experiment,
    write_rows_csv,
)
from run_online_demo import run_online_experiment  # noqa: E402


OUTPUT_CSV = OUTPUT_DIR / "online_bsrem_refinement_pearson.csv"


def main():
    rows = []
    total_online_updates = (
        ONLINE_BASE_OVERRIDES["NUM_MEASUREMENTS"]
        // ONLINE_BASE_OVERRIDES["ONLINE_MEASUREMENT_BATCH_SIZE"]
        * ONLINE_BASE_OVERRIDES["ONLINE_UPDATES_PER_BATCH"]
    )
    for refinement_enabled in (False, True):
        for case in TRUE_SOURCE_CASES:
            overrides = {
                **ONLINE_BASE_OVERRIDES,
                "TRUE_SOURCE_SPECS": case["true_sources"],
                "reconstruction_method": "online_bsrem",
                "RECONSTRUCTION_PRIORS": ARTICLE_PRIORS,
                "ENABLE_THRESHOLD_REFINEMENT": refinement_enabled,
            }
            started_at = perf_counter()
            experiment = run_online_experiment(overrides=overrides)
            elapsed_seconds = perf_counter() - started_at
            pearson = pearson_from_experiment(experiment)
            component_metrics = component_metrics_from_experiment(experiment)
            rows.append(
                {
                    "source_case": case["name"],
                    "reconstruction_method": "online_bsrem",
                    "refinement_enabled": refinement_enabled,
                    "num_measurements": overrides["NUM_MEASUREMENTS"],
                    "online_measurement_batch_size": overrides["ONLINE_MEASUREMENT_BATCH_SIZE"],
                    "online_updates_per_batch": overrides["ONLINE_UPDATES_PER_BATCH"],
                    "total_online_updates": total_online_updates,
                    "tv_beta": 0.01,
                    "tv_epsilon": 0.001,
                    "l1_beta": 0.01,
                    "outside_support_decay": overrides["THRESHOLD_REFINEMENT_OUTSIDE_SUPPORT_DECAY"],
                    "pearson_correlation": pearson,
                    **component_metrics,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            print(
                f"{case['name']} refinement={refinement_enabled}: "
                f"pearson={pearson:.6f}, elapsed={elapsed_seconds:.1f}s"
            )

    write_rows_csv(
        OUTPUT_CSV,
        [
            "source_case",
            "reconstruction_method",
            "refinement_enabled",
            "num_measurements",
            "online_measurement_batch_size",
            "online_updates_per_batch",
            "total_online_updates",
            "tv_beta",
            "tv_epsilon",
            "l1_beta",
            "outside_support_decay",
            "pearson_correlation",
            "true_component_count",
            "reconstructed_component_count",
            "matched_component_count",
            "missed_true_count",
            "extra_reconstructed_count",
            "centroid_distance_mean_cm",
            "closest_voxel_distance_mean_cm",
            "activity_recovery_mean",
            "activity_error_abs_mean",
            "activity_error_abs_max",
            "true_total_activity",
            "reconstructed_total_activity",
            "scale_to_true_total",
            "elapsed_seconds",
        ],
        rows,
    )
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
