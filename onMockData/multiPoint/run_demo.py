"""
Entrypoint for the synthetic multi-point source demo.
"""

from pathlib import Path
import sys

import numpy as np

ON_MOCK_DATA_DIR = Path(__file__).resolve().parents[1]
if str(ON_MOCK_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(ON_MOCK_DATA_DIR))

from reconstruction_algorithms import reconstruct_sources
from metrics import print_activity_localization_summary, summarize_activity_localization
from simulation import generate_synthetic_dataset
from visualization import (
    visualize_history_score,
    visualize_mlem_activity,
    visualize_multi_source_scene,
)


def print_summary(dataset):
    """Print a short summary of the generated multi-source dataset."""
    counts = dataset["counts"]

    print("=== Synthetic multi-point source demo ===")
    print()
    print("True sources:")
    for index, source in enumerate(dataset["true_sources"], start=1):
        print(
            " "
            f"S{index}: position={np.round(source['position'], 4)}, "
            f"strength={source['strength']:.2f}"
        )

    print()
    print(
        "Counts min/mean/max:        "
        f"{counts.min():.1f} / {counts.mean():.1f} / {counts.max():.1f}"
    )
    print(
        "Measurement mean min/max:   "
        f"{counts.mean(axis=1).min():.1f} / {counts.mean(axis=1).max():.1f}"
    )
    print(f"Background mean:            {dataset['background_mean']:.1f}")


def print_reconstruction_summary(method, result):
    """Print compact ML-EM reconstruction output."""
    print()
    print(f"Reconstruction method:       {method}")
    print()
    print("Estimated activity peaks:")
    for index, source in enumerate(result["estimated_sources"], start=1):
        print(
            " "
            f"P{index}: position={np.round(source['position'], 4)}, "
            f"activity={source['activity']:.2f}"
        )

    print()
    print("Reconstruction history:")
    for index, item in enumerate(result["history"], start=1):
        print(f" step={index}")
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {np.round(value, 4).tolist()}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def print_localization_metrics(result, true_sources, relative_drop):
    """Print localization metrics for the activity map."""
    metrics = summarize_activity_localization(
        voxel_positions=result["candidate_positions"],
        activity=result["activity"],
        true_sources=true_sources,
        max_relative_drop=relative_drop,
    )
    print_activity_localization_summary(metrics)


def main():
    # Change only this value while testing multi-source reconstruction methods.
    # Currently available: "mlem".
    reconstruction_method = "mlem"

    NUM_MEASUREMENTS = 30
    SEED = 11
    TRUE_SOURCE_SPECS = [
        {"position": (0.18, -0.12, 0.26), "relative_strength": 1.0},
        {"position": (-0.10, -0.18, 0.16), "relative_strength": 0.65},
        {"position": (0.34, -0.07, 0.38), "relative_strength": 0.45},
    ]
    NUM_CHANNELS = 16
    DIODE_PITCH = 0.006
    BACKGROUND_MEAN = 1000.0
    BACKGROUND_CHANNEL_STD = 50.0
    CHANNEL_SCALE_MEAN = 1.0
    CHANNEL_SCALE_STD = 0.08
    GENERATION_K = 4.5
    PEAK_TOTAL_SOURCE_SIGNAL = 5000.0
    ORIENTATION_MODE = "fixed"  # "fixed" or "towards_target"
    FIXED_VIEW_DIRECTION = (0, 1, 0)  # e.g. (1, 0, 0) to "look right" in world coords
    SCAN_CENTER = None
    ANGULAR_JITTER_DEG = 3.0
    LATERAL_SPAN = 0.60
    VERTICAL_SPAN = 0.36
    BASE_DISTANCE = 0.65
    DISTANCE_SPAN = 0.18

    dataset = generate_synthetic_dataset(
        num_measurements=NUM_MEASUREMENTS,
        seed=SEED,
        orientation_mode=ORIENTATION_MODE,
        fixed_view_direction=FIXED_VIEW_DIRECTION,
        source_specs=TRUE_SOURCE_SPECS,
        num_channels=NUM_CHANNELS,
        diode_pitch=DIODE_PITCH,
        background_mean=BACKGROUND_MEAN,
        background_channel_std=BACKGROUND_CHANNEL_STD,
        channel_scale_mean=CHANNEL_SCALE_MEAN,
        channel_scale_std=CHANNEL_SCALE_STD,
        generation_k=GENERATION_K,
        peak_total_source_signal=PEAK_TOTAL_SOURCE_SIGNAL,
        scan_center=SCAN_CENTER,
        angular_jitter_deg=ANGULAR_JITTER_DEG,
        lateral_span=LATERAL_SPAN,
        vertical_span=VERTICAL_SPAN,
        base_distance=BASE_DISTANCE,
        distance_span=DISTANCE_SPAN,
    )
    print_summary(dataset)

    num_channels = dataset["counts"].shape[1]

    # Reconstruction volume. With one-sided detector scans, this range matters a lot:
    # if it includes regions close to the detector, ML-EM may place activity there
    # because depth is weakly constrained.
    search_min = (-0.55, -0.65, -0.15)
    search_max = (0.75, 0.35, 0.75)

    ASSUMED_BACKGROUND_MEAN = 1000
    ASSUMED_CHANNEL_SCALE = 1.0
    ASSUMED_K = 4.0

    GRID_SIZE = 15
    NUM_ITERATIONS = 10000
    MAX_SOURCES_TO_REPORT = 3
    ACTIVITY_MAX_RELATIVE_DROP = 0.25

    assumed_model = {
        "assumed_backgrounds": np.full(num_channels, ASSUMED_BACKGROUND_MEAN, dtype=float),
        "assumed_channel_scales": np.full(num_channels, ASSUMED_CHANNEL_SCALE, dtype=float),
        "assumed_k": ASSUMED_K,
    }

    result = reconstruct_sources(
        counts=dataset["counts"],
        geometry=dataset["geometry"],
        rotations=dataset["rotations"],
        translations=dataset["translations"],
        **assumed_model,
        search_min=search_min,
        search_max=search_max,
        method=reconstruction_method,
        grid_size=GRID_SIZE,
        num_iterations=NUM_ITERATIONS,
        history_interval=1000,
        max_sources=MAX_SOURCES_TO_REPORT,
        min_source_distance=0.08,
    )
    print_reconstruction_summary(reconstruction_method, result)
    print_localization_metrics(
        result=result,
        true_sources=dataset["true_sources"],
        relative_drop=ACTIVITY_MAX_RELATIVE_DROP,
    )

    output_dir = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations"
    visualize_multi_source_scene(
        dataset,
        save_path=output_dir / "multi_source_scene.png",
        show=False,
    )
    visualize_mlem_activity(
        dataset,
        candidate_positions=result["candidate_positions"],
        activity=result["activity"],
        estimated_sources=result["estimated_sources"],
        save_path=output_dir / "mlem_activity_reconstruction.png",
        show=False,
    )
    visualize_history_score(
        result["history"],
        save_path=output_dir / "mlem_score_history.png",
        show=False,
    )

    print()
    print("Saved plots:")
    print(" ", output_dir / "multi_source_scene.png")
    print(" ", output_dir / "mlem_activity_reconstruction.png")
    print(" ", output_dir / "mlem_score_history.png")


if __name__ == "__main__":
    main()
