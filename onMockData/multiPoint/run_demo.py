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


def main():
    # Change only this value while testing multi-source reconstruction methods.
    # Currently available: "mlem".
    reconstruction_method = "mlem"

    dataset = generate_synthetic_dataset(num_measurements=30, seed=11)
    print_summary(dataset)

    num_channels = dataset["counts"].shape[1]

    # Reconstruction volume. With one-sided detector scans, this range matters a lot:
    # if it includes regions close to the detector, ML-EM may place activity there
    # because depth is weakly constrained.
    search_min = (-0.35, -0.45, -0.05)
    search_max = (0.55, 0.20, 0.55)

    ASSUMED_BACKGROUND_MEAN = 10000
    ASSUMED_CHANNEL_SCALE = 1.0
    ASSUMED_K = 4.0

    GRID_SIZE = 11
    NUM_ITERATIONS = 10000
    MAX_SOURCES_TO_REPORT = 3

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
