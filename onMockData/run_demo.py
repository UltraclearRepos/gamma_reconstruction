"""
Short entrypoint for the synthetic point-source reconstruction demo.
"""

from pathlib import Path

import numpy as np

from reconstruction_algorithms import reconstruct_source
from simulation import generate_synthetic_dataset
from visualization import (
    visualize_detector_geometry,
    visualize_detector_trajectory_world,
    visualize_reconstruction_history_3d,
)


def print_summary(dataset, method, estimated_position, estimated_strength, history):
    """Print a short summary to the terminal."""
    position_error = np.linalg.norm(estimated_position - dataset["true_source_position"])
    num_channels = dataset["counts"].shape[1]
    best_score = history[-1].get("best_score") if history else None

    print("=== Synthetic point-source reconstruction demo ===")
    print(f"Reconstruction method:       {method}")
    print()
    print("True source position:       ", np.round(dataset["true_source_position"], 4))
    print("Estimated source position:  ", np.round(estimated_position, 4))
    print(f"Position error:              {100.0 * position_error:.2f} cm")
    print()
    print(f"True source strength A:      {dataset['true_source_strength']:.2f}")
    print(f"Estimated source strength A: {estimated_strength:.2f}")
    if best_score is not None:
        print(f"Best score:                  {best_score:.2f}")
    print()
    print(f"First 5 measurements [channels 0..{num_channels - 1}]:")
    for row in dataset["counts"][:5]:
        print(" ", row.astype(int).tolist())


def format_history_value(key, value):
    """Format one history value without dumping huge numpy arrays."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return f"array(shape={value.shape}, empty)"

        if value.ndim == 1 and value.size <= 6:
            return str(np.round(value, 4).tolist())

        if key.endswith("_scores"):
            return (
                f"array(shape={value.shape}, "
                f"min={np.min(value):.2f}, max={np.max(value):.2f})"
            )

        return f"array(shape={value.shape})"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def print_history(history):
    print("Search history:")
    for index, item in enumerate(history, start=1):
        print(f" step={index}")
        for key, value in item.items():
            print(f"  {key}: {format_history_value(key, value)}")


def main():
    # Change only this value while testing reconstruction methods:
    # "coarse_to_fine", "grid_search", or "random_search".
    reconstruction_method = "random_search"

    dataset = generate_synthetic_dataset(num_measurements=24, seed=7)
    num_channels = dataset["counts"].shape[1]
    search_min = (-0.40, -0.50, -0.10)
    search_max = (0.60, 0.30, 0.60)

    # Assumed model used during reconstruction.
    # It is intentionally a bit simpler than the model used for data generation.
    assumed_backgrounds = np.full(num_channels, 10.0, dtype=float)
    assumed_channel_scales = np.ones(num_channels, dtype=float)
    assumed_k = 4.0

    estimated_position, estimated_strength, history = reconstruct_source(
        counts=dataset["counts"],
        geometry=dataset["geometry"],
        rotations=dataset["rotations"],
        translations=dataset["translations"],
        assumed_backgrounds=assumed_backgrounds,
        assumed_channel_scales=assumed_channel_scales,
        assumed_k=assumed_k,
        search_min=search_min,
        search_max=search_max,
        method=reconstruction_method,
        grid_sizes=(13, 11, 9),
        grid_size=21,
        random_samples=4000,
        random_seed=0,
    )

    print_summary(
        dataset,
        reconstruction_method,
        estimated_position,
        estimated_strength,
        history,
    )
    print()
    print_history(history)

    output_dir = Path("onMockData/visualizations")
    visualize_detector_geometry(
        dataset["geometry"],
        save_path=output_dir / "detector_geometry.png",
        show=False,
    )
    visualize_detector_trajectory_world(
        dataset,
        save_path=output_dir / "detector_trajectory_world.png",
        show=False,
    )

    visualize_reconstruction_history_3d(
        history=history,
        estimated_position=estimated_position,
        true_source_position=dataset["true_source_position"],
        detector_positions=dataset["translations"],
        save_path=output_dir / f"reconstruction_history_3d_{reconstruction_method}.png",
        max_points=200,
        show=False,
    )

    print()
    print("Saved plots:")
    print(" ", output_dir / "detector_geometry.png")
    print(" ", output_dir / "detector_trajectory_world.png")
    print(" ", output_dir / f"reconstruction_history_3d_{reconstruction_method}.png")


if __name__ == "__main__":
    main()
