"""
Short entrypoint for the synthetic point-source reconstruction demo.
"""

from pathlib import Path

import numpy as np

from reconstruction import coarse_to_fine_grid_search
from simulation import generate_synthetic_dataset
from visualization import (
    visualize_detector_geometry,
    visualize_detector_trajectory_world,
)


def print_summary(dataset, estimated_position, estimated_strength):
    """Print a short summary to the terminal."""
    position_error = np.linalg.norm(estimated_position - dataset["true_source_position"])

    print("=== Synthetic point-source reconstruction demo ===")
    print()
    print("True source position:       ", np.round(dataset["true_source_position"], 4))
    print("Estimated source position:  ", np.round(estimated_position, 4))
    print(f"Position error:              {100.0 * position_error:.2f} cm")
    print()
    print(f"True source strength A:      {dataset['true_source_strength']:.2f}")
    print(f"Estimated source strength A: {estimated_strength:.2f}")
    print()
    print("First 5 measurements [channels 0..5]:")
    for row in dataset["counts"][:5]:
        print(" ", row.astype(int).tolist())


def main():
    dataset = generate_synthetic_dataset(num_measurements=24, seed=7)

    # Assumed model used during reconstruction.
    # It is intentionally a bit simpler than the model used for data generation.
    assumed_backgrounds = np.full(6, 10.0, dtype=float)
    assumed_channel_scales = np.ones(6, dtype=float)
    assumed_k = 4.0

    estimated_position, estimated_strength, history = coarse_to_fine_grid_search(
        counts=dataset["counts"],
        geometry=dataset["geometry"],
        rotations=dataset["rotations"],
        translations=dataset["translations"],
        assumed_backgrounds=assumed_backgrounds,
        assumed_channel_scales=assumed_channel_scales,
        assumed_k=assumed_k,
        search_min=(-0.40, -0.50, -0.10),
        search_max=(0.60, 0.30, 0.60),
        grid_sizes=(13, 11, 9),
    )

    print_summary(dataset, estimated_position, estimated_strength)
    print()
    print("Search history:")
    for item in history:
        print(
            " "
            f"level={int(item['level'])}, grid={int(item['grid_size'])}, "
            f"best=({item['best_x']:.4f}, {item['best_y']:.4f}, {item['best_z']:.4f}), "
            f"score={item['best_score']:.2f}"
        )

    output_dir = Path("visualizations")
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

    print()
    print("Saved plots:")
    print(" ", output_dir / "detector_geometry.png")
    print(" ", output_dir / "detector_trajectory_world.png")


if __name__ == "__main__":
    main()
