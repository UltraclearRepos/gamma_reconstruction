"""
Entrypoint for the synthetic multi-point source demo.
"""

from pathlib import Path
import sys

import numpy as np

ON_MOCK_DATA_DIR = Path(__file__).resolve().parents[1]
if str(ON_MOCK_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(ON_MOCK_DATA_DIR))

from simulation import generate_synthetic_dataset
from visualization import visualize_multi_source_scene


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


def main():
    dataset = generate_synthetic_dataset(num_measurements=30, seed=11)
    print_summary(dataset)

    output_dir = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations"
    visualize_multi_source_scene(
        dataset,
        save_path=output_dir / "multi_source_scene.png",
        show=False,
    )

    print()
    print("Saved plots:")
    print(" ", output_dir / "multi_source_scene.png")


if __name__ == "__main__":
    main()
