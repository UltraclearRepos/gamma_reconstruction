"""
Visualization helpers for multi-point source experiments.
"""

import numpy as np

from common.geometry import EPS
from common.visualization import finalize_figure, set_equal_3d_axes


def visualize_multi_source_scene(dataset, save_path=None, show=False):
    """Show detector trajectory and all true point sources in world coordinates."""
    import matplotlib.pyplot as plt

    translations = dataset["translations"]
    source_positions = np.array([source["position"] for source in dataset["true_sources"]])
    source_strengths = np.array([source["strength"] for source in dataset["true_sources"]])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        linewidth=2.0,
        label="detector trajectory",
    )
    ax.scatter(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        s=18,
    )

    max_strength = max(float(np.max(source_strengths)), EPS)
    source_sizes = 120.0 + 180.0 * source_strengths / max_strength
    ax.scatter(
        source_positions[:, 0],
        source_positions[:, 1],
        source_positions[:, 2],
        color="tab:red",
        s=source_sizes,
        marker="*",
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        label="true sources",
    )

    for index, source in enumerate(dataset["true_sources"], start=1):
        position = source["position"]
        ax.text(
            position[0],
            position[1],
            position[2] + 0.035,
            f"S{index}",
            fontsize=9,
        )

    points_for_scale = np.vstack([translations, source_positions])
    set_equal_3d_axes(ax, points_for_scale)

    ax.set_title("Multi-point source scene")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_zlabel("z world [m]")
    ax.legend(loc="upper right")

    finalize_figure(fig, save_path=save_path, show=show)
