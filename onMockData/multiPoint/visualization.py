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


def visualize_mlem_activity(
    dataset,
    candidate_positions,
    activity,
    estimated_sources=None,
    save_path=None,
    show=False,
    max_voxels=300,
):
    """Show the strongest voxels from an ML-EM activity reconstruction."""
    import matplotlib.pyplot as plt

    translations = dataset["translations"]
    true_source_positions = np.array([source["position"] for source in dataset["true_sources"]])
    activity = np.asarray(activity, dtype=float)
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    estimated_sources = estimated_sources or []

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    points_for_scale = [translations, true_source_positions]

    if len(activity) > 0 and np.max(activity) > 0.0:
        num_voxels = min(max_voxels, len(activity))
        strongest_indices = np.argsort(activity)[-num_voxels:]
        voxel_positions = candidate_positions[strongest_indices]
        voxel_activity = activity[strongest_indices]

        normalized_activity = voxel_activity / max(float(np.max(voxel_activity)), EPS)
        voxel_sizes = 10.0 + 50.0 * normalized_activity

        scatter = ax.scatter(
            voxel_positions[:, 0],
            voxel_positions[:, 1],
            voxel_positions[:, 2],
            c=np.log1p(voxel_activity),
            s=voxel_sizes,
            cmap="inferno",
            alpha=0.30,
            depthshade=False,
            label="reconstructed activity",
        )
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
        colorbar.set_label("log(1 + activity)")
        points_for_scale.append(voxel_positions)

    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        linewidth=1.5,
        alpha=0.65,
        label="detector trajectory",
    )

    ax.scatter(
        true_source_positions[:, 0],
        true_source_positions[:, 1],
        true_source_positions[:, 2],
        color="lime",
        s=230,
        marker="*",
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        label="true sources",
    )

    if estimated_sources:
        estimated_positions = np.array([source["position"] for source in estimated_sources])
        ax.scatter(
            estimated_positions[:, 0],
            estimated_positions[:, 1],
            estimated_positions[:, 2],
            color="cyan",
            s=130,
            marker="X",
            edgecolor="black",
            linewidth=0.8,
            depthshade=False,
            label="estimated peaks",
        )
        points_for_scale.append(estimated_positions)

    set_equal_3d_axes(ax, np.vstack(points_for_scale))

    ax.set_title("ML-EM activity reconstruction")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_zlabel("z world [m]")
    ax.legend(loc="upper right")

    finalize_figure(fig, save_path=save_path, show=show)


def visualize_history_score(history, save_path=None, show=False):
    """Plot reconstruction score from history entries."""
    import matplotlib.pyplot as plt

    steps = []
    scores = []
    for index, item in enumerate(history, start=1):
        if "score" in item:
            steps.append(item.get("iteration", index))
            scores.append(item["score"])

    if not scores:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, scores, marker="o", linewidth=1.8, color="tab:purple")
    ax.set_title("Reconstruction score history")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Poisson negative log-likelihood")
    ax.grid(True, alpha=0.3)

    finalize_figure(fig, save_path=save_path, show=show)
