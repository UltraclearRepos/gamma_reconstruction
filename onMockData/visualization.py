"""
Visualization helpers for detector geometry and motion.
"""

from pathlib import Path

import numpy as np

from geometry import EPS


def set_equal_3d_axes(ax, points, margin=0.15):
    """
    Make 3D axes use the same visual scale on X, Y and Z.

    Without this, matplotlib often distorts the geometry.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_range = 0.5 * np.max(maxs - mins)

    if half_range < EPS:
        half_range = 1.0

    half_range *= 1.0 + margin

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)


def finalize_figure(fig, save_path=None, show=False):
    """Save figure to file and optionally display it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if show:
        import matplotlib.pyplot as plt

        plt.show()

    import matplotlib.pyplot as plt

    plt.close(fig)


def visualize_detector_geometry(geometry, save_path=None, show=False):
    """
    Show diode positions and their local viewing directions.

    This plot is in the detector local coordinate system.
    """
    import matplotlib.pyplot as plt

    positions = geometry["diode_positions_local"]
    normals = geometry["diode_normals_local"]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=60, color="tab:blue", label="diodes")
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="tab:blue",
        linewidth=1.5,
        alpha=0.65,
    )

    label_stride = 1 if len(positions) <= 8 else 2
    for index, (position, normal) in enumerate(zip(positions, normals)):
        ax.quiver(
            position[0],
            position[1],
            position[2],
            normal[0],
            normal[1],
            normal[2],
            length=0.03,
            color="tab:red",
            normalize=True,
        )
        if index % label_stride == 0 or index == len(positions) - 1:
            ax.text(position[0], position[1], position[2] + 0.008, f"ch{index}", fontsize=8)

    points_for_scale = np.vstack([positions, positions + 0.03 * normals, np.zeros((1, 3))])
    set_equal_3d_axes(ax, points_for_scale)

    ax.set_title("Linear detector geometry in local coordinates")
    ax.set_xlabel("x local [m]")
    ax.set_ylabel("y local [m]")
    ax.set_zlabel("z local [m]")
    ax.legend()

    finalize_figure(fig, save_path=save_path, show=show)


def visualize_detector_trajectory_world(dataset, save_path=None, show=False):
    """
    Show the detector path in world coordinates together with the source.
    """
    import matplotlib.pyplot as plt

    translations = dataset["translations"]
    rotations = dataset["rotations"]
    source_position = dataset["true_source_position"]
    forward_local = dataset["geometry"]["detector_forward_local"]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        linewidth=2.0,
        label="one-sided detector trajectory",
    )

    ax.scatter(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        s=18,
    )

    ax.scatter(
        source_position[0],
        source_position[1],
        source_position[2],
        color="tab:red",
        s=120,
        marker="*",
        label="source",
    )

    # Draw the actual detector forward direction used by the trajectory model.
    num_measurements = len(translations)

    for index in range(num_measurements):
        position = translations[index]
        forward = rotations[index] @ forward_local
        ax.quiver(
            position[0],
            position[1],
            position[2],
            forward[0],
            forward[1],
            forward[2],
            length=0.12,
            color="tab:green",
            normalize=True,
        )

    points_for_scale = np.vstack([translations, source_position[None, :]])
    set_equal_3d_axes(ax, points_for_scale)

    ax.set_title("One-sided detector scan in world coordinates")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_zlabel("z world [m]")
    ax.legend()

    finalize_figure(fig, save_path=save_path, show=show)


def visualize_source_fit_3d(
    xs,
    ys,
    zs,
    score_map,
    estimated_position,
    true_source_position=None,
    detector_positions=None,
    save_path=None,
    show=False,
    max_points=500,
):
    """
    Show where the source could be in 3D.

    score_map contains negative log-likelihood values, so lower is better.
    To keep the plot readable, we draw only the best-scoring grid points.
    """
    import matplotlib.pyplot as plt

    score_delta = score_map - np.min(score_map)
    flat_score_delta = score_delta.ravel()
    num_points = min(max_points, len(flat_score_delta))
    best_flat_indices = np.argpartition(flat_score_delta, num_points - 1)[:num_points]
    y_indices, x_indices, z_indices = np.unravel_index(best_flat_indices, score_map.shape)

    point_scores = np.log1p(flat_score_delta[best_flat_indices])
    point_sizes = 8.0 + 20.0 * (1.0 - point_scores / max(np.max(point_scores), EPS))

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        xs[x_indices],
        ys[y_indices],
        zs[z_indices],
        c=point_scores,
        s=point_sizes,
        cmap="viridis_r",
        alpha=0.25,
        depthshade=False,
    )

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
    colorbar.set_label("log(1 + score - best score), lower is better")

    if detector_positions is not None:
        ax.plot(
            detector_positions[:, 0],
            detector_positions[:, 1],
            detector_positions[:, 2],
            color="cyan",
            linewidth=1.5,
            alpha=0.75,
            label="detector path",
        )

    if true_source_position is not None:
        ax.scatter(
            true_source_position[0],
            true_source_position[1],
            true_source_position[2],
            marker="*",
            s=260,
            color="lime",
            edgecolor="black",
            linewidth=0.8,
            depthshade=False,
            label="true source",
        )

    ax.scatter(
        estimated_position[0],
        estimated_position[1],
        estimated_position[2],
        marker="X",
        s=170,
        color="red",
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        label="estimated source",
    )

    points_for_scale = [
        np.column_stack([xs[x_indices], ys[y_indices], zs[z_indices]]),
        estimated_position[None, :],
    ]
    if true_source_position is not None:
        points_for_scale.append(true_source_position[None, :])
    if detector_positions is not None:
        points_for_scale.append(detector_positions)

    set_equal_3d_axes(ax, np.vstack(points_for_scale))

    ax.set_title("Source fit map in 3D")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_zlabel("z world [m]")
    ax.legend(loc="upper right")

    finalize_figure(fig, save_path=save_path, show=show)
