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
        ax.text(position[0], position[1], position[2] + 0.008, f"ch{index}", fontsize=9)

    points_for_scale = np.vstack([positions, positions + 0.03 * normals, np.zeros((1, 3))])
    set_equal_3d_axes(ax, points_for_scale)

    ax.set_title("Detector geometry in local coordinates")
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
        label="detector trajectory",
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

    ax.set_title("Detector trajectory in world coordinates")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_zlabel("z world [m]")
    ax.legend()

    finalize_figure(fig, save_path=save_path, show=show)
