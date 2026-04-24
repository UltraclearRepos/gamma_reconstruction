"""
Shared visualization helpers.
"""

from pathlib import Path

import numpy as np

from common.geometry import EPS


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


def set_3d_axis_limits(ax, axis_limits):
    """Set fixed 3D axis limits from ((xmin, xmax), (ymin, ymax), (zmin, zmax))."""
    x_limits, y_limits, z_limits = axis_limits
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_zlim(*z_limits)


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
