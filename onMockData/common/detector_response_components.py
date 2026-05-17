"""
Physical response components used by the gamma detector forward model.
"""

import math

import numpy as np

from common.geometry import EPS


def angular_response(cos_theta, k, angular_peak_angle_deg=0.0, angular_width_deg=None):
    """
    Angular acceptance of the probe/detector head.

    cos_theta <= 0 means the source is behind the diode and the response is 0.
    If angular_width_deg is not provided, this falls back to the old cos(theta)^k
    model. Otherwise it uses a Gaussian-like angular curve that can peak away
    from 0 degrees, which is closer to a calibrated detector response table.
    """
    cos_theta = float(cos_theta)
    if cos_theta <= 0.0:
        return 0.0

    if angular_width_deg is None:
        return float(np.maximum(0.0, cos_theta) ** k)

    theta = math.degrees(math.acos(float(np.clip(cos_theta, -1.0, 1.0))))
    peak = float(angular_peak_angle_deg)
    width = max(float(angular_width_deg), EPS)
    return math.exp(-0.5 * ((theta - peak) / width) ** 2)


def skin_attenuation(cos_theta, skin_thickness_cm, skin_mu_per_cm):
    """
    Attenuation through a thin skin slab directly in front of the detector.

    The slab is approximated as locally parallel to the detector face. Oblique
    rays travel a longer path through the same skin thickness.
    """
    thickness = float(skin_thickness_cm)
    mu = float(skin_mu_per_cm)
    if thickness <= 0.0 or mu <= 0.0:
        return 1.0

    path_length = thickness / max(float(cos_theta), 0.05)
    return math.exp(-mu * path_length)


def channel_surface_offsets(geometry):
    """
    Return local offsets and patch area for numerical integration over a channel.
    """
    width = float(geometry.get("channel_width", geometry.get("diode_pitch", 0.0)))
    height = float(geometry.get("channel_height", width))
    sample_count = int(geometry.get("surface_sample_count", 3))
    sample_count = max(1, sample_count)

    if width <= 0.0 or height <= 0.0:
        return np.zeros((1, 3), dtype=float), 1.0

    xs = np.linspace(
        -0.5 * width + 0.5 * width / sample_count,
        0.5 * width - 0.5 * width / sample_count,
        sample_count,
    )
    ys = np.linspace(
        -0.5 * height + 0.5 * height / sample_count,
        0.5 * height - 0.5 * height / sample_count,
        sample_count,
    )
    offsets = np.array(
        [[x, y, 0.0] for x in xs for y in ys],
        dtype=float,
    )
    patch_area = width * height / float(sample_count * sample_count)
    return offsets, patch_area


def solid_angle_fraction(patch_area, cos_theta, distance):
    """
    Approximate the fraction of isotropic photons intercepted by one surface patch.
    """
    return float(patch_area) * float(cos_theta) / max(
        4.0 * math.pi * float(distance) * float(distance),
        EPS,
    )


def empirical_distance_factor(distance, distance_power):
    """
    Optional extra distance falloff beyond the solid-angle term.

    Keep distance_power at 0.0 when the finite detector surface already provides
    the desired physical 1/r^2 falloff.
    """
    distance_power = float(distance_power)
    if distance_power < 0.0:
        raise ValueError("distance_power must be >= 0.")
    return 1.0 / max(float(distance) ** distance_power, EPS)
