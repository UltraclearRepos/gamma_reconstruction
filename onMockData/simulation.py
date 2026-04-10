"""
Synthetic data generation for the point-source reconstruction demo.
"""

import numpy as np

from forward_model import expected_counts
from geometry import make_detector_geometry, make_detector_trajectory


def generate_synthetic_dataset(
    num_measurements=24,
    seed=7,
):
    """
    Generate one synthetic experiment.

    A small model mismatch is introduced on purpose:
    - each channel has slightly different sensitivity,
    - backgrounds are not identical,
    - the true angular sensitivity exponent is not exactly the same as the
      one used later in reconstruction.
    """
    rng = np.random.default_rng(seed)

    geometry = make_detector_geometry(num_channels=6, detector_radius=0.04)
    true_source_position = np.array([0.18, -0.12, 0.26], dtype=float)
    true_source_strength = 190.0

    true_backgrounds = rng.uniform(8.0, 12.0, size=6)
    true_channel_scales = rng.normal(loc=1.0, scale=0.08, size=6)
    generation_k = 4.5

    rotations, translations = make_detector_trajectory(
        geometry=geometry,
        source_position=true_source_position,
        num_measurements=num_measurements,
        rng=rng,
        angular_jitter_deg=3.0,
    )

    mean_counts = expected_counts(
        source_position=true_source_position,
        source_strength=true_source_strength,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        channel_scales=true_channel_scales,
        backgrounds=true_backgrounds,
        k=generation_k,
    )

    # Photon counts are naturally modeled with a Poisson distribution.
    noisy_counts = rng.poisson(mean_counts).astype(float)

    return {
        "counts": noisy_counts,
        "rotations": rotations,
        "translations": translations,
        "geometry": geometry,
        "true_source_position": true_source_position,
        "true_source_strength": true_source_strength,
        "true_backgrounds": true_backgrounds,
        "true_channel_scales": true_channel_scales,
        "generation_k": generation_k,
    }
