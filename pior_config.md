1. Good priors for case: mine
<details>
        "RECONSTRUCTION_PRIORS": [
            # {"type": "smoothness", "beta": 0.1},  # Penalizes differences from the six-neighbor local mean; smooths noisy voxel spikes.
            # {"type": "huber", "beta": 0.1, "delta": 0.3},  # Smooths small neighbor differences while preserving sharper blob edges; delta is the activity-difference scale where the penalty changes from quadratic-like to edge-preserving, try 0.1-5.0 depending on reconstructed activity scale.
            # {"type": "rdp", "beta": 0.001, "gamma": 0.3},  # Relative Difference Penalty; smooths weak/noisy regions while being gentler near strong activity contrast; gamma controls edge preservation in the relative denominator, try 0.1-3.0, with larger values preserving stronger contrasts.
            {"type": "tv", "beta": 0.1, "epsilon": 0.01},  # Smoothed total variation; encourages compact blobs with sharper boundaries than Gaussian sieve.
            {"type": "l1", "beta": 0.1},  # Adds sparsity pressure; discourages diffuse low activity spread across many voxels.
        ],  # Priors shared by MAP-EM and BSREM

</details>