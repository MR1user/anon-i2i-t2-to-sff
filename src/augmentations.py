"""
Generic ("naive") augmentation for the T2w input, following the default
nnU-Net augmentation ranges (Isensee et al., 2021, Nature Methods 18:203-211).

Applied to the T2w input ONLY, BEFORE min-max normalisation. The SFF target
is never modified.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

NNUNET_DEFAULT_RANGES = dict(
    gamma_range=(0.7, 1.5),
    brightness_factor_range=(0.75, 1.25),
    gaussian_noise_sigma_range=(0.0, 0.1),   # fraction of the intensity range
    gaussian_blur_sigma_range=(0.5, 1.5),    # pixels
)


def naive_augmentation_t2_only(
    t2_raw: np.ndarray,
    rng: np.random.Generator,
    ranges: dict = NNUNET_DEFAULT_RANGES,
) -> np.ndarray:
    """Apply gamma correction, brightness scaling, additive Gaussian noise and
    Gaussian blur (in this order), each with a parameter drawn uniformly from
    `ranges`. Output has the same shape as the input and is NOT normalised."""
    out = t2_raw.astype(np.float32).copy()

    # Gamma correction on a locally [0, 1]-rescaled image, then rescale back
    gamma = rng.uniform(*ranges["gamma_range"])
    v_min, v_max = out.min(), out.max()
    span = v_max - v_min + 1e-8
    unit = np.clip((out - v_min) / span, 0.0, 1.0) ** gamma
    out = unit * span + v_min

    # Brightness
    out = out * rng.uniform(*ranges["brightness_factor_range"])

    # Additive Gaussian noise (sigma relative to the current intensity range)
    sigma_fraction = rng.uniform(*ranges["gaussian_noise_sigma_range"])
    sigma_abs = sigma_fraction * (out.max() - out.min() + 1e-8)
    out = out + rng.normal(0.0, sigma_abs, size=out.shape)

    # Blur
    out = gaussian_filter(out, sigma=rng.uniform(*ranges["gaussian_blur_sigma_range"]))

    return out.astype(np.float32)
