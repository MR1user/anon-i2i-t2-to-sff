"""Preprocessing helpers: intensity normalisation and orientation conversion."""

import numpy as np


def normalize_t2(t2: np.ndarray) -> np.ndarray:
    """Per-slice min-max normalisation to [0, 1]. This is the ONLY intensity
    preprocessing the models were trained with. The target (SFF) map is never
    normalised: it already lies in [0, 1]."""
    t2_min, t2_max = t2.min(), t2.max()
    return ((t2 - t2_min) / (t2_max - t2_min + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------
# The models were trained on arrays in a "raw" layout (see README).
#
# STANDARD layout (what most viewers show, radiological convention):
#   rows    : anterior (abdomen) at the top -> posterior (back) at the bottom
#   columns : patient's right on the image left -> patient's left on the right
#
# RAW layout (training layout) = STANDARD layout transposed and flipped:
#   raw = standard[::-1, ::-1].T
# Only square slices are supported (pure pixel permutation, no interpolation).

def standard_to_raw(arr: np.ndarray) -> np.ndarray:
    """Convert a slice in standard layout to the raw training layout."""
    return np.ascontiguousarray(arr[::-1, ::-1].T)


def raw_to_standard(arr: np.ndarray) -> np.ndarray:
    """Convert a slice in the raw training layout to standard layout
    (exact inverse of `standard_to_raw`)."""
    return np.ascontiguousarray(arr.T[::-1, ::-1])
