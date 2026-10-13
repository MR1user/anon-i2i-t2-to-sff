"""
Apply a trained model to a NEW, unseen axial T2w slice and obtain a
synthetic SFF (signal fat fraction) map.

Example:
    python infer.py --input my_slice.npy --model baseline  --orientation standard
    python infer.py --input my_slice.npz --key T2 --model augmented --orientation standard

See README.md for the exact input requirements.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import UNet
from src.preprocessing import normalize_t2, standard_to_raw, raw_to_standard

DEFAULT_WEIGHTS = {
    "baseline": "weights/unet_baseline.pth",
    "augmented": "weights/unet_augmented.pth",
}


def load_slice(path: str, key: str) -> np.ndarray:
    """Load a 2D slice from .npy, or from .npz (array stored under `key`)."""
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        if key in data.files:
            arr = data[key]
        elif len(data.files) == 1:
            arr = data[data.files[0]]
        else:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {data.files}")
    else:
        raise ValueError("Input must be a .npy or .npz file.")
    return np.asarray(arr, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="T2w -> SFF inference on a single 2D slice.")
    p.add_argument("--input", required=True, help=".npy or .npz file with ONE 2D slice")
    p.add_argument("--key", default="T2", help="array name inside a .npz file (default: T2)")
    p.add_argument("--model", required=True, choices=["baseline", "augmented"])
    p.add_argument("--weights", default=None, help="override the default weights path")
    p.add_argument("--orientation", required=True, choices=["standard", "raw"],
                   help="orientation of your input array (see README). "
                        "'standard': anterior at top, patient right on image left. "
                        "'raw': layout used for training.")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # ---- load & validate -------------------------------------------------
    t2 = load_slice(args.input, args.key)
    if t2.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {t2.shape}.")
    h, w = t2.shape
    if h != w:
        raise ValueError(f"Slice must be square, got {t2.shape} (256x256 recommended).")
    if h % 16 != 0:
        raise ValueError(f"Slice size must be a multiple of 16, got {h} (256 recommended).")
    if not np.isfinite(t2).all():
        raise ValueError("Input contains NaN/Inf values.")

    # ---- bring input into the training layout, normalise ------------------
    t2_raw_layout = standard_to_raw(t2) if args.orientation == "standard" else t2
    x = torch.from_numpy(normalize_t2(t2_raw_layout))[None, None].to(args.device)

    # ---- model --------------------------------------------------------------
    weights_path = args.weights or DEFAULT_WEIGHTS[args.model]
    model = UNet().to(args.device)
    model.load_state_dict(torch.load(weights_path, map_location=args.device, weights_only=True))
    model.eval()

    with torch.no_grad():
        sff_raw_layout = model(x)[0, 0].cpu().numpy()

    # ---- return result in the SAME orientation as the input -----------------
    sff = raw_to_standard(sff_raw_layout) if args.orientation == "standard" else sff_raw_layout

    # ---- save ---------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    npy_path = os.path.join(args.output_dir, f"{stem}_{args.model}_sff.npy")
    png_path = os.path.join(args.output_dir, f"{stem}_{args.model}_sff.png")
    np.save(npy_path, sff.astype(np.float32))

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(t2, cmap="gray")
    ax[0].set_title("Input T2w")
    im = ax[1].imshow(sff, cmap="jet", vmin=0, vmax=1)
    ax[1].set_title(f"Synthetic SFF ({args.model})")
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Saved: {npy_path}\nSaved: {png_path}")


if __name__ == "__main__":
    main()
