"""
Create SYNTHETIC dummy data to test the training and inference pipeline
without real MRI data. The images are random ellipses, NOT anatomical, and
the models trained on them are meaningless: this only checks that the code
runs end to end.

    python examples/make_dummy_data.py
    python train.py --train_dir examples/dummy_data/train --val_dir examples/dummy_data/val \
                    --out_dir runs/dummy --augment --epochs 5 --num_workers 0
    python infer.py --input examples/dummy_data/dummy_input.npy --model baseline \
                    --weights runs/dummy/final.pth --orientation raw
"""

import argparse
import os

import numpy as np


def make_slice(rng, size=256):
    yy, xx = np.mgrid[:size, :size]
    sff = np.zeros((size, size), np.float32)
    for _ in range(6):
        cy, cx = rng.uniform(60, size - 60, 2)
        ry, rx = rng.uniform(10, 40, 2)
        sff[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1] = rng.uniform(0.05, 0.9)
    t2 = 200.0 * (0.3 + 0.7 * sff) + rng.normal(0, 10, sff.shape)
    return np.clip(t2, 0, None).astype(np.float32), sff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="examples/dummy_data")
    p.add_argument("--n_train", type=int, default=64)
    p.add_argument("--n_val", type=int, default=16)
    args = p.parse_args()
    rng = np.random.default_rng(0)

    for split, n in (("train", args.n_train), ("val", args.n_val)):
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)
        for i in range(n):
            t2, sff = make_slice(rng)
            np.savez_compressed(os.path.join(args.out_dir, split, f"slice_{i:04d}.npz"),
                                T2=t2, SFF=sff)

    t2, _ = make_slice(rng)
    np.save(os.path.join(args.out_dir, "dummy_input.npy"), t2)
    print(f"Dummy data written to {args.out_dir}")


if __name__ == "__main__":
    main()
