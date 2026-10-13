"""
Train the T2w -> SFF U-Net, with or without generic (nnU-Net-style) augmentation.

    python train.py --train_dir data/train --val_dir data/val --out_dir runs/baseline  --no-augment
    python train.py --train_dir data/train --val_dir data/val --out_dir runs/augmented --augment

Data format: see README.md ("Training data format").
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from src.model import UNet
from src.preprocessing import normalize_t2
from src.augmentations import naive_augmentation_t2_only

# ---- fixed hyper-parameters (identical for both models, for a fair comparison)
LR = 2e-4
VAL_EVERY = 5          # validate every N epochs
VAL_SUBSET_N = 500     # fixed validation subset size
WARMUP_EPOCHS = 5      # early stopping disabled during warm-up
PATIENCE = 10          # validation rounds without improvement
MIN_DELTA = 5e-4
LR_PATIENCE, LR_FACTOR, LR_MIN = 3, 0.5, 1e-6
AUGMENT_PROB = 0.7     # probability of augmenting a given training slice
VAL_AUGMENT_SEED = 4242


class SliceDataset(Dataset):
    """Returns the RAW (T2, SFF) arrays of one slice; augmentation and
    normalisation happen later, per batch."""

    def __init__(self, files, input_key, target_key):
        self.files, self.input_key, self.target_key = files, input_key, target_key

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        return (torch.from_numpy(d[self.input_key].astype(np.float32)),
                torch.from_numpy(d[self.target_key].astype(np.float32)))


def make_batch(t2_np, sff_np, augment, rng):
    """Optional per-slice augmentation (T2 only) -> min-max normalisation.
    The SFF target is NEVER modified or normalised."""
    t2_out = []
    for t2 in t2_np:
        if augment and rng.random() < AUGMENT_PROB:
            t2 = naive_augmentation_t2_only(t2, rng)
        t2_out.append(normalize_t2(t2))
    x = torch.from_numpy(np.stack(t2_out)).unsqueeze(1)
    y = torch.from_numpy(sff_np.astype(np.float32)).unsqueeze(1)
    return x, y


def build_frozen_val(files, input_key, target_key, augment):
    """Fixed validation subset. If augmenting, it is augmented ONCE with a
    fixed seed and kept frozen, so the validation metric stays comparable
    across epochs."""
    rng = np.random.default_rng(VAL_AUGMENT_SEED)
    xs, ys = [], []
    for f in files:
        d = np.load(f)
        x, y = make_batch(d[input_key][None].astype(np.float32),
                          d[target_key][None], augment, rng)
        xs.append(x), ys.append(y)
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def validate(model, x_val, y_val, device, batch_size):
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    model.eval()
    scores = []
    for i in range(0, len(x_val), batch_size):
        pred = model(x_val[i:i + batch_size].to(device))
        scores.append(ssim_fn(pred, y_val[i:i + batch_size].to(device)).item())
    return float(np.mean(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False,
                   help="--augment: generic nnU-Net-style augmentation; --no-augment: none")
    p.add_argument("--input_key", default="T2", help="array name of the T2w slice in each .npz")
    p.add_argument("--target_key", default="SFF", help="array name of the SFF map in each .npz")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument("--init_weights", default=None,
                   help="optional .pth with shared initial weights")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_files = sorted(glob.glob(os.path.join(args.train_dir, "*.npz")))
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*.npz")))
    if not train_files or not val_files:
        raise FileNotFoundError("No .npz files found in --train_dir / --val_dir.")

    # fixed validation subset (seed 0), identical for every run
    sel = np.random.default_rng(0).choice(
        len(val_files), size=min(VAL_SUBSET_N, len(val_files)), replace=False)
    val_files = [val_files[i] for i in sel]
    x_val, y_val = build_frozen_val(val_files, args.input_key, args.target_key, args.augment)

    loader = DataLoader(
        SliceDataset(train_files, args.input_key, args.target_key),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, generator=torch.Generator().manual_seed(args.seed))

    model = UNet().to(device)
    if args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location=device, weights_only=True))
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN)
    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_ssim, best_epoch, stagnant = -1.0, 0, 0
    print(f"Device: {device} | augmentation: {'ON' if args.augment else 'OFF'} | "
          f"train slices: {len(train_files)} | val slices: {len(x_val)}")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for t2_raw, sff_raw in tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False):
            x, y = make_batch(t2_raw.numpy(), sff_raw.numpy(), args.augment, rng)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch:03d} | train L1 {total / len(loader):.5f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if epoch % VAL_EVERY == 0:
            ssim = validate(model, x_val, y_val, device, args.batch_size)
            scheduler.step(ssim)
            if ssim >= best_ssim + MIN_DELTA:
                best_ssim, best_epoch, stagnant = ssim, epoch, 0
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
                print(f"  val SSIM {ssim:.4f} (new best, checkpoint saved)")
            else:
                stagnant += 1
                print(f"  val SSIM {ssim:.4f} (best {best_ssim:.4f}, stagnation {stagnant}/{PATIENCE})")
            if epoch > WARMUP_EPOCHS and stagnant >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    torch.save(model.state_dict(), os.path.join(args.out_dir, "final.pth"))
    print(f"Done in {(time.time() - start) / 60:.1f} min. "
          f"Best val SSIM {best_ssim:.4f} at epoch {best_epoch}. Files in {args.out_dir}")


if __name__ == "__main__":
    main()
