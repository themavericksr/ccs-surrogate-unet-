"""
predict.py
----------
Single-sample inference — useful for the demo notebook and quick sanity checks.

Usage:
    python predict.py --file data/val_data/data_0014.npz
    python predict.py --file data/val_data/data_0014.npz --timestep 23
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import config as C
from dataset import CCSDataset, _standardise, _build_perf_mask
from model import CCSUNet
from evaluate import denorm_pressure, r2_score, mae_score, TIMES


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file",       required=True, help="Path to .npz sample file")
    p.add_argument("--checkpoint", default=str(C.CKPT_DIR / "best_model.pt"))
    p.add_argument("--timestep",   type=int, default=23,
                   help="Timestep index to visualise (0–23, default: 23 = 30 years)")
    return p.parse_args()


def load_single_sample(fpath: str):
    """Load a single .npz file and return inputs, targets, mask (all as numpy)."""
    ds = CCSDataset(str(Path(fpath).parent))
    # Find the file in the dataset
    fname = Path(fpath).name
    if fname not in ds.files:
        raise FileNotFoundError(f"{fname} not found in {Path(fpath).parent}")
    idx = ds.files.index(fname)
    inputs, targets, mask = ds[idx]
    return inputs.unsqueeze(0), targets.unsqueeze(0), mask.unsqueeze(0)


def predict():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get('config', {})
    model = CCSUNet(
        in_ch       = cfg.get('in_ch',      C.N_INPUT_CHANNELS),
        base_ch     = cfg.get('base_ch',    C.BASE_CHANNELS),
        n_timesteps = cfg.get('n_timesteps', C.N_TIMESTEPS),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[INFO] Model loaded from {args.checkpoint}")

    # Load sample
    inputs, targets, mask = load_single_sample(args.file)
    inputs  = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        pred_p, pred_g = model(inputs)

    nz = mask[0].sum().item()
    t  = args.timestep

    # Unpad and denorm
    pp = denorm_pressure(pred_p[0, t, :nz, :].cpu().numpy())
    tp = denorm_pressure(targets[0, 0, t, :nz, :].cpu().numpy())
    pg = pred_g[0, t, :nz, :].cpu().numpy()
    tg = targets[0, 1, t, :nz, :].cpu().numpy()

    # Metrics
    print(f"\n  Timestep: {TIMES[t]}")
    print(f"  Pressure  R²={r2_score(tp, pp):.4f}  MAE={mae_score(tp, pp):.4f} bar")
    print(f"  Gas sat   R²={r2_score(tg, pg):.4f}  MAE={mae_score(tg, pg):.4f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(
        f"CCS Surrogate Prediction — {Path(args.file).name}  |  t = {TIMES[t]}",
        fontsize=12
    )

    for row, (true, pred, title, cmap) in enumerate([
        (tp, pp, 'Pressure Buildup (bar)', 'RdYlBu_r'),
        (tg, pg, 'Gas Saturation',         'YlOrRd'),
    ]):
        vmin = min(true.min(), pred.min())
        vmax = max(true.max(), pred.max())

        im0 = axes[row, 0].imshow(true, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axes[row, 0].set_title(f"Simulated — {title}")
        plt.colorbar(im0, ax=axes[row, 0])

        im1 = axes[row, 1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axes[row, 1].set_title(f"Predicted — {title}")
        plt.colorbar(im1, ax=axes[row, 1])

        err = pred - true
        im2 = axes[row, 2].imshow(err, cmap='bwr',
                                   vmin=-np.abs(err).max(), vmax=np.abs(err).max(),
                                   aspect='auto')
        axes[row, 2].set_title(f"Error (Pred − Simulated)")
        plt.colorbar(im2, ax=axes[row, 2])

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out = C.OUTPUT_DIR / f"prediction_{Path(args.file).stem}_t{t}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot saved → {out}")


if __name__ == "__main__":
    predict()
