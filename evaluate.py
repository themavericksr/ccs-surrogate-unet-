"""
evaluate.py
-----------
Evaluation of the trained CCS U-Net surrogate model.

Computes:
    - R² and MAE for gas saturation (all timesteps + per-timestep)
    - R² and MAE for pressure buildup (all timesteps + per-timestep)
    - Side-by-side spatial comparison plots (predicted vs. actual)
    - Scatter plots of predicted vs. actual values

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pt --split val
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from torch.utils.data import DataLoader

import config as C
from dataset import CCSDataset
from model import CCSUNet


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(C.CKPT_DIR / "best_model.pt"))
    p.add_argument("--split",      default="val", choices=["train", "val"])
    p.add_argument("--n-plots",    type=int, default=3,
                   help="Number of sample plots to generate")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² computed over all elements (flattened)."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.flatten() - y_pred.flatten())))


# ---------------------------------------------------------------------------
# Inverse normalisation
# ---------------------------------------------------------------------------

def denorm_pressure(x: np.ndarray, norm: dict = C.NORM) -> np.ndarray:
    """Convert normalised pressure back to bar (pressure buildup units)."""
    return x * norm['pressure_std'] + norm['pressure_mean']


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get('config', {})

    model = CCSUNet(
        in_ch       = cfg.get('in_ch',       C.N_INPUT_CHANNELS),
        base_ch     = cfg.get('base_ch',      C.BASE_CHANNELS),
        n_timesteps = cfg.get('n_timesteps',  C.N_TIMESTEPS),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print(f"\n[INFO] Loaded checkpoint: {args.checkpoint}")
    print(f"[INFO] Trained to epoch {ckpt['epoch']+1}  |  Best val loss: {ckpt['best_val']:.5f}")

    # ── Dataset ───────────────────────────────────────────────────────────
    data_dir = C.VAL_DIR if args.split == "val" else C.TRAIN_DIR
    dataset  = CCSDataset(data_dir, augment=False)
    loader   = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"[INFO] Evaluating on {len(dataset)} samples from {data_dir.name}/\n")

    # ── Collect predictions ───────────────────────────────────────────────
    all_press_true, all_press_pred = [], []
    all_gas_true,   all_gas_pred   = [], []
    masks_list = []

    with torch.no_grad():
        for inputs, targets, mask in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            pred_p, pred_g = model(inputs)

            # Move to CPU numpy — keep only valid (unmasked) rows
            nz = mask[0].sum().item()

            # shapes: (1, T, GNZ, GNR) → (T, nz, NR)
            pp = pred_p[0, :, :nz, :].cpu().numpy()
            pg = pred_g[0, :, :nz, :].cpu().numpy()
            tp = targets[0, 0, :, :nz, :].cpu().numpy()
            tg = targets[0, 1, :, :nz, :].cpu().numpy()

            # Denormalise pressure
            pp = denorm_pressure(pp)
            tp = denorm_pressure(tp)

            all_press_pred.append(pp)
            all_press_true.append(tp)
            all_gas_pred.append(pg)
            all_gas_true.append(tg)
            masks_list.append(nz)

    # ── Global metrics ────────────────────────────────────────────────────
    # Concatenate along sample axis (each has different nz — use per-sample then average)
    press_r2_list, press_mae_list = [], []
    gas_r2_list,   gas_mae_list   = [], []

    for i in range(len(all_press_true)):
        press_r2_list.append(r2_score(all_press_true[i], all_press_pred[i]))
        press_mae_list.append(mae_score(all_press_true[i], all_press_pred[i]))
        gas_r2_list.append(r2_score(all_gas_true[i], all_gas_pred[i]))
        gas_mae_list.append(mae_score(all_gas_true[i], all_gas_pred[i]))

    print("=" * 55)
    print(f"  EVALUATION RESULTS  ({args.split} split, {len(dataset)} samples)")
    print("=" * 55)
    print(f"  Pressure buildup:")
    print(f"    R²  = {np.mean(press_r2_list):.4f}  ±  {np.std(press_r2_list):.4f}")
    print(f"    MAE = {np.mean(press_mae_list):.4f}  ±  {np.std(press_mae_list):.4f}  bar")
    print(f"  Gas saturation:")
    print(f"    R²  = {np.mean(gas_r2_list):.4f}  ±  {np.std(gas_r2_list):.4f}")
    print(f"    MAE = {np.mean(gas_mae_list):.4f}  ±  {np.std(gas_mae_list):.4f}")
    print("=" * 55)

    # ── Per-timestep metrics ──────────────────────────────────────────────
    T = C.N_TIMESTEPS
    ts_press_r2  = np.zeros(T)
    ts_gas_r2    = np.zeros(T)

    for i in range(len(all_press_true)):
        for t in range(T):
            ts_press_r2[t] += r2_score(all_press_true[i][t], all_press_pred[i][t])
            ts_gas_r2[t]   += r2_score(all_gas_true[i][t],   all_gas_pred[i][t])

    ts_press_r2 /= len(all_press_true)
    ts_gas_r2   /= len(all_gas_true)

    _save_timestep_plot(ts_press_r2, ts_gas_r2)

    # ── Scatter plot ──────────────────────────────────────────────────────
    _save_scatter_plot(all_press_true, all_press_pred, all_gas_true, all_gas_pred)

    # ── Spatial comparison plots ──────────────────────────────────────────
    n_plots = min(args.n_plots, len(dataset))
    for i in range(n_plots):
        _save_spatial_plot(
            i,
            all_press_true[i], all_press_pred[i],
            all_gas_true[i],   all_gas_pred[i],
        )

    print(f"\n[INFO] All plots saved to {C.OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

TIMES = [
    '1d','2d','4d','7d','11d','17d','25d','37d','53d','77d','111d','158d',
    '226d','323d','1.3y','1.8y','2.6y','3.6y','5.2y','7.3y','10.4y','14.8y',
    '21.1y','30y'
]


def _save_timestep_plot(press_r2: np.ndarray, gas_r2: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    t = range(len(press_r2))

    for ax, r2, title, color in zip(
        axes,
        [press_r2, gas_r2],
        ['Pressure Buildup R² by Timestep', 'Gas Saturation R² by Timestep'],
        ['#1A5276', '#117A65'],
    ):
        ax.plot(t, r2, 'o-', color=color, lw=2, ms=5)
        ax.axhline(np.mean(r2), color='grey', ls='--', lw=1, label=f'Mean R² = {np.mean(r2):.3f}')
        ax.set_xticks(range(0, len(TIMES), 4))
        ax.set_xticklabels(TIMES[::4], rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("R²")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = C.OUTPUT_DIR / "r2_by_timestep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Timestep R² plot → {path}")


def _save_scatter_plot(press_true, press_pred, gas_true, gas_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, true_list, pred_list, title, unit, color in zip(
        axes,
        [press_true, gas_true],
        [press_pred, gas_pred],
        ['Pressure Buildup', 'Gas Saturation'],
        ['bar', '—'],
        ['#1A5276', '#117A65'],
    ):
        # Subsample for speed
        y_true = np.concatenate([a.flatten()[::50] for a in true_list])
        y_pred = np.concatenate([a.flatten()[::50] for a in pred_list])

        ax.scatter(y_true, y_pred, s=2, alpha=0.3, color=color)
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1, label='1:1 line')
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{title}  (R² = {r2:.4f})")
        ax.set_xlabel(f"Simulated ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = C.OUTPUT_DIR / "scatter_predicted_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Scatter plot → {path}")


def _save_spatial_plot(
    sample_idx: int,
    press_true: np.ndarray,  # (T, nz, nr)
    press_pred: np.ndarray,
    gas_true:   np.ndarray,
    gas_pred:   np.ndarray,
    timesteps_to_plot: list = [0, 6, 12, 18, 23],
):
    """Side-by-side spatial maps at selected timesteps."""
    nts = len(timesteps_to_plot)
    fig = plt.figure(figsize=(4 * nts, 10))
    gs  = gridspec.GridSpec(4, nts, figure=fig, hspace=0.4, wspace=0.3)

    for col, t in enumerate(timesteps_to_plot):
        for row, (data, cmap, label) in enumerate([
            (press_true[t],         'RdYlBu_r', f'Press True  t={TIMES[t]}'),
            (press_pred[t],         'RdYlBu_r', f'Press Pred  t={TIMES[t]}'),
            (gas_true[t],           'YlOrRd',   f'Gas Sat True  t={TIMES[t]}'),
            (gas_pred[t],           'YlOrRd',   f'Gas Sat Pred  t={TIMES[t]}'),
        ]):
            ax  = fig.add_subplot(gs[row, col])
            im  = ax.imshow(data, aspect='auto', cmap=cmap, origin='upper')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(label, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"Sample {sample_idx} — Predicted vs. Simulated", fontsize=11, y=1.01)
    path = C.OUTPUT_DIR / f"spatial_sample_{sample_idx:03d}.png"
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Spatial plot → {path}")


if __name__ == "__main__":
    evaluate()
