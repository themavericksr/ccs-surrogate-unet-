"""
train.py
--------
Training loop for the CCS dual-output U-Net surrogate model.

Features:
    - Masked MSE loss (ignores padded rows)
    - Early stopping on validation loss
    - StepLR learning rate schedule
    - Checkpointing: best model saved to checkpoints/best_model.pt
    - Loss curve saved to outputs/loss_curve.png
    - Full training log to outputs/train_log.csv

Usage:
    python train.py
    python train.py --epochs 100 --batch-size 4 --lr 5e-4
"""

import argparse
import time
import csv
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import config as C
from dataset import CCSDataset
from model import CCSUNet, MaskedLoss


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train CCS U-Net surrogate model")
    p.add_argument("--epochs",     type=int,   default=C.N_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=C.BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=C.LEARNING_RATE)
    p.add_argument("--patience",   type=int,   default=C.PATIENCE)
    p.add_argument("--base-ch",    type=int,   default=C.BASE_CHANNELS)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--resume",     type=str,   default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_loss_curve(train_losses, val_losses, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss", color="#1A5276", lw=2)
    ax.plot(epochs, val_losses,   label="Val loss",   color="#E74C3C", lw=2, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE loss")
    ax.set_title("CCS U-Net Training Loss Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Loss curve saved → {path}")


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = total_p = total_g = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for inputs, targets, mask in loader:
            inputs  = inputs.to(device)    # (B, 9, GNZ, GNR)
            targets = targets.to(device)   # (B, 2, 24, GNZ, GNR)
            mask    = mask.to(device)      # (B, GNZ)

            pred_p, pred_g = model(inputs)

            loss, lp, lg = criterion(pred_p, pred_g, targets, mask)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_p    += lp
            total_g    += lg
            n_batches  += 1

    return total_loss / n_batches, total_p / n_batches, total_g / n_batches


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CCS Surrogate U-Net — Training")
    print(f"{'='*60}")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  Batch : {args.batch_size}  |  LR : {args.lr}")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = CCSDataset(C.TRAIN_DIR, augment=not args.no_augment)
    val_ds   = CCSDataset(C.VAL_DIR,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    print(f"  Train samples : {len(train_ds)}  |  Val samples : {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = CCSUNet(base_ch=args.base_ch).to(device)
    print(f"  Parameters    : {model.parameter_count():,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=C.LR_STEP, gamma=C.LR_GAMMA
    )
    criterion = MaskedLoss()

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_val    = float('inf')

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"  Resumed from  : {args.resume} (epoch {start_epoch})")

    # ── Logging ───────────────────────────────────────────────────────────
    log_path = C.OUTPUT_DIR / "train_log.csv"
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'val_loss',
             'train_loss_pressure', 'train_loss_gas',
             'val_loss_pressure',   'val_loss_gas',
             'lr', 'epoch_time_s']
        )

    train_losses, val_losses = [], []
    patience_counter = 0

    print(f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'P_loss':>10}  {'G_loss':>10}  {'LR':>10}")
    print("-" * 65)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        tr_loss, tr_p, tr_g = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_p, vl_g = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        scheduler.step()
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"{epoch+1:>6}  {tr_loss:>10.5f}  {vl_loss:>10.5f}  "
              f"{vl_p:>10.5f}  {vl_g:>10.5f}  {current_lr:>10.2e}")

        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch+1, tr_loss, vl_loss, tr_p, tr_g, vl_p, vl_g,
                current_lr, round(elapsed, 1)
            ])

        # Checkpoint — best model
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val':        best_val,
                'config': {
                    'base_ch':     args.base_ch,
                    'n_timesteps': C.N_TIMESTEPS,
                    'in_ch':       C.N_INPUT_CHANNELS,
                }
            }, C.CKPT_DIR / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[INFO] Early stopping at epoch {epoch+1} "
                  f"(no improvement for {args.patience} epochs)")
            break

    print(f"\n[INFO] Training complete. Best val loss: {best_val:.5f}")
    print(f"[INFO] Best model saved → {C.CKPT_DIR / 'best_model.pt'}")

    # Save loss curve
    save_loss_curve(train_losses, val_losses, C.OUTPUT_DIR / "loss_curve.png")

    return model


if __name__ == "__main__":
    train()
