# ccs-surrogate-unet

**Deep learning surrogate model for CO₂ geological storage (CCS) simulation.**

A dual-output U-Net trained to replace expensive numerical reservoir simulation for CCS feasibility screening — predicting full spatiotemporal fields of CO₂ pressure buildup and gas saturation across 24 simulation timesteps from reservoir property inputs alone.

Built as part of a machine learning challenge at Imperial College London.

---

## Problem

Numerical simulation of CO₂ injection into geological storage formations is computationally expensive — each run can take hours. For uncertainty quantification and site screening, thousands of simulations are needed across different geological realisations and injection parameters.

This surrogate model learns the input-output mapping of the simulator directly from training data, enabling near-instant predictions once trained.

**Input:** Reservoir property fields (porosity, permeability, perforation location) + injection parameters (rate, temperature, depth, initial water saturation)

**Output:** Full 2D spatiotemporal fields of pressure buildup (bar) and CO₂ gas saturation across 24 simulation timesteps spanning 1 day to 30 years

---

## Architecture

```
9-channel input
    │
    ▼
┌─────────────────────────────┐
│       Shared Encoder         │
│  Down1 → Down2 → Down3 → Down4 → Bottleneck  │
│  (Conv → BN → ReLU) × 2 + MaxPool at each    │
└──────────────┬──────────────┘
               │ skip connections
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────┐        ┌──────────┐
│ Pressure │        │  Gas Sat │
│ Decoder  │        │ Decoder  │
│ (Up × 4) │        │ (Up × 4) │
└────┬─────┘        └────┬─────┘
     ▼                   ▼
(B, 24, NZ, NR)    (B, 24, NZ, NR)
pressure buildup   gas saturation
```

**Key design decisions:**

**Two independent decoders** — pressure and gas saturation have different physical scales and spatial structure (pressure spreads diffusively; CO₂ plume is compact). Separate decoder weights outperform a shared decoder.

**Full time series prediction** — predicts all 24 timesteps simultaneously as separate output channels, rather than training 24 separate models or recurrently stepping.

**Masked loss** — reservoir grids have variable depth (nz ranges 20–54 rows). Samples are padded to a fixed grid; the loss function ignores padded rows so training is unaffected by the zero-padding.

**Log-normalised permeability** — perm_r spans 28–590 mD (>1 order of magnitude). Log transformation before standardisation is critical for model convergence on high-contrast samples.

**Perforation mask channel** — perf_interval (top/bottom row integers) is encoded as a binary spatial mask rather than two scalar values, giving the model direct spatial context of where injection occurs.

---

## Results

| Metric | Pressure Buildup | Gas Saturation |
|---|---|---|
| R² (validation) | reported after training | reported after training |
| MAE (validation) | — bar | — |

*Populate after training on the full GEMS4-CCS dataset.*

---

## Quick start

```bash
git clone https://github.com/nhoyidi-nsan/ccs-surrogate-unet
cd ccs-surrogate-unet
pip install -r requirements.txt

# Place .npz files in data/train_data/ and data/val_data/
# Then:

# 1. (Optional) Recompute normalisation stats from your training set
python -c "
from dataset import compute_normalisation_stats
stats = compute_normalisation_stats('data/train_data')
print(stats)
"
# Paste results into config.py NORM dict

# 2. Train
python train.py --epochs 100 --batch-size 4

# 3. Evaluate
python evaluate.py

# 4. Single-sample prediction
python predict.py --file data/val_data/data_0014.npz --timestep 23
```

---

## Input channels

| Channel | Description | Preprocessing |
|---|---|---|
| porosity | Effective porosity (fraction) | Standardised |
| perm_r | Radial permeability (mD) | Log → standardised |
| perm_z | Vertical permeability (mD) | Log → standardised |
| inj_rate | CO₂ injection rate (Mt/yr) | Standardised, broadcast to grid |
| temperature | Formation temperature (°C) | Standardised, broadcast |
| depth | Formation depth (m) | Standardised, broadcast |
| Swi | Initial water saturation | Standardised, broadcast |
| lam | Brooks-Corey exponent | Standardised, broadcast |
| perf_mask | Perforation interval | Binary spatial mask [0,1] |

---

## Grid convention

The dataset uses an irregular radial grid (coarsening with radius). All samples have 200 radial cells (nr = 200) but variable vertical layers (nz = 20–54). Samples are zero-padded to 64 × 200 for batching; a boolean mask tracks valid rows.

The 24 simulation timesteps span:
`1d, 2d, 4d, 7d, 11d, 17d, 25d, 37d, 53d, 77d, 111d, 158d, 226d, 323d, 1.3y, 1.8y, 2.6y, 3.6y, 5.2y, 7.3y, 10.4y, 14.8y, 21.1y, 30y`

---

## Project structure

```
ccs-surrogate-unet/
├── config.py       — all hyperparameters and paths
├── dataset.py      — PyTorch Dataset with normalisation, padding, masking
├── model.py        — Dual-output U-Net + MaskedLoss
├── train.py        — training loop: early stopping, LR schedule, checkpointing
├── evaluate.py     — R², MAE, timestep plots, spatial comparison figures
├── predict.py      — single-sample inference and visualisation
├── data/
│   ├── train_data/ — .npz training samples
│   └── val_data/   — .npz validation samples
├── checkpoints/    — best_model.pt saved here during training
└── outputs/        — loss curves, evaluation plots, prediction figures
```

---

## Requirements

```
torch>=2.0
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
```

---

## References

- Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
- GEMS4-CCS dataset — Imperial College London MSc Geo-energy with ML and Data Science module.
