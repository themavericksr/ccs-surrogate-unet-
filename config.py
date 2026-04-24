"""
config.py
---------
All hyperparameters and paths in one place.
Change settings here — nothing else needs to be edited.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
TRAIN_DIR   = DATA_DIR / "train_data"
VAL_DIR     = DATA_DIR / "val_data"
CKPT_DIR    = ROOT / "checkpoints"
OUTPUT_DIR  = ROOT / "outputs"

for d in [CKPT_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Grid ──────────────────────────────────────────────────────────────────
# All samples are padded to this fixed size.
# 64 rows covers the observed max nz (~54). 200 cols is fixed in the dataset.
GRID_NZ     = 64        # padded height (rows = depth layers)
GRID_NR     = 200       # width  (radial cells) — fixed in dataset
N_TIMESTEPS = 24        # number of simulation time steps

# ── Model ─────────────────────────────────────────────────────────────────
N_INPUT_CHANNELS  = 9   # porosity, perm_r, perm_z, inj_rate, temp,
                        # depth, Swi, lam, perf_mask (binary)
N_OUTPUT_TIMESTEPS = N_TIMESTEPS  # predict full time series

BASE_CHANNELS = 64      # U-Net first-layer channel count

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE     = 2
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
N_EPOCHS       = 50
PATIENCE       = 10     # early stopping patience (epochs)
LR_STEP        = 10     # reduce LR every N epochs
LR_GAMMA       = 0.5    # LR reduction factor

# Loss weights: pressure + gas saturation (weighted sum)
LOSS_WEIGHT_PRESSURE = 0.5
LOSS_WEIGHT_GAS_SAT  = 0.5

# ── Normalisation statistics ───────────────────────────────────────────────
# Computed from the full training set. Update these after running
# utils.compute_normalisation_stats() on your complete dataset.
#
# Spatial fields: log-transform perm (heavy right skew), standardise pressure
# Scalars:        standardise all

NORM = {
    # Log-permeability (log of mD)
    "log_perm_r_mean": 4.30,   "log_perm_r_std": 0.85,
    "log_perm_z_mean": 0.15,   "log_perm_z_std": 0.85,
    # Porosity
    "porosity_mean":   0.20,   "porosity_std":   0.03,
    # Scalars
    "inj_rate_mean":   1.00,   "inj_rate_std":   0.50,
    "temperature_mean":90.0,   "temperature_std": 35.0,
    "depth_mean":      200.0,  "depth_std":       80.0,
    "Swi_mean":        0.24,   "Swi_std":         0.03,
    "lam_mean":        0.45,   "lam_std":         0.10,
    # Targets
    "pressure_mean":   5.0,    "pressure_std":    10.0,
    # gas_saturation is already in [0,1] — no normalisation applied
}
