"""
dataset.py
----------
PyTorch Dataset for the CCS surrogate modelling challenge.

Key design decisions vs. the original code3.py:
    1. Variable nz is handled by padding to GRID_NZ and returning a binary
       mask so the loss function ignores padded rows.
    2. perm_r and perm_z are log-transformed before standardisation —
       their distributions span >1 order of magnitude.
    3. perf_interval (two integers: top/bottom row) is encoded as a binary
       channel on the full grid rather than broadcast as a scalar.
    4. Both targets (pressure_buildup AND gas_saturation) are returned
       across all 24 time steps — not just the final step.
    5. Pressure is standardised; gas saturation is kept in [0,1].
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C


# ---------------------------------------------------------------------------
# Helper: build normalisation statistics from a data directory
# ---------------------------------------------------------------------------

def compute_normalisation_stats(data_dir: str, max_files: int = 200) -> dict:
    """
    Compute mean/std for all input channels from a set of training files.
    Run once on your full training set and paste results into config.py.

    Parameters
    ----------
    data_dir : str
        Path to directory containing .npz files.
    max_files : int
        Maximum number of files to use (for speed).

    Returns
    -------
    dict of normalisation statistics.
    """
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    )[:max_files]

    poro_all, logr_all, logz_all = [], [], []
    inj_all, temp_all, depth_all, swi_all, lam_all = [], [], [], [], []
    press_all = []

    for fname in files:
        with np.load(os.path.join(data_dir, fname)) as d:
            poro_all.append(d['porosity'].mean())
            logr_all.append(np.log(d['perm_r']).mean())
            logz_all.append(np.log(d['perm_z']).mean())
            inj_all.append(float(d['inj_rate']))
            temp_all.append(float(d['temperature']))
            depth_all.append(float(d['depth']))
            swi_all.append(float(d['Swi']))
            lam_all.append(float(d['lam']))
            press_all.append(d['pressure_buildup'].mean())

    def ms(arr): return float(np.mean(arr)), float(np.std(arr) + 1e-8)

    return {
        "porosity_mean":     ms(poro_all)[0],  "porosity_std":     ms(poro_all)[1],
        "log_perm_r_mean":   ms(logr_all)[0],  "log_perm_r_std":   ms(logr_all)[1],
        "log_perm_z_mean":   ms(logz_all)[0],  "log_perm_z_std":   ms(logz_all)[1],
        "inj_rate_mean":     ms(inj_all)[0],   "inj_rate_std":     ms(inj_all)[1],
        "temperature_mean":  ms(temp_all)[0],  "temperature_std":  ms(temp_all)[1],
        "depth_mean":        ms(depth_all)[0], "depth_std":        ms(depth_all)[1],
        "Swi_mean":          ms(swi_all)[0],   "Swi_std":          ms(swi_all)[1],
        "lam_mean":          ms(lam_all)[0],   "lam_std":          ms(lam_all)[1],
        "pressure_mean":     ms(press_all)[0], "pressure_std":     ms(press_all)[1],
    }


def _standardise(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


def _build_perf_mask(nz: int, nr: int, perf_interval: np.ndarray) -> np.ndarray:
    """
    Build a binary perforation mask of shape (nz, nr).
    Rows in [perf_top, perf_bottom) are set to 1.
    """
    mask = np.zeros((nz, nr), dtype=np.float32)
    top  = int(np.clip(perf_interval[0], 0, nz - 1))
    bot  = int(np.clip(perf_interval[1], 0, nz))
    mask[top:bot, :] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CCSDataset(Dataset):
    """
    GEMS4-CCS surrogate dataset.

    Each sample returns:
        inputs  : torch.Tensor  shape (N_INPUT_CHANNELS, GRID_NZ, GRID_NR)
        targets : torch.Tensor  shape (2, N_TIMESTEPS, GRID_NZ, GRID_NR)
                  channel 0 = normalised pressure buildup
                  channel 1 = gas saturation (already in [0,1])
        mask    : torch.Tensor  shape (GRID_NZ,)  bool
                  True for valid rows, False for padded rows

    Parameters
    ----------
    data_dir : str
        Directory containing .npz files.
    norm : dict
        Normalisation statistics (from config.NORM or compute_normalisation_stats).
    grid_nz : int
        Target padded height.
    grid_nr : int
        Target width (should match dataset — always 200).
    augment : bool
        If True, apply random left-right flip augmentation.
    """

    def __init__(
        self,
        data_dir: str,
        norm: dict = None,
        grid_nz: int = C.GRID_NZ,
        grid_nr: int = C.GRID_NR,
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.norm     = norm if norm is not None else C.NORM
        self.grid_nz  = grid_nz
        self.grid_nr  = grid_nr
        self.augment  = augment

        self.files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fpath = os.path.join(self.data_dir, self.files[idx])

        with np.load(fpath) as d:
            porosity    = d['porosity'].astype(np.float32)   # (nz, nr)
            perm_r      = d['perm_r'].astype(np.float32)
            perm_z      = d['perm_z'].astype(np.float32)
            inj_rate    = float(d['inj_rate'])
            temperature = float(d['temperature'])
            depth       = float(d['depth'])
            swi         = float(d['Swi'])
            lam         = float(d['lam'])
            perf        = d['perf_interval']                  # (2,)
            pressure    = d['pressure_buildup'].astype(np.float32)  # (nz, nr, 24)
            gas_sat     = d['gas_saturation'].astype(np.float32)    # (nz, nr, 24)

        nz, nr = porosity.shape

        # ── 1. Normalise spatial inputs ───────────────────────────────────
        n = self.norm
        poro_n   = _standardise(porosity, n['porosity_mean'],    n['porosity_std'])
        log_perm_r = np.log(np.clip(perm_r, 1e-3, None))
        log_perm_z = np.log(np.clip(perm_z, 1e-6, None))
        permr_n  = _standardise(log_perm_r, n['log_perm_r_mean'], n['log_perm_r_std'])
        permz_n  = _standardise(log_perm_z, n['log_perm_z_mean'], n['log_perm_z_std'])

        # ── 2. Scalar channels — broadcast to (nz, nr) ───────────────────
        def scalar_channel(val, mean, std):
            return np.full((nz, nr), (val - mean) / (std + 1e-8), dtype=np.float32)

        inj_ch   = scalar_channel(inj_rate,    n['inj_rate_mean'],    n['inj_rate_std'])
        temp_ch  = scalar_channel(temperature, n['temperature_mean'], n['temperature_std'])
        dep_ch   = scalar_channel(depth,       n['depth_mean'],       n['depth_std'])
        swi_ch   = scalar_channel(swi,         n['Swi_mean'],         n['Swi_std'])
        lam_ch   = scalar_channel(lam,         n['lam_mean'],         n['lam_std'])

        # ── 3. Perforation mask channel ───────────────────────────────────
        perf_ch  = _build_perf_mask(nz, nr, perf)

        # ── 4. Stack input channels: (9, nz, nr) ─────────────────────────
        inputs_raw = np.stack(
            [poro_n, permr_n, permz_n, inj_ch, temp_ch, dep_ch, swi_ch, lam_ch, perf_ch],
            axis=0
        )  # (9, nz, nr)

        # ── 5. Normalise targets ──────────────────────────────────────────
        # Pressure: standardise
        press_n = _standardise(pressure, n['pressure_mean'], n['pressure_std'])
        # Gas saturation: already in [0,1] — no normalisation
        # Rearrange from (nz, nr, 24) → (24, nz, nr)
        press_n  = press_n.transpose(2, 0, 1)   # (24, nz, nr)
        gas_sat  = gas_sat.transpose(2, 0, 1)   # (24, nz, nr)

        # ── 6. Pad to fixed grid (GRID_NZ, GRID_NR) ──────────────────────
        def pad_spatial(arr, target_nz, target_nr):
            """Pad array to (*, target_nz, target_nr) with zeros."""
            shape = arr.shape[:-2] + (target_nz, target_nr)
            out = np.zeros(shape, dtype=np.float32)
            out[..., :nz, :nr] = arr
            return out

        inputs  = pad_spatial(inputs_raw, self.grid_nz, self.grid_nr)  # (9, GNZ, GNR)
        press_p = pad_spatial(press_n,    self.grid_nz, self.grid_nr)  # (24, GNZ, GNR)
        gas_p   = pad_spatial(gas_sat,    self.grid_nz, self.grid_nr)  # (24, GNZ, GNR)

        # ── 7. Stack dual targets: (2, 24, GNZ, GNR) ─────────────────────
        targets = np.stack([press_p, gas_p], axis=0)

        # ── 8. Mask: True = valid row, False = padded ─────────────────────
        mask = np.zeros(self.grid_nz, dtype=bool)
        mask[:nz] = True

        # ── 9. Optional augmentation ──────────────────────────────────────
        if self.augment and np.random.rand() > 0.5:
            # Flip radial (left-right) direction
            inputs  = inputs[..., ::-1].copy()
            targets = targets[..., ::-1].copy()

        return (
            torch.from_numpy(inputs),           # (9, GNZ, GNR)
            torch.from_numpy(targets),          # (2, 24, GNZ, GNR)
            torch.from_numpy(mask),             # (GNZ,) bool
        )
