"""
model.py
--------
Dual-output U-Net for simultaneous prediction of CO2 pressure buildup
and gas saturation across all 24 simulation timesteps.

Architecture:
    Shared encoder  — 4 down-sampling blocks (Conv → BN → ReLU × 2 → MaxPool)
    Bottleneck      — double conv without pooling
    Pressure decoder   — 4 up-sampling blocks with skip connections
    Gas sat decoder    — identical structure, independent weights

Each decoder predicts N_TIMESTEPS channels (one per simulation time step),
giving output shape (batch, N_TIMESTEPS, GRID_NZ, GRID_NR) per target.

Improvements over the original code3.py:
    1. Batch normalisation in every block — more stable training
    2. Dropout in bottleneck — regularisation for limited data
    3. Two independent decoders — pressure and gas saturation have
       different physical scales and spatial structure; joint decoding
       with separate weights outperforms a single shared decoder
    4. Predicts full time series (24 steps) not just the final state
    5. Weight initialisation (Kaiming) for better gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """
    Two consecutive  Conv2d → BatchNorm → ReLU  blocks.
    This is the standard U-Net building block.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """DoubleConv followed by 2×2 MaxPool."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        x_conv = self.conv(x)
        return x_conv, self.pool(x_conv)   # return both for skip connection


class Up(nn.Module):
    """
    Bilinear upsampling + concatenation with skip + DoubleConv.
    Bilinear upsampling avoids the checkerboard artefacts of transposed conv.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial dims (handles odd input sizes)
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Shared encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    4-level encoder shared between both decoder branches.
    Returns the bottleneck feature map and all skip connection tensors.
    """
    def __init__(self, in_ch: int, base_ch: int):
        super().__init__()
        c = base_ch
        self.down1 = Down(in_ch, c)
        self.down2 = Down(c,     c * 2)
        self.down3 = Down(c * 2, c * 4)
        self.down4 = Down(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16, dropout=0.3)

    def forward(self, x: torch.Tensor):
        s1, p1 = self.down1(x)
        s2, p2 = self.down2(p1)
        s3, p3 = self.down3(p2)
        s4, p4 = self.down4(p3)
        b  = self.bottleneck(p4)
        return b, (s1, s2, s3, s4)


# ---------------------------------------------------------------------------
# Decoder (used twice — once per output target)
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    4-level decoder with skip connections from the shared encoder.
    Outputs N_TIMESTEPS channels (one spatial map per simulation time step).
    """
    def __init__(self, base_ch: int, n_timesteps: int):
        super().__init__()
        c = base_ch
        self.up4 = Up(c * 16, c * 8,  c * 8)
        self.up3 = Up(c * 8,  c * 4,  c * 4)
        self.up2 = Up(c * 4,  c * 2,  c * 2)
        self.up1 = Up(c * 2,  c,      c)
        self.out = nn.Conv2d(c, n_timesteps, kernel_size=1)

    def forward(self, bottleneck: torch.Tensor, skips) -> torch.Tensor:
        s1, s2, s3, s4 = skips
        x = self.up4(bottleneck, s4)
        x = self.up3(x,          s3)
        x = self.up2(x,          s2)
        x = self.up1(x,          s1)
        return self.out(x)   # (B, n_timesteps, H, W)


# ---------------------------------------------------------------------------
# Full dual-output model
# ---------------------------------------------------------------------------

class CCSUNet(nn.Module):
    """
    Dual-output U-Net for CCS surrogate modelling.

    Inputs
    ------
    x : torch.Tensor  shape (B, N_INPUT_CHANNELS, GRID_NZ, GRID_NR)
        Stacked reservoir property channels.

    Outputs
    -------
    pressure : torch.Tensor  shape (B, N_TIMESTEPS, GRID_NZ, GRID_NR)
        Normalised pressure buildup predictions.
    gas_sat  : torch.Tensor  shape (B, N_TIMESTEPS, GRID_NZ, GRID_NR)
        Gas saturation predictions (raw logits — apply sigmoid for [0,1]).

    Parameters
    ----------
    in_ch : int
        Number of input channels (default = N_INPUT_CHANNELS from config).
    base_ch : int
        Channel count at the first encoder level (doubles at each level).
    n_timesteps : int
        Number of simulation time steps to predict (default = 24).
    """

    def __init__(
        self,
        in_ch:       int = C.N_INPUT_CHANNELS,
        base_ch:     int = C.BASE_CHANNELS,
        n_timesteps: int = C.N_TIMESTEPS,
    ):
        super().__init__()
        self.encoder          = Encoder(in_ch, base_ch)
        self.pressure_decoder = Decoder(base_ch, n_timesteps)
        self.gas_decoder      = Decoder(base_ch, n_timesteps)
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation for Conv2d layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        bottleneck, skips = self.encoder(x)
        pressure = self.pressure_decoder(bottleneck, skips)
        gas_sat  = self.gas_decoder(bottleneck, skips)
        # Apply sigmoid to gas saturation (physically bounded [0,1])
        gas_sat  = torch.sigmoid(gas_sat)
        return pressure, gas_sat

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Masked loss function
# ---------------------------------------------------------------------------

class MaskedLoss(nn.Module):
    """
    MSE loss that ignores padded rows.

    The spatial padding mask (shape: B × GRID_NZ) is broadcast across
    all timestep and column dimensions so only valid reservoir cells
    contribute to the gradient.

    Parameters
    ----------
    pressure_weight : float
        Relative weight for the pressure loss term.
    gas_weight : float
        Relative weight for the gas saturation loss term.
    """

    def __init__(
        self,
        pressure_weight: float = C.LOSS_WEIGHT_PRESSURE,
        gas_weight:      float = C.LOSS_WEIGHT_GAS_SAT,
    ):
        super().__init__()
        self.pw = pressure_weight
        self.gw = gas_weight

    def forward(
        self,
        pred_pressure: torch.Tensor,   # (B, T, NZ, NR)
        pred_gas:      torch.Tensor,   # (B, T, NZ, NR)
        target:        torch.Tensor,   # (B, 2, T, NZ, NR)
        mask:          torch.Tensor,   # (B, NZ)   bool
    ):
        tgt_p = target[:, 0]   # (B, T, NZ, NR) — pressure
        tgt_g = target[:, 1]   # (B, T, NZ, NR) — gas sat

        # Broadcast mask: (B, NZ) → (B, 1, NZ, 1) → matches (B, T, NZ, NR)
        m = mask.unsqueeze(1).unsqueeze(-1).float()   # (B, 1, NZ, 1)

        def masked_mse(pred, tgt):
            sq  = (pred - tgt) ** 2 * m
            return sq.sum() / (m.sum() * pred.shape[1] * pred.shape[3] + 1e-8)

        loss_p = masked_mse(pred_pressure, tgt_p)
        loss_g = masked_mse(pred_gas,      tgt_g)

        return self.pw * loss_p + self.gw * loss_g, loss_p.item(), loss_g.item()
