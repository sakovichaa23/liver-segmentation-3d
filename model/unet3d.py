# Архитектура 3D U-Net
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    # Остаточный блок (Conv3d → InstanceNorm3d → LeakyReLU → Dropout3d)
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout3d(dropout)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        return self.act(y + residual)


class UNet3D(nn.Module):
    # 3D U-Net (3 уровня энкодера + bottleneck + 3 уровня декодера)
    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.enc1 = ResidualBlock3D(1, base)
        self.enc2 = ResidualBlock3D(base, base * 2)
        self.enc3 = ResidualBlock3D(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ResidualBlock3D(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, 2)
        self.dec3 = ResidualBlock3D(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, 2)
        self.dec2 = ResidualBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, 2)
        self.dec1 = ResidualBlock3D(base * 2, base)
        self.out = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
