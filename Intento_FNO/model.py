"""
model.py - Conditional Fourier Neural Operator (FNO) 2D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compl_mul2d(input_tensor, weights):
    """Complex multiplication for spectral conv."""
    return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    def forward(self, x):
        # x: (batch, in_ch, height, width)
        batchsize, _, height, width = x.shape
        orig_dtype = x.dtype
        x = x.float()
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, height)
        m2 = min(self.modes2, width // 2 + 1)

        out_ft[:, :, :m1, :m2] = compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2],
        )
        out_ft[:, :, -m1:, :m2] = compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2],
        )

        x = torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return x.to(orig_dtype)


class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2, dropout=0.0):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.spectral(x) + self.w(x)
        x = F.gelu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ConditionalFNO2d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        cond_dim=1,
        modes1=16,
        modes2=16,
        width=64,
        depth=4,
        use_coords=True,
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.use_coords = use_coords

        extra_channels = cond_dim + (2 if use_coords else 0)
        self.fc0 = nn.Linear(in_channels + extra_channels, width)

        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes1, modes2, dropout=dropout) for _ in range(depth)]
        )

        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, out_channels)

    @staticmethod
    def _build_coord_grid(batch, height, width, device, dtype):
        grid_y = torch.linspace(0, 1, height, device=device, dtype=dtype)
        grid_x = torch.linspace(0, 1, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid = torch.stack([xx, yy], dim=0)
        grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)
        return grid

    def forward(self, x, cond):
        # x: (B, 1, H, W), cond: (B, cond_dim)
        batch, _, height, width = x.shape
        cond_map = cond.view(batch, self.cond_dim, 1, 1).expand(
            batch, self.cond_dim, height, width
        )

        inputs = [x, cond_map]
        if self.use_coords:
            coords = self._build_coord_grid(batch, height, width, x.device, x.dtype)
            inputs.append(coords)

        x_in = torch.cat(inputs, dim=1)
        x_in = x_in.permute(0, 2, 3, 1)

        x = self.fc0(x_in)
        x = x.permute(0, 3, 1, 2)

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x
