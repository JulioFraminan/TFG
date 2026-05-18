
"""
model.py — Conditional UNet AE plug & play
Mejorado con:
- FiLM condicional (encoder alto + bottleneck + decoder)
- Attention gates
- Dropout en bottleneck
- GroupNorm
- Deep supervision (opcional internamente)
Incluye modo inpainting opcional via mascara
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# BLOQUES
# -------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
        )
        self.act = nn.SiLU(inplace=True)
        # Residual connection
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        return self.act(out + residual)


class FiLM(nn.Module):
    def __init__(self, n_channels, cond_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_channels * 2),
        )

    def forward(self, x, cond):
        params = self.mlp(cond)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi + x


# -------------------------
# MODELO
# -------------------------

class ConditionalUNetAE(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.film_enc3 = FiLM(128)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)
        self.film_bn = FiLM(256)
        self.dropout = nn.Dropout2d(dropout_prob)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.film3 = FiLM(128)
        self.film3_g = FiLM(128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.film2 = FiLM(64)
        self.film2_g = FiLM(64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.film1 = FiLM(32)
        self.film1_g = FiLM(32)

        # Attention
        self.att3 = AttentionBlock(128, 128, 64)
        self.att2 = AttentionBlock(64, 64, 32)
        self.att1 = AttentionBlock(32, 32, 16)

        # Deep supervision (interno)
        self.out3 = nn.Conv2d(128, 1, 1)
        self.out2 = nn.Conv2d(64, 1, 1)

        # Output final
        self.out_conv = nn.Conv2d(32, 1, 1)

    # -------------------------
    # ENCODER
    # -------------------------
    def encode(self, x, cond):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e3 = self.film_enc3(e3, cond)
        b = self.bottleneck(self.pool(e3))
        return e1, e2, e3, b

    # -------------------------
    # DECODER
    # -------------------------
    def decode(self, e1, e2, e3, b, cond):
        b = self.film_bn(b, cond)
        b = self.dropout(b)

        # Nivel 3
        d3 = self.up3(b)
        g3 = self.film3_g(d3, cond)
        e3_att = self.att3(g3, e3)
        d3 = self._match_and_cat(d3, e3_att)
        d3 = self.film3(self.dec3(d3), cond)

        # Nivel 2
        d2 = self.up2(d3)
        g2 = self.film2_g(d2, cond)
        e2_att = self.att2(g2, e2)
        d2 = self._match_and_cat(d2, e2_att)
        d2 = self.film2(self.dec2(d2), cond)

        # Nivel 1
        d1 = self.up1(d2)
        g1 = self.film1_g(d1, cond)
        e1_att = self.att1(g1, e1)
        d1 = self._match_and_cat(d1, e1_att)
        d1 = self.film1(self.dec1(d1), cond)

        out = self.out_conv(d1)
        return out  # SOLO out para entrenamiento plug & play

    # -------------------------
    # UTIL
    # -------------------------
    @staticmethod
    def _match_and_cat(up_feat, skip_feat):
        if up_feat.shape[2:] != skip_feat.shape[2:]:
            up_feat = F.interpolate(up_feat, size=skip_feat.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([up_feat, skip_feat], dim=1)

    @staticmethod
    def _prepare_mask(mask, x):
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[2:] != x.shape[2:]:
            mask = F.interpolate(mask.float(), size=x.shape[2:], mode="nearest")
        return mask.float().clamp(0.0, 1.0)

    def inpaint(self, x, cond, known_mask):
        pred = self.forward(x, cond)
        known_mask = self._prepare_mask(known_mask, x)
        return x * known_mask + pred * (1.0 - known_mask)

    # -------------------------
    # FORWARD
    # -------------------------
    def forward(self, x, cond, known_mask=None):
        e1, e2, e3, b = self.encode(x, cond)
        out = self.decode(e1, e2, e3, b, cond)
        if known_mask is None:
            return out
        known_mask = self._prepare_mask(known_mask, x)
        return x * known_mask + out * (1.0 - known_mask)


class PINN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, num_layers=6, out_dim=1,
                 activation="tanh", dropout=0.0):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.Tanh)

        layers = [nn.Linear(in_dim, hidden_dim), act_cls()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords):
        return self.net(coords)

