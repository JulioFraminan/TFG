"""
Simple VAE encoder/decoder for latent compression.
No external dependencies beyond PyTorch.
Architecture: Conv encoder → bottleneck (8 channels) → Conv decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVAEEncoder(nn.Module):
    """Encode images (1, 700, 2000) → latent (8, 87, 250)"""
    
    def __init__(self, in_channels=1, latent_channels=8):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        
        # Downsampling: 700×2000 → 350×1000 → 175×500 → 87×250 (3x downsampling ≈ 8x area)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # 700×2000 → 350×1000
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)           # 350×1000 → 175×500
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)         # 175×500 → 87×250
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(128, latent_channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: (B, 1, 700, 2000) normalized to [-1, 1]
        x = self.relu(self.conv1(x))      # (B, 32, 350, 1000)
        x = self.relu(self.conv2(x))      # (B, 64, 175, 500)
        x = self.relu(self.conv3(x))      # (B, 128, 87, 250)
        z = self.bottleneck(x)            # (B, 8, 87, 250)
        return z


class SimpleVAEDecoder(nn.Module):
    """Decode latent (8, 87, 250) → images (1, 700, 2000)"""
    
    def __init__(self, latent_channels=8, out_channels=1):
        super().__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        
        # Upsampling: 87×250 → 175×500 → 350×1000 → 700×2000
        self.deconv1 = nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1)  # 87×250 → 175×500
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)              # 175×500 → 350×1000
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)              # 350×1000 → 700×2000
        
        # Output projection
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, z):
        # z: (B, 8, 87, 250)
        x = self.relu(self.deconv1(z))    # (B, 128, 175, 500)
        x = self.relu(self.deconv2(x))    # (B, 64, 350, 1000)
        x = self.relu(self.deconv3(x))    # (B, 32, 700, 2000)
        
        # Output projection + tanh for [-1, 1] range
        x = self.out_conv(x)              # (B, 1, 700, 2000)
        x = torch.tanh(x)
        return x


class SimpleVAE(nn.Module):
    """Combined VAE with deterministic encode/decode (no sampling)."""
    
    def __init__(self, in_channels=1, latent_channels=8):
        super().__init__()
        self.encoder = SimpleVAEEncoder(in_channels, latent_channels)
        self.decoder = SimpleVAEDecoder(latent_channels, in_channels)
        self.latent_channels = latent_channels
    
    def encode(self, x):
        """Encode image to latent."""
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        """Decode latent to image."""
        x = self.decoder(z)
        return x
    
    def forward(self, x):
        """Full autoencoder forward pass."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
