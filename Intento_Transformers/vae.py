"""
VAE wrapper using pretrained conditional UNet Autoencoder from unet_ae_modular.
Wraps encode/decode to handle conditioning automatically.
"""

import torch
import torch.nn as nn
import math
import sys
import os
import importlib.util


class UNetAEFromCheckpoint(nn.Module):
    """Load and wrap ConditionalUNetAE from unet_ae_modular checkpoint."""

    def __init__(self, checkpoint_path):
        super().__init__()
        # Import the UNet AE model class using importlib to avoid namespace conflicts
        unet_ae_model_path = os.path.join(
            os.path.dirname(__file__), "..", "unet_ae_modular", "model.py"
        )
        spec = importlib.util.spec_from_file_location("unet_ae_model", unet_ae_model_path)
        unet_ae_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unet_ae_model_module)
        ConditionalUNetAE = unet_ae_model_module.ConditionalUNetAE
        
        # Instantiate and load
        self.unet_ae = ConditionalUNetAE()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle both direct state dict and wrapped state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        self.unet_ae.load_state_dict(state_dict, strict=False)
        print(f"[VAE] Loaded ConditionalUNetAE checkpoint from {checkpoint_path}")

    def encode(self, x, cond):
        """
        Encode image to bottleneck representation.
        
        Args:
            x: (B, 1, H, W) image tensor
            cond: (B, 1) conditioning (angle) tensor, normalized to [0, 1]
        
        Returns:
            z: (B, 256, H/8, W/8) latent tensor
        """
        e1, e2, e3, b = self.unet_ae.encode(x, cond)
        # b is bottleneck with shape (B, 256, H/8, W/8)
        return b

    def decode(self, z, cond, output_shape):
        """
        Decode latent to image.
        
        Args:
            z: (B, 256, H/8, W/8) latent tensor
            cond: (B, 1) conditioning tensor
            output_shape: (H, W) target output shape
        
        Returns:
            x: (B, 1, H, W) reconstructed image
        """
        # Reconstruct skip connections (simplified: use encoder on dummy input)
        # For now, we'll create dummy skip connections of appropriate shape
        b, c, h, w = z.shape
        
        # Create dummy skip connections with correct spatial dims
        # e1: (B, 32, H, W), e2: (B, 64, H/2, W/2), e3: (B, 128, H/4, W/4)
        device = z.device
        e1_dummy = torch.zeros(b, 32, h * 8, w * 8, device=device)
        e2_dummy = torch.zeros(b, 64, h * 4, w * 4, device=device)
        e3_dummy = torch.zeros(b, 128, h * 2, w * 2, device=device)
        
        # Decode with dummy skip connections
        x = self.unet_ae.decode(e1_dummy, e2_dummy, e3_dummy, z, cond)
        
        # Resize to target output shape if needed
        if x.shape[2:] != output_shape:
            import torch.nn.functional as F
            x = F.interpolate(x, size=output_shape, mode="bicubic", align_corners=False)
        
        return x

    def forward(self, x, cond):
        """Full encode-decode cycle."""
        z = self.encode(x, cond)
        x_recon = self.decode(z, cond, (x.shape[2], x.shape[3]))
        return x_recon
    
    def latent_shape(self, height, width):
        """
        Get latent spatial shape for given input shape.
        UNet AE has 3 pooling operations, so latent is 1/8 of input.
        """
        h_lat = height // 8
        w_lat = width // 8
        return h_lat, w_lat


class VAECodec(nn.Module):
    """
    VAE Codec interface compatible with DiT training.
    """
    
    def __init__(self, vae_model, normalizer=None):
        """
        Args:
            vae_model: UNetAEFromCheckpoint instance
            normalizer: Normalizer object with normalize_angle() method
        """
        super().__init__()
        self.vae = vae_model
        self.normalizer = normalizer
        # VAE has 256 latent channels at 1/8 resolution
        self.latent_channels = 256

    def encode(self, x, cond=None):
        """
        Encode image to latent space.
        
        Args:
            x: (B, 1, H, W) image in [-1, 1]
            cond: (B, 1) angle conditioning (already normalized)
        
        Returns:
            z: (B, 256, H/8, W/8) latent
        """
        if cond is None:
            cond = torch.zeros(x.shape[0], 1, device=x.device)
        
        return self.vae.encode(x, cond)

    def decode(self, z, output_shape, cond=None):
        """
        Decode from latent to image space.
        
        Args:
            z: (B, 256, H/8, W/8) latent
            output_shape: (H, W) target output size
            cond: (B, 1) angle conditioning
        
        Returns:
            x: (B, 1, H, W) reconstructed image
        """
        if cond is None:
            cond = torch.zeros(z.shape[0], 1, device=z.device)
        
        return self.vae.decode(z, cond, output_shape)

    def latent_shape(self, height, width):
        """Get latent shape for input dimensions."""
        return self.vae.latent_shape(height, width)
