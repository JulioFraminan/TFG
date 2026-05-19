import sys
import os
sys.path.insert(0, os.getcwd())
from unet_ae_modular_inpainting import config, data_utils
import numpy as np

folder = config.DATA_FOLDER
print(f"Checking data in {folder}")

# Manual list of planes or similar to what load_validation_rois does
plane_files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
for f in plane_files:
    path = os.path.join(folder, f)
    tl = np.load(path)
    print(f"File: {f}, Shape: {tl.shape}")
    # Physical coordinates logic from data_utils
    # X physical = X_RES * col + X_MIN
    # Z physical = Z_RES * row + Z_MAX
    # We want to know row, col for config.ROI_CORNER = (7, -2)
    # data_utils.py seems to have:
    # col = int((x_f - X_MIN) / X_RES)
    # row = int((z_f - Z_MAX) / Z_RES)
    
    # Let's peek at X_MIN, X_RES, Z_MAX, Z_RES in data_utils or config
