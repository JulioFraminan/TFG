import os
import re

import h5py
import numpy as np
import torch


def _safe_get_dataset(f, keys):
    """Return the first dataset present in keys inside h5py.File f."""
    for k in keys:
        if k in f:
            return f[k][:]
    avail = list(f.keys())
    raise KeyError(f"None of the keys {keys} found. Available: {avail}")


def _sanitize_tl_array(tl):
    """Convert TL to float32 and replace NaN/Inf with a stable finite value."""
    tl = np.asarray(tl, dtype=np.float32)
    finite_mask = np.isfinite(tl)
    if np.all(finite_mask):
        return tl

    if np.any(finite_mask):
        fill_value = float(np.nanmin(tl[finite_mask]))
    else:
        fill_value = 0.0

    return np.nan_to_num(tl, nan=fill_value, posinf=fill_value, neginf=fill_value)


# ------------------------------
# ROI EXTRACTION
# ------------------------------

def extract_roi_at(tl, ci, cj, roi_h, roi_w):
    """Extract a roi_h x roi_w patch centered at (ci, cj)."""
    half_h = roi_h // 2
    half_w = roi_w // 2
    rows, cols = tl.shape

    i0, i1 = max(ci - half_h, 0), min(ci + half_h, rows)
    j0, j1 = max(cj - half_w, 0), min(cj + half_w, cols)

    if i1 - i0 < roi_h:
        if i0 == 0:
            i1 = min(roi_h, rows)
        else:
            i0 = max(i1 - roi_h, 0)
    if j1 - j0 < roi_w:
        if j0 == 0:
            j1 = min(roi_w, cols)
        else:
            j0 = max(j1 - roi_w, 0)

    return tl[i0:i1, j0:j1], i0, j0


def _get_z_axis(Z_raw):
    """Detecta automaticamente la estructura de Z_raw y devuelve el eje Z completo."""
    z_row = Z_raw[0, :].flatten()
    z_col = Z_raw[:, 0].flatten()
    if z_row.size > 1 and z_col.size <= 1:
        return z_row
    if z_col.size > 1 and z_row.size <= 1:
        return z_col
    return z_row if z_row.size >= z_col.size else z_col


def phys_to_pixel(X_raw, Z_raw, x_phys, z_phys):
    """Convert physical coordinates (X, Z) to pixel indices in tl.T.
    
    Soporta ambas estructuras de Z_raw:
      • Antigua: shape (1, 60)  → Z[0, :] tiene datos
      • Nueva:  shape (60, 1)   → Z[:, 0] tiene datos
    """
    x_axis = X_raw[:, 0]
    col_idx = int(np.argmin(np.abs(x_axis - x_phys)))

    # Detectar automáticamente la estructura de Z
    z_axis = _get_z_axis(Z_raw)
    row_idx = int(np.argmin(np.abs(z_axis - z_phys)))

    return row_idx, col_idx


def phys_size_to_pixels(X_raw, Z_raw, roi_h, roi_w):
    """Convert physical roi size (meters) to pixels using X_raw/Z_raw."""
    if isinstance(roi_h, (int, np.integer)) and isinstance(roi_w, (int, np.integer)):
        return int(roi_h), int(roi_w)

    try:
        x_axis = X_raw[:, 0]
        z_axis = _get_z_axis(Z_raw)
    except Exception:
        dx = 1.0
        dz = 1.0
    else:
        dxs = np.diff(x_axis)
        dzs = np.diff(z_axis)
        dx = float(np.median(dxs)) if dxs.size > 0 else 1.0
        dz = float(np.median(dzs)) if dzs.size > 0 else 1.0

    roi_h_px = int(max(1, round(float(roi_h) / dz))) if not isinstance(roi_h, (int, np.integer)) else int(roi_h)
    roi_w_px = int(max(1, round(float(roi_w) / dx))) if not isinstance(roi_w, (int, np.integer)) else int(roi_w)

    return roi_h_px, roi_w_px


def extract_roi_at_corner(tl, corner_i, corner_j, roi_h, roi_w):
    """Extract roi_h x roi_w using (corner_i, corner_j) as top-left corner."""
    rows, cols = tl.shape

    i_top = int(corner_i)
    j0 = int(corner_j)
    i0 = i_top - roi_h + 1

    if i_top < 0 or i_top >= rows:
        raise ValueError(
            f"Top row ({i_top}) outside plane (0..{rows-1})."
        )
    if j0 < 0 or j0 >= cols:
        raise ValueError(
            f"Left column ({j0}) outside plane (0..{cols-1})."
        )
    if i0 < 0:
        raise ValueError(
            f"ROI exceeds plane: top row={i_top}, ROI_HEIGHT={roi_h} -> start={i0} (< 0)."
        )
    if j0 + roi_w > cols:
        raise ValueError(
            f"ROI exceeds plane: left col={j0}, ROI_WIDTH={roi_w}, cols={cols}."
        )

    roi = tl[i0:i_top + 1, j0:j0 + roi_w]
    return roi, i0, j0


def compute_roi_extent(X_raw, Z_raw, i0, j0, roi_h, roi_w):
    """Compute physical extent [x_left, x_right, z_bottom, z_top]."""
    x_axis = X_raw[:, 0]
    z_axis = _get_z_axis(Z_raw)

    x_left = float(x_axis[j0])
    x_right = float(x_axis[min(j0 + roi_w - 1, len(x_axis) - 1)])
    z_bottom = float(z_axis[i0])
    z_top = float(z_axis[min(i0 + roi_h - 1, len(z_axis) - 1)])

    return [x_left, x_right, z_bottom, z_top]


def extract_multiple_rois(tl, roi_h, roi_w, num_rois=5,
                          mode="center_max", corner_pixel=(0, 0)):
    """Extract ROIs from a plane according to the mode."""
    rois = []
    origins = []
    rows, cols = tl.shape

    if mode == "corner_fixed":
        roi, i0, j0 = extract_roi_at_corner(tl, corner_pixel[0],
                                            corner_pixel[1], roi_h, roi_w)
        rois.append(roi)
        origins.append((i0, j0))
        return rois, origins

    half_h = roi_h // 2
    half_w = roi_w // 2
    mi, mj = np.unravel_index(np.argmax(tl), tl.shape)
    roi, i0, j0 = extract_roi_at(tl, mi, mj, roi_h, roi_w)
    rois.append(roi)
    origins.append((i0, j0))

    threshold = np.percentile(tl, 80)
    high_coords = np.argwhere(tl >= threshold)
    if len(high_coords) > 0:
        for _ in range(num_rois - 1):
            ci, cj = high_coords[np.random.randint(len(high_coords))]
            ci = np.clip(ci, half_h, rows - half_h)
            cj = np.clip(cj, half_w, cols - half_w)
            roi, i0, j0 = extract_roi_at(tl, ci, cj, roi_h, roi_w)
            rois.append(roi)
            origins.append((i0, j0))
    else:
        for _ in range(num_rois - 1):
            ci = np.random.randint(half_h, rows - half_h)
            cj = np.random.randint(half_w, cols - half_w)
            roi, i0, j0 = extract_roi_at(tl, ci, cj, roi_h, roi_w)
            rois.append(roi)
            origins.append((i0, j0))
    return rois, origins


# ------------------------------
# DATA LOADING
# ------------------------------

from config import USE_BLOCK_VARIABLES

def _extract_angle(fname):
    """Extract angle from filename (PlaneAngle-104.47 -> -104.47)."""
    m = re.search(r"PlaneAngle(-?\d+(?:\.\d+)?)", fname)
    return float(m.group(1)) if m else 0.0


def _load_rois_from_folder(folder, roi_h, roi_w, rois_per_plane,
                           roi_mode, roi_corner, verbose_bounds=True):
    """Common loader: read .mat and extract ROIs."""
    all_rois, all_angles, all_extents = [], [], []

    if roi_mode == "corner_fixed":
        print(f"  ROI mode: corner_fixed  |  physical corner (X, Z) = {roi_corner}")

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".mat") or "PlaneAngle" not in fname:
            continue
        fpath = os.path.join(folder, fname)
        try:
            with h5py.File(fpath, "r") as f:
                tl_keys = ["tl_block", "tl", "TL", "tL"] if USE_BLOCK_VARIABLES else ["tl", "TL", "tL", "tl_block"]
                tl = _sanitize_tl_array(_safe_get_dataset(f, tl_keys).T)

                # Prefer full-resolution coordinate grids when they match TL
                X_raw = None
                Z_raw = None
                if "R" in f and "Z" in f:
                    r_full = f["R"][:]
                    z_full = f["Z"][:]
                    if r_full.shape == tl.shape and z_full.shape == tl.shape:
                        X_raw = r_full
                        Z_raw = z_full

                if X_raw is None or Z_raw is None:
                    X_raw = _safe_get_dataset(f, ["R_block", "X", "x", "R", "r"])
                    Z_raw = _safe_get_dataset(f, ["Z_block", "Z", "z"])
        except Exception as e:
            print(f"  [!] Error with {fname}: {e}")
            continue

        angle = _extract_angle(fname)

        roi_h_px, roi_w_px = phys_size_to_pixels(X_raw, Z_raw, roi_h, roi_w)

        rows, cols = tl.shape
        roi_h_px = min(roi_h_px, rows)
        roi_w_px = min(roi_w_px, cols)

        corner_pixel = (0, 0)
        if roi_mode == "corner_fixed":
            x_phys, z_phys = roi_corner
            row_px, col_px = phys_to_pixel(X_raw, Z_raw, x_phys, z_phys)

            # Clamp corner so the ROI fits inside the plane
            max_top = max(roi_h_px - 1, 0)
            max_left = max(cols - roi_w_px, 0)
            adj_row_px = min(max(row_px, max_top), rows - 1)
            adj_col_px = min(max(col_px, 0), max_left)
            if verbose_bounds and (adj_row_px != row_px or adj_col_px != col_px):
                print(
                    f"  [i] {fname}: corner adjusted so ROI fits -> pixel ({adj_row_px}, {adj_col_px})"
                )
            row_px, col_px = adj_row_px, adj_col_px

            if verbose_bounds:
                x_min, x_max = float(X_raw.min()), float(X_raw.max())
                z_min, z_max = float(Z_raw.min()), float(Z_raw.max())
                clamped = False
                if x_phys < x_min or x_phys > x_max:
                    print(f"  [!] {fname}: X={x_phys} out of range [{x_min:.1f}, {x_max:.1f}] -> clamp")
                    clamped = True
                if z_phys < z_min or z_phys > z_max:
                    print(f"  [!] {fname}: Z={z_phys} out of range [{z_min:.1f}, {z_max:.1f}] -> clamp")
                    clamped = True
                if not clamped:
                    print(f"  [i] {fname}: physical corner ({x_phys}, {z_phys}) -> pixel ({row_px}, {col_px})")

            corner_pixel = (row_px, col_px)

        plane_rois, plane_origins = extract_multiple_rois(
            tl, roi_h_px, roi_w_px, rois_per_plane,
            mode=roi_mode, corner_pixel=corner_pixel,
        )
        for roi, (oi, oj) in zip(plane_rois, plane_origins):
            extent = compute_roi_extent(X_raw, Z_raw, oi, oj, roi_h_px, roi_w_px)
            all_rois.append(roi.astype(np.float32))
            all_angles.append(angle)
            all_extents.append(extent)
        print(f"  [OK] {fname} -> {len(plane_rois)} ROIs  (angle={angle:.2f})")

    if not all_rois:
        return np.array([], dtype=np.float32), [], []
    return np.array(all_rois, dtype=np.float32), all_angles, all_extents


def load_all_rois(folder, roi_h, roi_w, rois_per_plane=5,
                  roi_mode="center_max", roi_corner=(0, 0)):
    """Load all training .mat files and extract ROIs."""
    return _load_rois_from_folder(
        folder, roi_h, roi_w, rois_per_plane,
        roi_mode, roi_corner, verbose_bounds=True,
    )


def load_validation_rois(folder, roi_h, roi_w,
                         roi_mode="center_max", roi_corner=(0, 0)):
    """Load validation .mat files (1 ROI per plane)."""
    if not os.path.isdir(folder):
        return None, [], []

    print("\n" + "=" * 60)
    print("  LOADING VALIDATION PLANES")
    print("=" * 60)

    rois, angles, extents = _load_rois_from_folder(
        folder, roi_h, roi_w, 1,
        roi_mode, roi_corner, verbose_bounds=False,
    )
    if len(rois) == 0:
        return None, [], []
    return rois, angles, extents


# ------------------------------
# NORMALIZATION
# ------------------------------

class Normalizer:
    """Store TL and angle ranges and provide normalization helpers."""

    def __init__(self, rois, roi_angles):
        if rois is None or getattr(rois, "size", 0) == 0:
            raise ValueError(
                "No ROIs found. Check input folder, ROI settings, and file names."
            )

        self.tl_min = float(rois.min())
        self.tl_max = float(rois.max())

        angles_array = np.array(roi_angles, dtype=np.float32)
        if angles_array.size == 0:
            raise ValueError("No angles found for ROIs.")
        self.angle_min = float(angles_array.min())
        self.angle_max = float(angles_array.max())

    def normalize_tl(self, x):
        return (x - self.tl_min) / (self.tl_max - self.tl_min + 1e-8)

    def denormalize_tl(self, x_norm):
        return x_norm * (self.tl_max - self.tl_min) + self.tl_min

    def normalize_angle(self, angle_deg):
        return (angle_deg - self.angle_min) / (self.angle_max - self.angle_min + 1e-8)

    def normalize_angles(self, angles_array):
        a = np.array(angles_array, dtype=np.float32)
        return (a - self.angle_min) / (self.angle_max - self.angle_min + 1e-8)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        obj = cls.__new__(cls)
        obj.tl_min = checkpoint["tl_min"]
        obj.tl_max = checkpoint["tl_max"]
        obj.angle_min = checkpoint["angle_min"]
        obj.angle_max = checkpoint["angle_max"]
        return obj

    def state_dict(self):
        return {
            "tl_min": self.tl_min,
            "tl_max": self.tl_max,
            "angle_min": self.angle_min,
            "angle_max": self.angle_max,
        }


# ------------------------------
# DATA AUGMENTATION
# ------------------------------

def augment(data, angles):
    """Augment images with gaussian noise and contrast scaling."""
    aug_imgs = list(data)
    aug_angs = list(angles)

    for img, ang in zip(data, angles):
        for sigma in [0.01, 0.03]:
            noisy = img + np.random.normal(0, sigma, img.shape).astype(np.float32)
            aug_imgs.append(np.clip(noisy, 0, 1))
            aug_angs.append(ang)
        s = np.random.uniform(0.9, 1.1)
        aug_imgs.append(
            np.clip(img * s + np.random.uniform(-0.05, 0.05), 0, 1).astype(np.float32)
        )
        aug_angs.append(ang)
    return np.array(aug_imgs, dtype=np.float32), np.array(aug_angs, dtype=np.float32)


# ------------------------------
# MAT I/O
# ------------------------------

def create_xyz_grid(shape, x_range=None, z_range=None):
    """Create X/R, Y, Z grids with the same convention as input .mat."""
    rows, cols = shape

    if x_range is None:
        x_axis = np.arange(rows, dtype=np.float64)
    else:
        x_axis = np.linspace(float(x_range[0]), float(x_range[1]), rows, dtype=np.float64)

    if z_range is None:
        z_axis = np.arange(cols, dtype=np.float64)
    else:
        z_axis = np.linspace(float(z_range[0]), float(z_range[1]), cols, dtype=np.float64)

    X = np.repeat(x_axis[:, np.newaxis], cols, axis=1)
    Z = np.repeat(z_axis[np.newaxis, :], rows, axis=0)
    Y = np.zeros_like(X)
    return X, Y, Z


def save_mat(path, tl_array, extent=None):
    """Save a .mat HDF5 with axes compatible with original convention."""
    tl_save = tl_array.T

    if extent is not None:
        x_left, x_right, z_bottom, z_top = [float(v) for v in extent]
        X, Y, Z = create_xyz_grid(
            tl_save.shape,
            x_range=(x_left, x_right),
            z_range=(z_bottom, z_top),
        )
    else:
        X, Y, Z = create_xyz_grid(tl_save.shape)

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("R", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("Z", data=Z)
        f.create_dataset("tl", data=tl_save)


# ------------------------------
# FIGSIZE HELPER
# ------------------------------

def adaptive_figsize(data_h, data_w, n_rows, n_cols,
                     base=3.0, max_aspect=4.0, extra_w=1.2):
    """Compute a figure size based on data aspect ratio."""
    ar = data_h / data_w
    ar = max(1.0 / max_aspect, min(ar, max_aspect))

    if ar >= 1.0:
        subplot_w = base
        subplot_h = base * ar
    else:
        subplot_h = base
        subplot_w = base / ar

    fig_w = (subplot_w + extra_w) * n_cols
    fig_h = subplot_h * n_rows + 1.5
    return (fig_w, fig_h)


# ------------------------------
# METRICS
# ------------------------------

def compute_error_metrics(actual, predicted):
    """Compute MAE, RMSE, MAPE, and max error."""
    diff = np.abs(actual - predicted)

    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    mask = np.abs(actual) >= 1e-6
    if np.any(mask):
        mape = 100.0 * np.mean(np.abs(actual[mask] - predicted[mask]) / np.abs(actual[mask]))
    else:
        mape = np.inf

    max_error = np.max(diff)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "max_error": max_error,
    }


# ------------------------------
# GENERATION
# ------------------------------

def generate_from_seed(model, seed_tensor, target_angle, norm, noise_std, device):
    """Generate a plane from a seed using conditional FNO."""
    cond = torch.tensor(
        [[norm.normalize_angle(target_angle)]],
        dtype=torch.float32,
        device=device,
    )

    noisy_seed = seed_tensor + torch.randn_like(seed_tensor) * noise_std
    gen_norm = model(noisy_seed, cond)

    gen_np = np.clip(gen_norm.cpu().squeeze().numpy(), 0, 1)
    gen_tl = norm.denormalize_tl(gen_np)
    return gen_tl


# ------------------------------
# INFERENCE PIPELINE
# ------------------------------

def load_inference_pipeline(model_path, data_folder, roi_h, roi_w,
                            rois_per_plane,
                            roi_mode="center_max", roi_corner=(0, 0)):
    """Load trained model and data for inference (generate/analysis)."""
    from model import ConditionalFNO2d
    from config import (
        FNO_MODES1, FNO_MODES2, FNO_WIDTH, FNO_DEPTH, FNO_USE_COORDS, FNO_DROPOUT,
        verify_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    verify_config(checkpoint)

    model = ConditionalFNO2d(
        modes1=FNO_MODES1,
        modes2=FNO_MODES2,
        width=FNO_WIDTH,
        depth=FNO_DEPTH,
        use_coords=FNO_USE_COORDS,
        dropout=FNO_DROPOUT,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from: {model_path}")

    norm = Normalizer.from_checkpoint(checkpoint)

    print("Loading original data...")
    rois, roi_angles, roi_extents = load_all_rois(
        data_folder, roi_h, roi_w,
        rois_per_plane,
        roi_mode=roi_mode, roi_corner=roi_corner,
    )
    rois_norm = norm.normalize_tl(rois)
    angles_norm = norm.normalize_angles(roi_angles)

    print(f"ROIs: {len(rois)}  |  Shape: {rois.shape[1]}x{rois.shape[2]}")
    print(f"TL range: [{norm.tl_min:.2f}, {norm.tl_max:.2f}]")
    print(f"Angle range: [{norm.angle_min:.2f}, {norm.angle_max:.2f}]")

    return device, model, rois_norm, roi_angles, angles_norm, norm, roi_extents
