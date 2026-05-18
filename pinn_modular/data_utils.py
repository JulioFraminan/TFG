import os
import re

import h5py
import numpy as np
import torch


def _safe_get_dataset(f, keys):
    """Return the first dataset found in keys inside an h5 file."""
    for k in keys:
        if k in f:
            return f[k][:]
    avail = list(f.keys())
    raise KeyError(f"No dataset found for keys {keys}. Available: {avail}")


# -----------------------------------------------------------------------------
# ROI extraction helpers
# -----------------------------------------------------------------------------

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
    """Detecta automáticamente la estructura de Z_raw (antigua o nueva) y devuelve el eje Z completo."""
    z_row = Z_raw[0, :].flatten()
    z_col = Z_raw[:, 0].flatten()
    return z_row if z_row.size > z_col.size else z_col


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
    """Convert a physical ROI size to pixels.

    If roi_h/roi_w are ints they are returned as-is.
    """
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
    """Extract a roi_h x roi_w patch using (corner_i, corner_j) as upper-left.

    corner_i is the visual top row (origin='lower'). The ROI grows down and right.
    """
    rows, cols = tl.shape

    i_top = int(corner_i)
    j0 = int(corner_j)
    i0 = i_top - roi_h + 1

    if i_top < 0 or i_top >= rows:
        raise ValueError(
            f"Top row ({i_top}) outside plane (0..{rows - 1}). Check ROI_CORNER Z."
        )
    if j0 < 0 or j0 >= cols:
        raise ValueError(
            f"Left col ({j0}) outside plane (0..{cols - 1}). Check ROI_CORNER X."
        )
    if i0 < 0:
        raise ValueError(
            f"ROI goes below plane: top={i_top}, ROI_HEIGHT={roi_h} -> start={i0} (< 0)."
        )
    if j0 + roi_w > cols:
        overflow = j0 + roi_w - cols
        raise ValueError(
            f"ROI goes beyond right edge: left={j0}, ROI_WIDTH={roi_w}, cols={cols}. Overflow={overflow} px."
        )

    roi = tl[i0:i_top + 1, j0:j0 + roi_w]
    return roi, i0, j0


def compute_roi_extent(X_raw, Z_raw, i0, j0, roi_h, roi_w):
    """Compute physical extent [x_left, x_right, z_bottom, z_top] for an ROI."""
    x_axis = X_raw[:, 0]
    z_axis = Z_raw[0, :]

    x_left = float(x_axis[j0])
    x_right = float(x_axis[min(j0 + roi_w - 1, len(x_axis) - 1)])
    z_bottom = float(z_axis[i0])
    z_top = float(z_axis[min(i0 + roi_h - 1, len(z_axis) - 1)])

    return [x_left, x_right, z_bottom, z_top]


def extract_multiple_rois(tl, roi_h, roi_w, num_rois=5,
                          mode="center_max", corner_pixel=(0, 0)):
    """Extract ROIs from a plane with a selected mode."""
    rois = []
    origins = []
    rows, cols = tl.shape

    if mode == "corner_fixed":
        roi, i0, j0 = extract_roi_at_corner(tl, corner_pixel[0], corner_pixel[1], roi_h, roi_w)
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


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

from config import USE_BLOCK_VARIABLES

def _extract_angle(fname):
    """Extract plane angle from a filename containing PlaneAngle."""
    m = re.search(r"PlaneAngle(-?\d+(?:\.\d+)?)", fname)
    return float(m.group(1)) if m else 0.0


def _load_rois_from_folder(folder, roi_h, roi_w, rois_per_plane,
                           roi_mode, roi_corner, verbose_bounds=True):
    """Load .mat files and extract ROIs."""
    all_rois, all_angles, all_extents = [], [], []

    if roi_mode == "corner_fixed":
        print(f"  ROI mode: corner_fixed | corner (X, Z) = {roi_corner}")

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".mat") or "PlaneAngle" not in fname:
            continue
        fpath = os.path.join(folder, fname)
        try:
            with h5py.File(fpath, "r") as f:
                tl_keys = ["tl_block", "tl", "TL", "tL"] if USE_BLOCK_VARIABLES else ["tl", "TL", "tL", "tl_block"]
                tl = _safe_get_dataset(f, tl_keys).T
                X_raw = _safe_get_dataset(f, ["R_block", "X", "x", "R", "r"])
                Z_raw = _safe_get_dataset(f, ["Z_block", "Z", "z"])
        except Exception as e:
            print(f"  [warn] Error loading {fname}: {e}")
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

            if verbose_bounds:
                x_min, x_max = float(X_raw.min()), float(X_raw.max())
                z_min, z_max = float(Z_raw.min()), float(Z_raw.max())
                clamped = False
                if x_phys < x_min or x_phys > x_max:
                    print(f"  [warn] {fname}: X={x_phys} out of range [{x_min:.1f}, {x_max:.1f}]")
                    clamped = True
                if z_phys < z_min or z_phys > z_max:
                    print(f"  [warn] {fname}: Z={z_phys} out of range [{z_min:.1f}, {z_max:.1f}]")
                    clamped = True
                if not clamped:
                    print(f"  [info] {fname}: corner (X, Z)=({x_phys}, {z_phys}) -> pixel ({row_px}, {col_px})")

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
        print(f"  [ok] {fname} -> {len(plane_rois)} ROIs (angle={angle:.2f})")

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
    print("  Loading validation planes")
    print("=" * 60)

    rois, angles, extents = _load_rois_from_folder(
        folder, roi_h, roi_w, 1,
        roi_mode, roi_corner, verbose_bounds=False,
    )
    if len(rois) == 0:
        return None, [], []
    return rois, angles, extents


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

class Normalizer:
    """Store TL and angle ranges and provide normalization helpers."""

    def __init__(self, rois, roi_angles):
        if rois is None or getattr(rois, "size", 0) == 0:
            raise ValueError(
                "No ROIs found. Check input data and ROI parameters."
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

    def normalize_angles(self, angles_array):
        a = np.array(angles_array, dtype=np.float32)
        return (a - self.angle_min) / (self.angle_max - self.angle_min + 1e-8)

    def state_dict(self):
        return {
            "tl_min": self.tl_min,
            "tl_max": self.tl_max,
            "angle_min": self.angle_min,
            "angle_max": self.angle_max,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        obj = cls.__new__(cls)
        obj.tl_min = checkpoint["tl_min"]
        obj.tl_max = checkpoint["tl_max"]
        obj.angle_min = checkpoint["angle_min"]
        obj.angle_max = checkpoint["angle_max"]
        return obj


class CoordNormalizer:
    """Normalize x, z, angle to [-1, 1]."""

    def __init__(self, x_min, x_max, z_min, z_max, angle_min, angle_max):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)

        self.scale_x = 2.0 / (self.x_max - self.x_min + 1e-8)
        self.scale_z = 2.0 / (self.z_max - self.z_min + 1e-8)
        self.scale_a = 2.0 / (self.angle_max - self.angle_min + 1e-8)

    def normalize(self, x, z, angle):
        x = np.asarray(x, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        angle = np.asarray(angle, dtype=np.float32)
        x_n = (x - self.x_min) * self.scale_x - 1.0
        z_n = (z - self.z_min) * self.scale_z - 1.0
        a_n = (angle - self.angle_min) * self.scale_a - 1.0
        return x_n, z_n, a_n

    def state_dict(self):
        return {
            "coord_x_min": self.x_min,
            "coord_x_max": self.x_max,
            "coord_z_min": self.z_min,
            "coord_z_max": self.z_max,
            "coord_angle_min": self.angle_min,
            "coord_angle_max": self.angle_max,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if "coord_x_min" not in checkpoint:
            return None
        return cls(
            checkpoint["coord_x_min"],
            checkpoint["coord_x_max"],
            checkpoint["coord_z_min"],
            checkpoint["coord_z_max"],
            checkpoint["coord_angle_min"],
            checkpoint["coord_angle_max"],
        )


# -----------------------------------------------------------------------------
# Data augmentation
# -----------------------------------------------------------------------------

def augment(data, angles):
    """Augment images and replicate angles.

    Only noise and contrast scaling are used.
    Factor: 4 (original + 3 augmentations).
    """
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


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def create_xyz_grid(shape, x_range=None, z_range=None):
    """Create X, Y, Z grids compatible with the input .mat convention."""
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
    """Save an HDF5 .mat with the same axis convention as input files."""
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


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

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


def compute_error_metrics(actual, predicted):
    """Compute MAE, RMSE, MAPE and max error."""
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


# -----------------------------------------------------------------------------
# Sampling and inference helpers
# -----------------------------------------------------------------------------

def build_axes_from_extent(extent, roi_h, roi_w):
    """Build x and z axes from an extent and ROI shape."""
    x_left, x_right, z_bottom, z_top = [float(v) for v in extent]
    x_axis = np.linspace(x_left, x_right, roi_w, dtype=np.float32)
    z_axis = np.linspace(z_bottom, z_top, roi_h, dtype=np.float32)
    return x_axis, z_axis


def get_global_bounds(roi_extents):
    """Return global bounds (x_min, x_max, z_min, z_max) across extents."""
    x_min = min(ext[0] for ext in roi_extents)
    x_max = max(ext[1] for ext in roi_extents)
    z_min = min(ext[2] for ext in roi_extents)
    z_max = max(ext[3] for ext in roi_extents)
    return x_min, x_max, z_min, z_max


def sample_points_from_rois(rois, roi_angles, roi_extents, points_per_roi, rng):
    """Sample points from each ROI and return x, z, angle, tl arrays."""
    xs, zs, angs, tls = [], [], [], []

    for roi, angle, extent in zip(rois, roi_angles, roi_extents):
        h, w = roi.shape
        x_axis, z_axis = build_axes_from_extent(extent, h, w)

        flat = roi.reshape(-1)
        total = flat.size
        n = min(points_per_roi, total)
        replace = total < n
        idx = rng.choice(total, size=n, replace=replace)

        rows = idx // w
        cols = idx % w
        xs.append(x_axis[cols])
        zs.append(z_axis[rows])
        angs.append(np.full(n, angle, dtype=np.float32))
        tls.append(flat[idx].astype(np.float32))

    return (
        np.concatenate(xs),
        np.concatenate(zs),
        np.concatenate(angs),
        np.concatenate(tls),
    )


def predict_on_grid(model, coord_norm, angle_deg, extent, shape, device, batch_size=20000):
    """Predict TL on a full grid for a given angle and extent."""
    h, w = shape
    x_axis, z_axis = build_axes_from_extent(extent, h, w)
    X, Z = np.meshgrid(x_axis, z_axis)
    A = np.full_like(X, angle_deg, dtype=np.float32)

    x_n, z_n, a_n = coord_norm.normalize(X, Z, A)
    coords = np.stack([x_n, z_n, a_n], axis=-1).reshape(-1, 3).astype(np.float32)

    preds = []
    with torch.no_grad():
        for i in range(0, coords.shape[0], batch_size):
            batch = torch.from_numpy(coords[i:i + batch_size]).to(device)
            out = model(batch).cpu().numpy()
            preds.append(out)

    pred = np.concatenate(preds, axis=0).reshape(h, w)
    return pred


# -----------------------------------------------------------------------------
# Inference pipeline
# -----------------------------------------------------------------------------

def load_inference_pipeline(model_path, data_folder, roi_h, roi_w,
                            rois_per_plane, roi_mode="center_max", roi_corner=(0, 0)):
    """Load a trained model and data for inference."""
    from model import PINN
    from config import (
        verify_config,
        PINN_HIDDEN_DIM, PINN_NUM_LAYERS, PINN_ACTIVATION, PINN_DROPOUT,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    verify_config(checkpoint)

    hidden_dim = checkpoint.get("pinn_hidden_dim", PINN_HIDDEN_DIM)
    num_layers = checkpoint.get("pinn_num_layers", PINN_NUM_LAYERS)
    activation = checkpoint.get("pinn_activation", PINN_ACTIVATION)
    dropout = checkpoint.get("pinn_dropout", PINN_DROPOUT)

    model = PINN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from: {model_path}")

    norm = Normalizer.from_checkpoint(checkpoint)
    coord_norm = CoordNormalizer.from_checkpoint(checkpoint)

    rois, roi_angles, roi_extents = load_all_rois(
        data_folder, roi_h, roi_w,
        rois_per_plane,
        roi_mode=roi_mode, roi_corner=roi_corner,
    )

    if coord_norm is None:
        x_min, x_max, z_min, z_max = get_global_bounds(roi_extents)
        coord_norm = CoordNormalizer(x_min, x_max, z_min, z_max, norm.angle_min, norm.angle_max)

    print(f"ROIs: {len(rois)}")
    print(f"TL range: [{norm.tl_min:.2f}, {norm.tl_max:.2f}]")
    print(f"Angle range: [{norm.angle_min:.2f}, {norm.angle_max:.2f}]")

    return device, model, rois, roi_angles, norm, coord_norm, roi_extents
