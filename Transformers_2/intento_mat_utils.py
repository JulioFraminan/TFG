import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

try:
    from config import USE_BLOCK_VARIABLES
except Exception:
    USE_BLOCK_VARIABLES = False


ANGLE_RE = re.compile(r"PlaneAngle(-?\d+(?:\.\d+)?)")


def _safe_get_dataset(file_obj: h5py.File, keys: Sequence[str]) -> np.ndarray:
    for key in keys:
        if key in file_obj:
            return file_obj[key][:]
    available = list(file_obj.keys())
    raise KeyError(f"None of keys {keys} found. Available keys: {available}")


def _sanitize_tl_array(tl: np.ndarray) -> np.ndarray:
    """Convert TL to float32 and replace NaN/Inf with a stable finite value.

    If there are finite values, use the minimum finite value as fill; otherwise use 0.0.
    """
    tl = np.asarray(tl, dtype=np.float32)
    finite_mask = np.isfinite(tl)
    if np.all(finite_mask):
        return tl
    if np.any(finite_mask):
        fill_value = float(np.nanmin(tl[finite_mask]))
    else:
        fill_value = 0.0
    return np.nan_to_num(tl, nan=fill_value, posinf=fill_value, neginf=fill_value)


def _get_z_axis(z_raw: np.ndarray) -> np.ndarray:
    """Resolve the physical Z axis from either row-major or column-major grids.

    Prefer the trace (row or column) with larger variation (std). This avoids
    selecting a constant column/row (e.g. all values -40) which produces a
    degenerate plotting extent and blank PNGs.
    """
    z_row = np.asarray(z_raw[0, :]).flatten()
    z_col = np.asarray(z_raw[:, 0]).flatten()

    # If one axis is trivially length 1, prefer the other when possible.
    if z_row.size > 1 and z_col.size <= 1:
        return z_row
    if z_col.size > 1 and z_row.size <= 1:
        return z_col

    # Compute robust estimate of variation (std) for each trace and choose
    # the one with greater variability. Fallback to length-based choice.
    try:
        std_row = float(np.nanstd(z_row)) if z_row.size > 0 else 0.0
        std_col = float(np.nanstd(z_col)) if z_col.size > 0 else 0.0
    except Exception:
        std_row = std_col = 0.0

    if std_row > std_col:
        return z_row
    if std_col > std_row:
        return z_col

    return z_row if z_row.size >= z_col.size else z_col


def extract_angle_from_filename(filename: str) -> float:
    match = ANGLE_RE.search(filename)
    return float(match.group(1)) if match else 0.0


def extract_roi_at(tl: np.ndarray, center_i: int, center_j: int, roi_h: int, roi_w: int) -> Tuple[np.ndarray, int, int]:
    half_h = roi_h // 2
    half_w = roi_w // 2
    rows, cols = tl.shape

    i0, i1 = max(center_i - half_h, 0), min(center_i + half_h, rows)
    j0, j1 = max(center_j - half_w, 0), min(center_j + half_w, cols)

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


def phys_to_pixel(x_raw: np.ndarray, z_raw: np.ndarray, x_phys: float, z_phys: float) -> Tuple[int, int]:
    x_axis = x_raw[:, 0]
    z_axis = _get_z_axis(z_raw)

    col_idx = int(np.argmin(np.abs(x_axis - x_phys)))
    row_idx = int(np.argmin(np.abs(z_axis - z_phys)))

    return row_idx, col_idx


def phys_size_to_pixels(x_raw: np.ndarray, z_raw: np.ndarray, roi_h: float, roi_w: float) -> Tuple[int, int]:
    def _is_integer_like(value) -> bool:
        return isinstance(value, (int, np.integer)) or (isinstance(value, float) and float(value).is_integer())

    # Keep backwards compatibility: values like 700 or 700.0 are interpreted as pixel sizes.
    if _is_integer_like(roi_h) and _is_integer_like(roi_w):
        return int(round(float(roi_h))), int(round(float(roi_w)))

    try:
        x_axis = x_raw[:, 0]
        z_axis = _get_z_axis(z_raw)
        dxs = np.diff(x_axis)
        dzs = np.diff(z_axis)
        dx = float(np.median(dxs)) if dxs.size > 0 else 1.0
        dz = float(np.median(dzs)) if dzs.size > 0 else 1.0
    except Exception:
        dx = 1.0
        dz = 1.0

    roi_h_px = int(max(1, round(float(roi_h) / dz))) if not isinstance(roi_h, (int, np.integer)) else int(roi_h)
    roi_w_px = int(max(1, round(float(roi_w) / dx))) if not isinstance(roi_w, (int, np.integer)) else int(roi_w)
    return roi_h_px, roi_w_px


def extract_roi_at_corner(
    tl: np.ndarray,
    corner_i: int,
    corner_j: int,
    roi_h: int,
    roi_w: int,
) -> Tuple[np.ndarray, int, int]:
    rows, cols = tl.shape

    i_top = int(corner_i)
    j0 = int(corner_j)
    i0 = i_top - roi_h + 1

    if i_top < 0 or i_top >= rows:
        raise ValueError(f"Top row {i_top} outside array bounds [0, {rows - 1}]")
    if j0 < 0 or j0 >= cols:
        raise ValueError(f"Left column {j0} outside array bounds [0, {cols - 1}]")
    if i0 < 0:
        raise ValueError(
            "ROI exceeds bottom edge. Increase Z corner or reduce roi_height. "
            f"Computed start row = {i0}."
        )
    if j0 + roi_w > cols:
        raise ValueError(
            "ROI exceeds right edge. Reduce roi_width or adjust X corner. "
            f"Requested end col = {j0 + roi_w}, max col = {cols}."
        )

    return tl[i0:i_top + 1, j0:j0 + roi_w], i0, j0


def compute_roi_extent(
    x_raw: np.ndarray,
    z_raw: np.ndarray,
    i0: int,
    j0: int,
    roi_h: int,
    roi_w: int,
) -> List[float]:
    x_axis = x_raw[:, 0]
    z_axis = _get_z_axis(z_raw)

    x_left = float(x_axis[j0])
    x_right = float(x_axis[min(j0 + roi_w - 1, len(x_axis) - 1)])
    z_bottom = float(z_axis[i0])
    z_top = float(z_axis[min(i0 + roi_h - 1, len(z_axis) - 1)])
    return [x_left, x_right, z_bottom, z_top]


def extract_multiple_rois(
    tl: np.ndarray,
    roi_h: int,
    roi_w: int,
    num_rois: int = 5,
    mode: str = "center_max",
    corner_pixel: Tuple[int, int] = (0, 0),
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    rois: List[np.ndarray] = []
    origins: List[Tuple[int, int]] = []
    rows, cols = tl.shape

    if mode == "corner_fixed":
        roi, i0, j0 = extract_roi_at_corner(tl, corner_pixel[0], corner_pixel[1], roi_h, roi_w)
        rois.append(roi)
        origins.append((i0, j0))
        return rois, origins

    half_h = roi_h // 2
    half_w = roi_w // 2

    max_i, max_j = np.unravel_index(np.argmax(tl), tl.shape)
    roi, i0, j0 = extract_roi_at(tl, max_i, max_j, roi_h, roi_w)
    rois.append(roi)
    origins.append((i0, j0))

    threshold = np.percentile(tl, 80)
    high_coords = np.argwhere(tl >= threshold)

    for _ in range(max(0, num_rois - 1)):
        if len(high_coords) > 0:
            center_i, center_j = high_coords[np.random.randint(len(high_coords))]
        else:
            center_i = np.random.randint(half_h, max(half_h + 1, rows - half_h))
            center_j = np.random.randint(half_w, max(half_w + 1, cols - half_w))

        center_i = int(np.clip(center_i, half_h, max(half_h, rows - half_h)))
        center_j = int(np.clip(center_j, half_w, max(half_w, cols - half_w)))
        roi, i0, j0 = extract_roi_at(tl, center_i, center_j, roi_h, roi_w)
        rois.append(roi)
        origins.append((i0, j0))

    return rois, origins


@dataclass
class DatasetBundle:
    rois: np.ndarray
    angles_deg: np.ndarray
    extents: List[List[float]]


def load_rois_from_folder(
    folder: str,
    roi_h: float,
    roi_w: float,
    rois_per_plane: int = 1,
    roi_mode: str = "corner_fixed",
    roi_corner: Tuple[float, float] = (0.0, 0.0),
    verbose: bool = True,
) -> DatasetBundle:
    all_rois: List[np.ndarray] = []
    all_angles: List[float] = []
    all_extents: List[List[float]] = []

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if os.path.isdir(path):
            continue
        if not filename.endswith(".mat") or "PlaneAngle" not in filename:
            continue

        try:
            with h5py.File(path, "r") as file_obj:
                # Prefer smoothed TL when available, then standard TL, then block TL (optional)
                tl = None
                tl_source = None
                candidates = ["tl_smooth", "tl", "TL", "tL"]
                if USE_BLOCK_VARIABLES:
                    candidates.append("tl_block")
                for candidate in candidates:
                    if candidate in file_obj:
                        tl_source = candidate
                        tl = _sanitize_tl_array(file_obj[candidate][:].T)
                        break
                if tl is None:
                    raise KeyError("No TL variant found in file")

                # Prefer full-resolution coordinate grids when they match the loaded TL.
                # This mirrors the hybrid loader and avoids using block axes for full-resolution/smoothed TL.
                x_raw = None
                z_raw = None
                if "R" in file_obj and "Z" in file_obj:
                    r_full = file_obj["R"][:]
                    z_full = file_obj["Z"][:]
                    if r_full.shape == tl.shape and z_full.shape == tl.shape:
                        x_raw = r_full
                        z_raw = z_full

                if x_raw is None or z_raw is None:
                    if tl_source == "tl_block":
                        x_raw = _safe_get_dataset(file_obj, ["R_block", "X", "x", "R", "r"])
                        z_raw = _safe_get_dataset(file_obj, ["Z_block", "Z", "z"])
                    else:
                        x_raw = _safe_get_dataset(file_obj, ["X", "x", "R", "r", "R_block"])
                        z_raw = _safe_get_dataset(file_obj, ["Z", "z", "Z_block"])
        except Exception as error:
            if verbose:
                print(f"[WARN] Could not read {filename}: {error}")
            continue

        angle = extract_angle_from_filename(filename)
        roi_h_px, roi_w_px = phys_size_to_pixels(x_raw, z_raw, roi_h, roi_w)

        rows, cols = tl.shape
        roi_h_px = min(roi_h_px, rows)
        roi_w_px = min(roi_w_px, cols)

        corner_pixel = (0, 0)
        if roi_mode == "corner_fixed":
            row_px, col_px = phys_to_pixel(x_raw, z_raw, roi_corner[0], roi_corner[1])

            # Clamp the corner so the ROI fits completely inside the plane.
            max_top = max(roi_h_px - 1, 0)
            max_left = max(cols - roi_w_px, 0)
            adj_row_px = min(max(row_px, max_top), rows - 1)
            adj_col_px = min(max(col_px, 0), max_left)
            if verbose and (adj_row_px != row_px or adj_col_px != col_px):
                print(
                    f"  [i] {filename}: corner adjusted so the ROI fits -> pixel ({adj_row_px}, {adj_col_px})"
                )
            row_px, col_px = adj_row_px, adj_col_px
            corner_pixel = (row_px, col_px)

        try:
            plane_rois, plane_origins = extract_multiple_rois(
                tl,
                roi_h=roi_h_px,
                roi_w=roi_w_px,
                num_rois=rois_per_plane,
                mode=roi_mode,
                corner_pixel=corner_pixel,
            )
        except Exception as error:
            if verbose:
                print(f"[WARN] Skipping {filename} due to ROI error: {error}")
            continue

        for roi, (origin_i, origin_j) in zip(plane_rois, plane_origins):
            extent = compute_roi_extent(x_raw, z_raw, origin_i, origin_j, roi_h_px, roi_w_px)
            all_rois.append(roi.astype(np.float32))
            all_angles.append(float(angle))
            all_extents.append(extent)

        if verbose:
            print(f"[OK] {filename} -> {len(plane_rois)} ROI(s), angle={angle:.2f}")

    if len(all_rois) == 0:
        return DatasetBundle(
            rois=np.array([], dtype=np.float32),
            angles_deg=np.array([], dtype=np.float32),
            extents=[],
        )

    return DatasetBundle(
        rois=np.array(all_rois, dtype=np.float32),
        angles_deg=np.array(all_angles, dtype=np.float32),
        extents=all_extents,
    )


@dataclass
class NormalizationStats:
    tl_min: float
    tl_max: float
    angle_mean: float
    angle_std: float

    @classmethod
    def from_training_data(cls, rois: np.ndarray, angles_deg: np.ndarray) -> "NormalizationStats":
        if rois.size == 0:
            raise ValueError("Training ROIs are empty")
        if angles_deg.size == 0:
            raise ValueError("Training angles are empty")

        angle_std = float(np.std(angles_deg))
        if angle_std < 1e-8:
            angle_std = 1.0

        return cls(
            tl_min=float(np.nanmin(rois)),
            tl_max=float(np.nanmax(rois)),
            angle_mean=float(np.mean(angles_deg)),
            angle_std=angle_std,
        )


def normalize_fields_01(fields: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    denom = max(1e-8, stats.tl_max - stats.tl_min)
    return np.clip((fields - stats.tl_min) / denom, 0.0, 1.0).astype(np.float32)


def denormalize_fields_01(fields_01: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (fields_01 * (stats.tl_max - stats.tl_min) + stats.tl_min).astype(np.float32)


def normalize_angles(angles_deg: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((angles_deg - stats.angle_mean) / max(1e-8, stats.angle_std)).astype(np.float32)


def normalize_single_angle(angle_deg: float, stats: NormalizationStats) -> float:
    return float((angle_deg - stats.angle_mean) / max(1e-8, stats.angle_std))


def resize_fields(fields: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if fields.ndim != 3:
        raise ValueError(f"Expected fields shape (N, H, W), got {fields.shape}")

    h, w = int(fields.shape[1]), int(fields.shape[2])
    if h == target_h and w == target_w:
        return fields.astype(np.float32)

    tensor = torch.from_numpy(fields[:, None, :, :]).float()
    resized = F.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized[:, 0, :, :].cpu().numpy().astype(np.float32)


def ensure_min_samples(
    fields_01: np.ndarray,
    angles_norm: np.ndarray,
    minimum: int = 100,
    noise_std: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = fields_01.shape[0]
    if n_samples >= minimum:
        return fields_01, angles_norm

    repeats = int(np.ceil(minimum / max(1, n_samples)))
    fields_rep = np.repeat(fields_01, repeats, axis=0)
    angles_rep = np.repeat(angles_norm, repeats, axis=0)

    noise = np.random.normal(0.0, noise_std, size=fields_rep.shape).astype(np.float32)
    fields_rep = np.clip(fields_rep + noise, 0.0, 1.0)

    return fields_rep[:minimum], angles_rep[:minimum]


def create_xyz_grid(
    shape: Tuple[int, int],
    extent: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Stored tl layout is (nx, nz): first axis maps to X, second axis maps to Z.
    nx, nz = int(shape[0]), int(shape[1])

    if extent is None:
        x_left, x_right = 0.0, float(max(0, nx - 1))
        z_bottom, z_top = 0.0, float(max(0, nz - 1))
    else:
        x_left, x_right, z_bottom, z_top = [float(value) for value in extent]

    x_axis = np.linspace(x_left, x_right, nx, dtype=np.float64)
    z_axis = np.linspace(z_bottom, z_top, nz, dtype=np.float64)

    x_grid = np.repeat(x_axis[:, None], nz, axis=1)
    z_grid = np.repeat(z_axis[None, :], nx, axis=0)
    y_grid = np.zeros((nx, nz), dtype=np.float64)
    return x_grid, y_grid, z_grid


def save_mat_h5(path: str, tl_array: np.ndarray, extent: Optional[Sequence[float]] = None) -> None:
    tl_to_save = tl_array.T
    x_grid, y_grid, z_grid = create_xyz_grid(tl_to_save.shape, extent=extent)
    with h5py.File(path, "w") as file_obj:
        file_obj.create_dataset("X", data=x_grid)
        file_obj.create_dataset("Y", data=y_grid)
        file_obj.create_dataset("Z", data=z_grid)
        file_obj.create_dataset("tl", data=tl_to_save)


def parse_angle_list(angle_text: str) -> List[float]:
    if not angle_text.strip():
        return []
    return [float(chunk.strip()) for chunk in angle_text.split(",") if chunk.strip()]


def default_intento_input_folder(repo_root: str) -> str:
    return os.path.join(repo_root, "Transformers_2", "input")


def default_intento_validation_folder(repo_root: str) -> str:
    return os.path.join(default_intento_input_folder(repo_root), "validation")


def maybe_load_reference_rois(
    input_folder: str,
    roi_h: int,
    roi_w: int,
    roi_mode: str,
    roi_corner: Tuple[float, float],
) -> Optional[DatasetBundle]:
    try:
        bundle = load_rois_from_folder(
            folder=input_folder,
            roi_h=roi_h,
            roi_w=roi_w,
            rois_per_plane=1,
            roi_mode=roi_mode,
            roi_corner=roi_corner,
            verbose=False,
        )
    except Exception:
        return None

    if bundle.rois.size == 0:
        return None
    return bundle
