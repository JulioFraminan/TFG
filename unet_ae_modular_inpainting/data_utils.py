import os
import re

import h5py
import numpy as np
import torch


def _safe_get_dataset(f, keys):
    """Devuelve el primer dataset presente en 'keys' dentro del archivo h5py.File f.

    keys: lista de nombres candidatos (orden de preferencia).
    Lanza KeyError si ninguno existe (incluye lista de claves disponibles).
    """
    for k in keys:
        if k in f:
            return f[k][:]
    avail = list(f.keys())
    raise KeyError(f"Ninguna de las claves {keys} encontrada en el .mat. Claves disponibles: {avail}")


def _sanitize_tl_array(tl):
    """Convierte TL a float32 y sustituye NaN/Inf por un valor finito estable.

    Si hay al menos algún valor finito se rellena NaN/Inf por el mínimo finito
    observado; si todo es no-finito se rellena con 0.0.
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


# ══════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE ROIs (rectangulares)
# ══════════════════════════════════════════════════════════════════════

def extract_roi_at(tl, ci, cj, roi_h, roi_w):
    """Extrae un parche de roi_h x roi_w centrado en (ci, cj).

    Returns:
        (roi, i0, j0) -- roi array y esquina superior-izquierda en pixeles.
    """
    half_h = roi_h // 2
    half_w = roi_w // 2
    rows, cols = tl.shape

    i0, i1 = max(ci - half_h, 0), min(ci + half_h, rows)
    j0, j1 = max(cj - half_w, 0), min(cj + half_w, cols)

    # Ajustar si el parche queda mas pequeño que lo pedido
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
    """Detecta automáticamente la estructura de Z_raw y devuelve el eje Z completo.

    En algunos .mat, una de las dos proyecciones puede ser constante aunque ambas
    tengan el mismo tamaño. En ese caso elegimos la que tenga más variación real.
    """
    z_row = Z_raw[0, :].flatten()
    z_col = Z_raw[:, 0].flatten()

    row_var = float(np.ptp(z_row)) if z_row.size > 1 else 0.0
    col_var = float(np.ptp(z_col)) if z_col.size > 1 else 0.0

    if row_var > col_var:
        return z_row
    if col_var > row_var:
        return z_col

    # Empate: mantener la heurística original por tamaño.
    return z_row if z_row.size >= z_col.size else z_col


def phys_to_pixel(X_raw, Z_raw, x_phys, z_phys):
    """Convierte coordenadas físicas (X, Z) a índices de píxel en tl.T.

    En el .mat (antes de transponer):
      • X varía a lo largo del eje 0 (filas raw)  → columnas tras .T
      • Z varía a lo largo del eje 1 (cols raw)   → filas tras .T

    Soporta ambas estructuras:
      • Z antiguo: shape (1, 60)  → Z[0, :] tiene datos
      • Z nuevo:  shape (60, 1)   → Z[:, 0] tiene datos

    Returns:
        (row_idx, col_idx) índices en la matriz transpuesta tl.T
    """
    # Eje X en raw: X[:, 0] (cada fila tiene una X distinta)
    x_axis = X_raw[:, 0]          # vector 1-D que contiene todos los X
    col_idx = int(np.argmin(np.abs(x_axis - x_phys)))  # → col en tl.T

    # Eje Z: usar helper para detectar automáticamente
    z_axis = _get_z_axis(Z_raw)
    
    row_idx = int(np.argmin(np.abs(z_axis - z_phys)))  # → row en tl.T

    return row_idx, col_idx


def phys_size_to_pixels(X_raw, Z_raw, roi_h, roi_w):
    """Convierte un tamaño físico (metros) de ROI a píxeles usando X_raw/Z_raw.

    Si roi_h/roi_w son enteros se devuelven tal cuales (se interpretan como píxeles).
    Si son float se interpretan como metros y se convierten usando la resolución
    promedio de X_raw (eje X) y Z_raw (eje Z).
    Devuelve (roi_h_px, roi_w_px).
    """
    # Si ya son enteros (píxeles) mantenerlos
    if isinstance(roi_h, (int, np.integer)) and isinstance(roi_w, (int, np.integer)):
        return int(roi_h), int(roi_w)

    # Calcular resolución física por píxel
    try:
        x_axis = X_raw[:, 0]
        z_axis = _get_z_axis(Z_raw)
    except Exception:
        # Fallback: asumir resolución 1.0 si no hay ejes válidos
        dx = 1.0
        dz = 1.0
    else:
        # usar diferencia media (robusta ante pequeños irregularidades)
        dxs = np.diff(x_axis)
        dzs = np.diff(z_axis)
        dx = float(np.median(dxs)) if dxs.size > 0 else 1.0
        dz = float(np.median(dzs)) if dzs.size > 0 else 1.0

    # Si roi_h/roi_w son floats -> metros
    roi_h_px = int(max(1, round(float(roi_h) / dz))) if not isinstance(roi_h, (int, np.integer)) else int(roi_h)
    roi_w_px = int(max(1, round(float(roi_w) / dx))) if not isinstance(roi_w, (int, np.integer)) else int(roi_w)

    return roi_h_px, roi_w_px


def extract_roi_at_corner(tl, corner_i, corner_j, roi_h, roi_w):
    """Extrae un parche de roi_h x roi_w usando (corner_i, corner_j)
    como esquina SUPERIOR-IZQUIERDA del grafico (origin='lower').

    corner_i = fila que corresponde al borde SUPERIOR visual (Z mas alto).
    corner_j = columna que corresponde al borde IZQUIERDO (X mas bajo).

    La ROI se extiende HACIA ABAJO (filas menores) y a la derecha.
    Si la ROI excede los limites, se lanza un error.

    Returns:
        (roi, i0, j0) -- roi array y esquina inferior-izq en pixeles.
    """
    rows, cols = tl.shape

    i_top = int(corner_i)   # fila del borde superior visual
    j0    = int(corner_j)   # columna del borde izquierdo
    i0    = i_top - roi_h + 1  # fila de inicio del slice

    # --- Validaciones ---
    if i_top < 0 or i_top >= rows:
        raise ValueError(
            f"Fila superior ({i_top}) fuera del plano (0..{rows-1}). "
            f"Revisa la coordenada Z de ROI_CORNER en config.py."
        )
    if j0 < 0 or j0 >= cols:
        raise ValueError(
            f"Columna izquierda ({j0}) fuera del plano (0..{cols-1}). "
            f"Revisa la coordenada X de ROI_CORNER en config.py."
        )
    if i0 < 0:
        raise ValueError(
            f"La ROI excede el plano por abajo: fila superior={i_top}, "
            f"ROI_HEIGHT={roi_h} -> fila inicio={i0} (< 0). "
            f"Sube la Z de ROI_CORNER o reduce ROI_HEIGHT en config.py."
        )
    if j0 + roi_w > cols:
        raise ValueError(
            f"La ROI excede el plano por la derecha: col izq={j0}, "
            f"ROI_WIDTH={roi_w}, columnas disponibles={cols}. "
            f"Sobran {j0 + roi_w - cols} px. "
            f"Reduce ROI_WIDTH o cambia la X de ROI_CORNER en config.py."
        )

    roi = tl[i0:i_top + 1, j0:j0 + roi_w]
    return roi, i0, j0


def compute_roi_extent(X_raw, Z_raw, i0, j0, roi_h, roi_w):
    """Calcula extension fisica [x_left, x_right, z_bottom, z_top]
    de una ROI dada por su esquina pixel (i0, j0) en tl.T.
    """
    x_axis = X_raw[:, 0]
    z_axis = _get_z_axis(Z_raw)

    x_left   = float(x_axis[j0])
    x_right  = float(x_axis[min(j0 + roi_w - 1, len(x_axis) - 1)])
    z_bottom = float(z_axis[i0])
    z_top    = float(z_axis[min(i0 + roi_h - 1, len(z_axis) - 1)])

    return [x_left, x_right, z_bottom, z_top]


def extract_multiple_rois(tl, roi_h, roi_w, num_rois=5,
                          mode="center_max", corner_pixel=(0, 0)):
    """Extrae ROIs de un plano segun el modo elegido.

    mode="center_max"  -> 1 ROI en el maximo + N-1 en zonas de alto TL.
    mode="corner_fixed" -> 1 ROI con la esquina superior-izquierda en
                          *corner_pixel* (indices de pixel ya convertidos).
                          (num_rois se ignora; siempre devuelve 1 ROI).

    Returns:
        (rois, origins) -- listas paralelas; origins = [(i0, j0), ...]
    """
    rois = []
    origins = []
    rows, cols = tl.shape

    if mode == "corner_fixed":
        roi, i0, j0 = extract_roi_at_corner(tl, corner_pixel[0],
                                            corner_pixel[1], roi_h, roi_w)
        rois.append(roi)
        origins.append((i0, j0))
        return rois, origins

    # --- mode == "center_max" (comportamiento original) ---
    half_h = roi_h // 2
    half_w = roi_w // 2
    # ROI 1: maximo absoluto
    mi, mj = np.unravel_index(np.argmax(tl), tl.shape)
    roi, i0, j0 = extract_roi_at(tl, mi, mj, roi_h, roi_w)
    rois.append(roi)
    origins.append((i0, j0))

    # ROIs adicionales
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


# ══════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════

def _extract_angle(fname):
    """Extrae el ángulo del nombre del fichero (ej. PlaneAngle-104.47 → -104.47)."""
    m = re.search(r'PlaneAngle(-?\d+(?:\.\d+)?)', fname)
    return float(m.group(1)) if m else 0.0


from config import USE_BLOCK_VARIABLES


def _load_rois_from_folder(folder, roi_h, roi_w, rois_per_plane,
                            roi_mode, roi_corner, verbose_bounds=True):
    """Nucleo comun: carga .mat de *folder* y extrae ROIs.

    Returns:
        (np.array rois, list angles, list extents).
        Array vacio si no hay .mat validos.
    """
    all_rois, all_angles, all_extents = [], [], []

    if roi_mode == "corner_fixed":
        print(f"  Modo ROI: corner_fixed  |  esquina fisica (X, Z) = {roi_corner}")

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".mat") or "PlaneAngle" not in fname:
            continue
        fpath = os.path.join(folder, fname)
        try:
            with h5py.File(fpath, "r") as f:
                tl_keys = ['tl_block', 'tl', 'TL', 'tL', 'tl_smooth'] if USE_BLOCK_VARIABLES else ['tl', 'TL', 'tL', 'tl_smooth', 'tl_block']
                tl_source = None
                tl = None
                for candidate in tl_keys:
                    if candidate in f:
                        tl_source = candidate
                        tl = _sanitize_tl_array(f[candidate][:].T)
                        break
                if tl is None:
                    raise KeyError(f"Ninguna de las claves {tl_keys} encontrada en el .mat")

                # Prefer full-resolution coordinate grids when they match the loaded TL.
                # This mirrors the hybrid loader and avoids using block axes for smoothed/full TL.
                X_raw = None
                Z_raw = None
                if 'R' in f and 'Z' in f:
                    r_full = f['R'][:]
                    z_full = f['Z'][:]
                    if r_full.shape == tl.shape and z_full.shape == tl.shape:
                        X_raw = r_full
                        Z_raw = z_full

                if X_raw is None or Z_raw is None:
                    if tl_source == 'tl_block':
                        X_raw = _safe_get_dataset(f, ['R_block', 'X', 'x', 'R', 'r'])
                        Z_raw = _safe_get_dataset(f, ['Z_block', 'Z', 'z'])
                    else:
                        X_raw = _safe_get_dataset(f, ['X', 'x', 'R', 'r', 'R_block'])
                        Z_raw = _safe_get_dataset(f, ['Z', 'z', 'Z_block'])
        except Exception as e:
            print(f"  [!] Error con {fname}: {e}")
            continue

        angle = _extract_angle(fname)

        # Convertir tamaños físicos (m) de ROI a píxeles para este plano
        roi_h_px, roi_w_px = phys_size_to_pixels(X_raw, Z_raw, roi_h, roi_w)

        # Clampear a la dimensión del plano
        rows, cols = tl.shape
        roi_h_px = min(roi_h_px, rows)
        roi_w_px = min(roi_w_px, cols)

        corner_pixel = (0, 0)
        if roi_mode == "corner_fixed":
            x_phys, z_phys = roi_corner
            row_px, col_px = phys_to_pixel(X_raw, Z_raw, x_phys, z_phys)

            # Clampear la esquina para que la ROI quepa completa dentro del plano.
            # Esto copia el comportamiento robusto del híbrido y evita que corner_fixed
            # falle cuando la conversión física cae demasiado cerca del borde superior/izquierdo.
            max_top = max(roi_h_px - 1, 0)
            max_left = max(cols - roi_w_px, 0)
            adj_row_px = min(max(row_px, max_top), rows - 1)
            adj_col_px = min(max(col_px, 0), max_left)
            if verbose_bounds and (adj_row_px != row_px or adj_col_px != col_px):
                print(
                    f"  [i] {fname}: esquina ajustada para que la ROI quepa "
                    f"-> pixel ({adj_row_px}, {adj_col_px})"
                )
            row_px, col_px = adj_row_px, adj_col_px

            # Informar conversión / ajuste
            if verbose_bounds:
                x_min, x_max = float(X_raw.min()), float(X_raw.max())
                z_min, z_max = float(Z_raw.min()), float(Z_raw.max())
                clamped = False
                if x_phys < x_min or x_phys > x_max:
                    print(f"  [!] {fname}: X={x_phys} fuera de rango "
                          f"[{x_min:.1f}, {x_max:.1f}] -> se ajusta al borde")
                    clamped = True
                if z_phys < z_min or z_phys > z_max:
                    print(f"  [!] {fname}: Z={z_phys} fuera de rango "
                          f"[{z_min:.1f}, {z_max:.1f}] -> se ajusta al borde")
                    clamped = True
                if not clamped:
                    print(f"  [i] {fname}: esquina física ({x_phys}, {z_phys}) "
                          f"-> pixel ({row_px}, {col_px})")

            corner_pixel = (row_px, col_px)

        # Extraer ROIs usando tamaños en píxeles calculados para este plano
        plane_rois, plane_origins = extract_multiple_rois(
            tl, roi_h_px, roi_w_px, rois_per_plane,
            mode=roi_mode, corner_pixel=corner_pixel,
        )
        for roi, (oi, oj) in zip(plane_rois, plane_origins):
            extent = compute_roi_extent(X_raw, Z_raw, oi, oj, roi_h_px, roi_w_px)
            all_rois.append(roi.astype(np.float32))
            all_angles.append(angle)
            all_extents.append(extent)
        print(f"  [OK] {fname} -> {len(plane_rois)} ROIs  (angulo={angle:.2f})")

    if not all_rois:
        return np.array([], dtype=np.float32), [], []
    return np.array(all_rois, dtype=np.float32), all_angles, all_extents


def load_all_rois(folder, roi_h, roi_w, rois_per_plane=5,
                  roi_mode="center_max", roi_corner=(0, 0)):
    """Carga todos los .mat de entrenamiento y extrae ROIs."""
    return _load_rois_from_folder(
        folder, roi_h, roi_w, rois_per_plane,
        roi_mode, roi_corner, verbose_bounds=True,
    )


def load_validation_rois(folder, roi_h, roi_w,
                         roi_mode="center_max", roi_corner=(0, 0)):
    """Carga .mat de validacion (1 ROI por plano).
    Devuelve (None, [], []) si la carpeta no existe o esta vacia.
    """
    if not os.path.isdir(folder):
        return None, [], []

    print("\n" + "=" * 60)
    print("  CARGANDO PLANOS DE VALIDACION")
    print("=" * 60)

    rois, angles, extents = _load_rois_from_folder(
        folder, roi_h, roi_w, 1,
        roi_mode, roi_corner, verbose_bounds=False,
    )
    if len(rois) == 0:
        return None, [], []
    return rois, angles, extents


# ══════════════════════════════════════════════════════════════════════
#  NORMALIZACIÓN / DESNORMALIZACIÓN
# ══════════════════════════════════════════════════════════════════════

class Normalizer:
    """Almacena los rangos de TL y ángulos y ofrece métodos de
    normalización / desnormalización para ambos."""

    def __init__(self, rois, roi_angles):
        # Validar que se recibieron ROIs
        if rois is None or getattr(rois, 'size', 0) == 0:
            raise ValueError(
                "No se encontraron ROIs: revisa la carpeta de datos y los parámetros de ROI. "
                "Asegúrate de que existan ficheros .mat con 'tl','X' y 'Z', que contengan 'PlaneAngle' en el nombre, "
                "y que ROI_HEIGHT/ROI_WIDTH o ROI_CORNER permitan extraer al menos una ROI por plano."
            )

        # usar nan-aware para evitar que NaNs rompan los rangos de normalización
        self.tl_min = float(np.nanmin(rois))
        self.tl_max = float(np.nanmax(rois))

        angles_array = np.array(roi_angles, dtype=np.float32)
        if angles_array.size == 0:
            raise ValueError("No se encontraron ángulos asociados a las ROIs (roi_angles vacía).")
        self.angle_min = float(angles_array.min())
        self.angle_max = float(angles_array.max())

    # --- TL ---
    def normalize_tl(self, x):
        return (x - self.tl_min) / (self.tl_max - self.tl_min + 1e-8)

    def denormalize_tl(self, x_norm):
        return x_norm * (self.tl_max - self.tl_min) + self.tl_min

    # --- Ángulos ---
    def normalize_angle(self, angle_deg):
        return (angle_deg - self.angle_min) / (self.angle_max - self.angle_min + 1e-8)

    def normalize_angles(self, angles_array):
        a = np.array(angles_array, dtype=np.float32)
        return (a - self.angle_min) / (self.angle_max - self.angle_min + 1e-8)

    @classmethod
    def from_checkpoint(cls, checkpoint):
        """Reconstruye el Normalizer a partir de los params guardados en el .pt,
        sin necesidad de recargar los datos originales."""
        obj = cls.__new__(cls)
        obj.tl_min = checkpoint["tl_min"]
        obj.tl_max = checkpoint["tl_max"]
        obj.angle_min = checkpoint["angle_min"]
        obj.angle_max = checkpoint["angle_max"]
        return obj

    def state_dict(self):
        """Devuelve un dict serializable para guardar junto al modelo."""
        return {
            "tl_min": self.tl_min,
            "tl_max": self.tl_max,
            "angle_min": self.angle_min,
            "angle_max": self.angle_max,
        }


# ══════════════════════════════════════════════════════════════════════
#  DATA AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

def augment(data, angles):
    """Augmenta imágenes y replica los ángulos correspondientes.

    Solo se aplican transformaciones físicamente coherentes:
    ruido gaussiano y escalado de contraste.
    Factor de aumento: ×4 (original + 2 ruidos + 1 contraste).
    """
    aug_imgs = list(data)
    aug_angs = list(angles)

    for img, ang in zip(data, angles):
        # Ruido gaussiano
        for sigma in [0.01, 0.03]:
            noisy = img + np.random.normal(0, sigma, img.shape).astype(np.float32)
            aug_imgs.append(np.clip(noisy, 0, 1))
            aug_angs.append(ang)
        # Escalado de contraste
        s = np.random.uniform(0.9, 1.1)
        aug_imgs.append(
            np.clip(img * s + np.random.uniform(-0.05, 0.05), 0, 1).astype(np.float32)
        )
        aug_angs.append(ang)
    return np.array(aug_imgs, dtype=np.float32), np.array(aug_angs, dtype=np.float32)


def generate_inpainting_known_masks(batch_size, height, width,
                                    min_holes=1, max_holes=4,
                                    min_hole_ratio=0.08, max_hole_ratio=0.30,
                                    keep_full_prob=0.0):
    """Genera mascaras known_mask para inpainting.

    known_mask = 1 en zona conocida y 0 en huecos a rellenar.
    Devuelve array float32 con shape (B, 1, H, W).
    """
    masks = np.ones((batch_size, 1, height, width), dtype=np.float32)

    min_holes = int(max(1, min_holes))
    max_holes = int(max(min_holes, max_holes))
    min_h = max(1, int(round(height * float(min_hole_ratio))))
    max_h = max(min_h, int(round(height * float(max_hole_ratio))))
    min_w = max(1, int(round(width * float(min_hole_ratio))))
    max_w = max(min_w, int(round(width * float(max_hole_ratio))))

    for b in range(batch_size):
        if np.random.rand() < keep_full_prob:
            continue

        n_holes = np.random.randint(min_holes, max_holes + 1)
        for _ in range(n_holes):
            hole_h = np.random.randint(min_h, max_h + 1)
            hole_w = np.random.randint(min_w, max_w + 1)

            i0 = np.random.randint(0, max(1, height - hole_h + 1))
            j0 = np.random.randint(0, max(1, width - hole_w + 1))
            masks[b, 0, i0:i0 + hole_h, j0:j0 + hole_w] = 0.0

    return masks


def apply_inpainting_mask(batch_imgs, known_masks, fill_value=0.0):
    """Aplica una known_mask a batch de imagenes normalizadas.

    batch_imgs:  np.ndarray (B, 1, H, W)
    known_masks: np.ndarray (B, 1, H, W)
    """
    known_masks = known_masks.astype(np.float32)
    return batch_imgs * known_masks + float(fill_value) * (1.0 - known_masks)


# ══════════════════════════════════════════════════════════════════════
#  UTILIDADES DE E/S (.mat HDF5)
# ══════════════════════════════════════════════════════════════════════

def create_xyz_grid(shape, x_range=None, z_range=None):
    """Crea grids X/R, Y, Z con la misma convencion que los .mat originales.

    Convencion raw (sin transponer):
      - eje 0 (filas)    -> X/R
      - eje 1 (columnas) -> Z

    Parameters
    ----------
    shape : tuple[int, int]
        Shape de la matriz tl a guardar (raw, antes de cargar con .T).
    x_range, z_range : tuple[float, float] | None
        Rango fisico [min, max] para X/R y Z. Si es None, usa indices de pixel.
    """
    rows, cols = shape

    if x_range is None:
        x_axis = np.arange(rows, dtype=np.float64)
    else:
        x_axis = np.linspace(float(x_range[0]), float(x_range[1]), rows, dtype=np.float64)

    if z_range is None:
        z_axis = np.arange(cols, dtype=np.float64)
    else:
        z_axis = np.linspace(float(z_range[0]), float(z_range[1]), cols, dtype=np.float64)

    # X/R debe variar por filas y Z por columnas (igual que en los .mat de entrada).
    X = np.repeat(x_axis[:, np.newaxis], cols, axis=1)
    Z = np.repeat(z_axis[np.newaxis, :], rows, axis=0)
    Y = np.zeros_like(X)
    return X, Y, Z


def save_mat(path, tl_array, extent=None):
    """Guarda un .mat HDF5 con ejes coherentes con la convencion original.

    tl_array entra en convencion interna (filas=Z, cols=X) y se transpone al
    formato raw (filas=X/R, cols=Z) para exportar.

    Parameters
    ----------
    path : str
        Ruta de salida.
    tl_array : np.ndarray
        Plano TL en convencion interna (Z, X).
    extent : list[float] | tuple[float, float, float, float] | None
        [x_left, x_right, z_bottom, z_top]. Si se pasa, guarda ejes fisicos.
        Si es None, guarda ejes en indices de pixel.
    """
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


# ══════════════════════════════════════════════════════════════════════
#  UTILIDAD PARA FIGSIZE ADAPTATIVO
# ══════════════════════════════════════════════════════════════════════

def adaptive_figsize(data_h, data_w, n_rows, n_cols,
                     base=3.0, max_aspect=4.0, extra_w=1.2):
    """Calcula un figsize proporcional al aspect ratio de los datos.

    Con ROIs cuadradas la figura sale cuadrada; con ROIs rectangulares
    (ej. 800×50) la figura refleja la proporción, recortando a
    max_aspect para evitar figuras gigantes.

    Parameters
    ----------
    data_h, data_w : int
        Altura y anchura de cada imagen (datos, no píxeles de pantalla).
    n_rows, n_cols : int
        Distribución de subplots.
    base : float
        Tamaño base (pulgadas) del eje más corto de cada subplot.
    max_aspect : float
        Ratio máximo permitido (se clampea para evitar figuras enormes).
    extra_w : float
        Margen extra por columna (colorbars, padding).

    Returns
    -------
    (fig_width, fig_height) : tuple[float, float]
    """
    ar = data_h / data_w
    ar = max(1.0 / max_aspect, min(ar, max_aspect))

    if ar >= 1.0:            # imagen alta (más filas que columnas)
        subplot_w = base
        subplot_h = base * ar
    else:                    # imagen ancha
        subplot_h = base
        subplot_w = base / ar

    fig_w = (subplot_w + extra_w) * n_cols
    fig_h = subplot_h * n_rows + 1.5   # +1.5 para suptitle
    return (fig_w, fig_h)


# ══════════════════════════════════════════════════════════════════════
#  GENERACIÓN DESDE SEMILLA
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
#  MÉTRICAS DE ERROR
# ══════════════════════════════════════════════════════════════════════

def compute_error_metrics(actual, predicted):
    """
    Calcula múltiples métricas de error entre actual y predicted.

    Parameters
    ----------
    actual      : np.ndarray, valores reales.
    predicted   : np.ndarray, valores predichos (mismo shape que actual).

    Returns
    -------
    dict con claves: 'mae', 'rmse', 'mape', 'max_error'
        mae       : Mean Absolute Error (promedio de |actual - pred|)
        rmse      : Root Mean Square Error (sqrt(mean((actual - pred)^2)))
        mape      : Mean Absolute Percentage Error (100 * mean(|actual - pred| / |actual|))
                    Nota: se evita división por cero ignorando píxeles donde |actual| < 1e-6
        max_error : Máximo error absoluto
    """
    diff = np.abs(actual - predicted)
    
    # MAE
    mae = np.mean(diff)
    
    # RMSE
    rmse = np.sqrt(np.mean(diff ** 2))
    
    # MAPE (ignorar píxeles con |actual| < 1e-6 para evitar división por cero)
    mask = np.abs(actual) >= 1e-6
    if np.any(mask):
        mape = 100.0 * np.mean(np.abs(actual[mask] - predicted[mask]) / np.abs(actual[mask]))
    else:
        mape = np.inf  # todos los píxeles son ~0
    
    # Max error
    max_error = np.max(diff)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'max_error': max_error,
    }


def generate_from_seed(model, seed_tensor, target_angle, norm, noise_std, device):
    """
    Genera un plano desde semilla usando encode/decode y FiLM condicional.

    Parameters
    ----------
    model        : ConditionalUNetAE (ya en device, eval mode).
    seed_tensor  : Tensor (1, 1, H, W) ya en device.
    target_angle : float, angulo objetivo en grados.
    norm         : Normalizer.
    noise_std    : float.
    device       : torch.device.

    Returns
    -------
    gen_tl : np.ndarray 2-D con valores TL desnormalizados.
    """
    # --- Normalizar ángulo y crear tensor condicional ---
    cond = torch.tensor(
        [[norm.normalize_angles(target_angle)]],
        dtype=torch.float32, device=device
    )

    # --- Encode con FiLM ---
    e1, e2, e3, b = model.encode(seed_tensor, cond)

    # --- Agregar ruido proporcional a cada nivel ---
    e1_noisy = e1 + torch.randn_like(e1) * (noise_std * 0.05)
    e2_noisy = e2 + torch.randn_like(e2) * (noise_std * 0.1)
    e3_noisy = e3 + torch.randn_like(e3) * (noise_std * 0.3)
    b_noisy  = b  + torch.randn_like(b)  * noise_std

    # --- Decode con FiLM ---
    gen_norm = model.decode(e1_noisy, e2_noisy, e3_noisy, b_noisy, cond)

    # --- Clip y desnormalizar ---
    gen_np = np.clip(gen_norm.cpu().squeeze().numpy(), 0, 1)
    gen_tl = norm.denormalize_tl(gen_np)

    return gen_tl

# ══════════════════════════════════════════════════════════════════════
#  PIPELINE DE INFERENCIA COMPARTIDO
# ══════════════════════════════════════════════════════════════════════

def load_inference_pipeline(model_path, data_folder, roi_h, roi_w,
                            rois_per_plane,
                            roi_mode="center_max", roi_corner=(0, 0)):
    """Carga modelo entrenado y datos para inferencia (generate / analysis).

    Usa los parametros de normalizacion guardados en el checkpoint para
    garantizar consistencia con el entrenamiento.

    Returns:
        (device, model, rois_norm, roi_angles, angles_norm, norm, roi_extents)
    """
    from model import ConditionalUNetAE
    from config import verify_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- Modelo ---
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    verify_config(checkpoint)

    model = ConditionalUNetAE().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Modelo cargado desde: {model_path}")

    # --- Normalizer desde checkpoint (consistente con entrenamiento) ---
    norm = Normalizer.from_checkpoint(checkpoint)

    # --- Datos ---
    print("Cargando datos originales...")
    rois, roi_angles, roi_extents = load_all_rois(
        data_folder, roi_h, roi_w,
        rois_per_plane,
        roi_mode=roi_mode, roi_corner=roi_corner,
    )
    rois_norm = norm.normalize_tl(rois)
    angles_norm = norm.normalize_angles(roi_angles)

    print(f"ROIs: {len(rois)}  |  Forma: {rois.shape[1]}x{rois.shape[2]}")
    print(f"Rango TL: [{norm.tl_min:.2f}, {norm.tl_max:.2f}]")
    print(f"Rango angulos: [{norm.angle_min:.2f}, {norm.angle_max:.2f}]")

    return device, model, rois_norm, roi_angles, angles_norm, norm, roi_extents
