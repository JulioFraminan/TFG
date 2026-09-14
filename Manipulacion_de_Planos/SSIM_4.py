

import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries


# =====================================================================
# CONFIGURACIÓN Y ORDEN DE MODELOS
# =====================================================================

PLANE_GRID_SHAPE = (1920, 504)

# Rutas reales del workspace para cada método / Ground Truth.
METODOS_SUBFOLDERS = {
    "Original (Ground Truth)": "/home/j.framinan/TFG_repo/Interpolation/input/validation",
    "Linear Interpolation": "/home/j.framinan/TFG_repo/Interpolation/output/validation/MAT",
    "Fourier Neural Operator (FNO)": "/home/j.framinan/TFG_repo/Intento_FNO/optuna_runs/optuna_fno_epochs_mape_30/trial_0003/output/validation/MAT",
    "Physics-Informed Neural Network (PINN)": "/home/j.framinan/TFG_repo/pinn_modular/optuna_runs/pinn_optuna_trial_049/validation/MAT",
    "Transformer-Based Conditional Diffusion Model": "/home/j.framinan/TFG_repo/Transformers_2/results/intento_mat/dit_gaussian/30/run_0001_dit-dit-sp8-linear_504x1920_gaussian_b4x1_lr1e-3/validation/MAT",
    "Conditional UNet Autoencoder": "/home/j.framinan/TFG_repo/unet_ae_modular_inpainting/results/optuna_unet_ae_epochs_only_30/trial_0023/output/validation/MAT",
    "Hybrid Physics-Informed UNet Architecture": "/home/j.framinan/TFG_repo/unet_hibrido/results/optuna_unet_hybrid_epochs_only_30/trial_0034_corner_fixed_roi504x1920_corner10p00_m5p00/output/validation/MAT",
    "Dual Channel U-net": "/home/j.framinan/TFG_repo/unet_dual_channel/results/optuna_dual_epochs_only_30/trial_0027/output/validation/MAT",
}

# El orden estricto solicitado para tablas y gráficas
ORDEN_MODELOS = [
    "Linear Interpolation",
    "Fourier Neural Operator (FNO)",
    "Physics-Informed Neural Network (PINN)",
    "Transformer-Based Conditional Diffusion Model",
    "Conditional UNet Autoencoder",
    "Hybrid Physics-Informed UNet Architecture",
    "Dual Channel U-net"
]

NOMBRES_CORTOS = {
    "Linear Interpolation": "Linear",
    "Fourier Neural Operator (FNO)": "FNO",
    "Physics-Informed Neural Network (PINN)": "PINN",
    "Transformer-Based Conditional Diffusion Model": "Transformer",
    "Conditional UNet Autoencoder": "Cond. UNet",
    "Hybrid Physics-Informed UNet Architecture": "Hybrid UNet",
    "Dual Channel U-net": "Dual U-Net"
}

# Parámetros fijos
TOLERANCE_PX = 2          
RAY_THRESHOLD = -50.0     
GT_ROI_CORNER = (10.0, -5.0)  # (X, Z) esquina superior izquierda de la ROI en coordenadas físicas.

ANGLE_REGEX = re.compile(r"([-+]?\d+(?:\.\d+)?)")

# =====================================================================
# FUNCIONES DE MÉTRICAS
# =====================================================================

def standardize_shape(array, nx, ny):
    if array.shape == (nx, ny): return array
    if array.shape == (ny, nx): return array.T
    raise ValueError(f"Shape inválido {array.shape}.")


def _pick_axis(grid):
    """Devuelve el eje 1D que realmente varía en una rejilla 2D."""
    a0 = grid[:, 0].flatten()
    a1 = grid[0, :].flatten()
    return a0 if np.ptp(a0) >= np.ptp(a1) else a1


def _load_gt_tl_roi(file_path, nx, ny, corner_x, corner_z):
    """Carga GT y lo recorta al mismo ROI (orientación y tamaño) que las predicciones."""
    with h5py.File(file_path, "r") as f:
        tl_raw = np.asarray(f["tl"][:], dtype=np.float32)

        # Si ya viene en tamaño ROI, usar normalización estándar.
        if tl_raw.shape == (nx, ny) or tl_raw.shape == (ny, nx):
            return standardize_shape(tl_raw, nx, ny)

        if "R" not in f or "Z" not in f:
            raise KeyError(
                f"GT sin rejillas R/Z para recorte ROI en {os.path.basename(file_path)}"
            )

        r_raw = np.asarray(f["R"][:], dtype=np.float32)
        z_raw = np.asarray(f["Z"][:], dtype=np.float32)

    # En tl_raw.T: filas = Z, columnas = X (misma convención que pipelines de entrenamiento).
    tl_t = tl_raw.T
    x_axis = _pick_axis(r_raw)
    z_axis = _pick_axis(z_raw)

    col_left = int(np.argmin(np.abs(x_axis - float(corner_x))))
    row_top = int(np.argmin(np.abs(z_axis - float(corner_z))))

    roi_h = ny  # altura en eje Z
    roi_w = nx  # anchura en eje X
    row_start = row_top - roi_h + 1
    col_end = col_left + roi_w

    rows, cols = tl_t.shape
    if row_start < 0 or row_top >= rows or col_left < 0 or col_end > cols:
        raise ValueError(
            f"ROI GT fuera de rango en {os.path.basename(file_path)}. "
            f"row_start={row_start}, row_top={row_top}, col_left={col_left}, col_end={col_end}, "
            f"shape={tl_t.shape}."
        )

    gt_roi_t = tl_t[row_start:row_top + 1, col_left:col_end]  # (ny, nx)
    gt_roi = gt_roi_t.T  # -> (nx, ny)
    if gt_roi.shape != (nx, ny):
        raise ValueError(
            f"ROI GT con shape {gt_roi.shape}, esperado {(nx, ny)} en {os.path.basename(file_path)}"
        )
    return gt_roi


def extract_angle(path):
    filename = os.path.basename(path)
    match = ANGLE_REGEX.search(filename)
    if match is None:
        return None
    return float(match.group(1))


def angle_key_from_path(path):
    angle = extract_angle(path)
    if angle is None:
        return None
    # Redondeo suave para evitar problemas por representación flotante.
    return round(angle, 2)


def collect_mat_files(method_root):
    """Recoge .mat desde train/validation (con y sin carpeta output) y fallback plano."""
    patterns = [
        os.path.join(method_root, "train", "MAT", "*.[mM][aA][tT]"),
        os.path.join(method_root, "validation", "MAT", "*.[mM][aA][tT]"),
        os.path.join(method_root, "output", "train", "MAT", "*.[mM][aA][tT]"),
        os.path.join(method_root, "output", "validation", "MAT", "*.[mM][aA][tT]"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    if not files:
        files = glob.glob(os.path.join(method_root, "*.[mM][aA][tT]"))

    # Elimina duplicados conservando orden de descubrimiento.
    unique_files = list(dict.fromkeys(files))
    unique_files.sort(key=lambda p: (extract_angle(p) is None, extract_angle(p), os.path.getmtime(p)))
    return unique_files


def build_angle_map(file_list):
    """Mapea angulo -> archivo; si hay repetidos para un angulo, conserva el más reciente."""
    angle_to_file = {}
    for path in file_list:
        key = angle_key_from_path(path)
        if key is None:
            continue
        if key not in angle_to_file:
            angle_to_file[key] = path
            continue
        if os.path.getmtime(path) > os.path.getmtime(angle_to_file[key]):
            angle_to_file[key] = path
    return angle_to_file

def compute_ssim(gt, pred):
    drange = np.nanmax(gt) - np.nanmin(gt)
    # SSIM estándar de skimage
    return structural_similarity(gt, pred, data_range=drange, gaussian_weights=True)

def compute_spectral_error(gt, pred):
    log_gt = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gt))))
    log_pr = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(pred))))
    return np.mean(np.abs(log_gt - log_pr))

def compute_skeleton_f1(mask_gt, mask_pred, tolerance=TOLERANCE_PX):
    skel_gt = skeletonize(mask_gt)
    skel_pr = skeletonize(mask_pred)
    
    d_gt, d_pr = distance_transform_edt(~skel_gt), distance_transform_edt(~skel_pr)
    tp_recall = np.sum(skel_gt & (d_pr <= tolerance))
    tp_precision = np.sum(skel_pr & (d_gt <= tolerance))
    
    num_gt, num_pr = np.sum(skel_gt), np.sum(skel_pr)
    recall = tp_recall / num_gt if num_gt > 0 else 0
    precision = tp_precision / num_pr if num_pr > 0 else 0
    
    skel_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return skel_f1

def compute_hd95(mask_gt, mask_pred):
    bound_gt = find_boundaries(mask_gt, mode='inner')
    bound_pr = find_boundaries(mask_pred, mode='inner')
    
    if not np.any(bound_gt) or not np.any(bound_pr):
        return float('nan')
        
    d_gt, d_pr = distance_transform_edt(~bound_gt), distance_transform_edt(~bound_pr)
    
    dist_pr_to_gt = d_gt[bound_pr]
    dist_gt_to_pr = d_pr[bound_gt]
    
    hd95 = max(np.percentile(dist_pr_to_gt, 95), np.percentile(dist_gt_to_pr, 95))
    return hd95

# =====================================================================
# EJECUCIÓN DEL ANÁLISIS
# =====================================================================

nx, ny = PLANE_GRID_SHAPE
archivos_metodos = {}

print("🔍 Localizando archivos .mat...")
for nombre, method_root in METODOS_SUBFOLDERS.items():
    archivos = collect_mat_files(method_root)
    if archivos:
        archivos_metodos[nombre] = archivos
        print(f"   {nombre}: {len(archivos)} archivo(s)")
    else:
        print(f"⚠️ Advertencia: No se encontraron .mat para {nombre} en {method_root}")

if "Original (Ground Truth)" not in archivos_metodos:
    raise FileNotFoundError("No se encontraron archivos de Ground Truth para 'Original (Ground Truth)'.")

# --- Construir índice de GT por ángulo ---
gt_map = build_angle_map(archivos_metodos["Original (Ground Truth)"])
if len(gt_map) == 0:
    raise ValueError("No se pudieron extraer ángulos válidos de los archivos Ground Truth.")

resultados = {}

print(f"\n⚙️ Calculando las 4 métricas finales (Threshold = {RAY_THRESHOLD} dB)...")
for nombre in ORDEN_MODELOS:
    if nombre not in archivos_metodos:
        continue

    print(f"   Analizando {nombre}...")
    pred_map = build_angle_map(archivos_metodos[nombre])
    common_angles = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    if not common_angles:
        print(f"   ⚠️ {nombre}: sin ángulos comunes con Ground Truth. Se omite.")
        continue

    vals_ssim, vals_spec, vals_skel, vals_hd95 = [], [], [], []
    for ang in common_angles:
        tl_gt = _load_gt_tl_roi(
            gt_map[ang],
            nx,
            ny,
            corner_x=GT_ROI_CORNER[0],
            corner_z=GT_ROI_CORNER[1],
        )

        with h5py.File(pred_map[ang], "r") as f_pr:
            tl_pred = standardize_shape(np.asarray(f_pr["tl"][:], dtype=np.float32), nx, ny)

        mask_gt = (tl_gt > RAY_THRESHOLD) & np.isfinite(tl_gt)
        mask_pred = (tl_pred > RAY_THRESHOLD) & np.isfinite(tl_pred)

        vals_ssim.append(compute_ssim(tl_gt, tl_pred))
        vals_spec.append(compute_spectral_error(tl_gt, tl_pred))
        vals_skel.append(compute_skeleton_f1(mask_gt, mask_pred))
        vals_hd95.append(compute_hd95(mask_gt, mask_pred))

    resultados[nombre] = {
        "SSIM": float(np.nanmean(vals_ssim)),
        "Skel_F1": float(np.nanmean(vals_skel)),
        "HD95": float(np.nanmean(vals_hd95)),
        "Spec_Err": float(np.nanmean(vals_spec)),
        "N": len(common_angles),
    }

# =====================================================================
# IMPRESIÓN DE TABLA (ORDEN ESTRICTO)
# =====================================================================

print("\n" + "="*85)
print("🎯 RESULTADOS FINALES DE EVALUACIÓN")
print("-" * 85)
print(f"{'Modelo':<45} | {'N':<4} | {'SSIM ↑':<7} | {'Skel F1 ↑':<9} | {'HD95 (px) ↓':<11} | {'Spec Err ↓':<10}")
print("-" * 85)
for nombre in ORDEN_MODELOS:
    if nombre in resultados:
        r = resultados[nombre]
        print(f"{nombre:<45} | {r['N']:<4d} | {r['SSIM']:<7.4f} | {r['Skel_F1']:<9.4f} | {r['HD95']:<11.4f} | {r['Spec_Err']:<10.4f}")
print("="*85)

# =====================================================================
# GRÁFICAS: PANEL 2x2 FIJO Y NEUTRAL
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Evaluación de Reconstrucción de Modelos ML", fontsize=16, fontweight="bold")
axes = axes.flatten()

metricas_plot = [
    ("SSIM", "SSIM (↑ Mejor)"),
    ("Skel_F1", "Skeleton F1 (↑ Mejor)"),
    ("HD95", "HD95 [píxeles] (↓ Mejor)"),
    ("Spec_Err", "Spectral Error (↓ Mejor)")
]

labels_cortos = [NOMBRES_CORTOS[m] for m in ORDEN_MODELOS if m in resultados]

for i, (met_key, title) in enumerate(metricas_plot):
    ax = axes[i]
    vals = [resultados[m][met_key] for m in ORDEN_MODELOS if m in resultados]
    
    # Renderizamos en azul clásico, sin resaltar ningún modelo, en orden fijo
    ax.bar(labels_cortos, vals, color='royalblue', edgecolor='black', alpha=0.8)
    
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("final_4_metrics_comparison.png", dpi=200, bbox_inches="tight")
print("\n✅ Panel de 4 métricas guardado: final_4_metrics_comparison.png")
plt.show()