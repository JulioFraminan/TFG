#!/usr/bin/env python3
"""Diagnóstico: convertir dos esquinas físicas a índices de píxel en cada .mat

Usa la esquina superior-izquierda definida en `config.ROI_CORNER` y la
esquina inferior-derecha fija (-1100, -80) (X, Z).

Para cada .mat en `config.DATA_FOLDER` imprime:
 - shapes de X, Z, tl
 - índices de fila/col para cada esquina (en la convención usada por tl = f['tl'][:].T)
 - píxeles cubiertos (height_px, width_px)
 - extent físico calculado a partir de esos índices
 - comparación con ROI_HEIGHT/ROI_WIDTH de `config`
 - avisos si la caja sale fuera de rango

No modifica ningún archivo.
"""
import os
import sys
import h5py
import numpy as np

# Importar configuración mínima (intentamos import normal y si falla cargar por ruta)
try:
    from config import DATA_FOLDER, ROI_CORNER, ROI_HEIGHT, ROI_WIDTH
except Exception:
    # Intentar cargar config.py usando su ruta absoluta relativa al script
    import importlib.util
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, 'config.py')
    if not os.path.isfile(config_path):
        print(f"No se encontró config.py en {config_path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location('config', config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    DATA_FOLDER = cfg.DATA_FOLDER
    ROI_CORNER = cfg.ROI_CORNER
    ROI_HEIGHT = cfg.ROI_HEIGHT
    ROI_WIDTH = cfg.ROI_WIDTH

# esquina inferior-derecha pedida por el usuario
CORNER_B = (-1100.0, -80.0)  # (X, Z)


def phys_to_pixel_indices(X_raw, Z_raw, x_phys, z_phys):
    """Convierte (x_phys, z_phys) a índices (row, col) según convención repo.

    - x_axis = X_raw[:, 0]  -> columna en tl.T
    - z_axis = Z_raw[0, :]  -> fila en tl.T
    """
    x_axis = X_raw[:, 0]
    z_axis = Z_raw[0, :]
    col = int(np.argmin(np.abs(x_axis - x_phys)))
    row = int(np.argmin(np.abs(z_axis - z_phys)))
    return row, col


def compute_extent_from_indices(X_raw, Z_raw, i0, i1, j0, j1):
    x_axis = X_raw[:, 0]
    z_axis = Z_raw[0, :]
    x_left = float(x_axis[j0])
    x_right = float(x_axis[min(j1, len(x_axis)-1)])
    z_bottom = float(z_axis[i0])
    z_top = float(z_axis[min(i1, len(z_axis)-1)])
    return [x_left, x_right, z_bottom, z_top]


def analyze_mat(fpath):
    with h5py.File(fpath, 'r') as f:
        # cargar en la misma convención que el repositorio usa para extracción
        tl = f['tl'][:].T
        X_raw = f['X'][:]
        Z_raw = f['Z'][:]

    rows, cols = tl.shape
    x_min, x_max = float(X_raw.min()), float(X_raw.max())
    z_min, z_max = float(Z_raw.min()), float(Z_raw.max())

    a_x, a_z = ROI_CORNER
    b_x, b_z = CORNER_B

    # Normalizar orden: queremos x_left < x_right y z_bottom < z_top
    x_left_phys = min(a_x, b_x)
    x_right_phys = max(a_x, b_x)
    z_bottom_phys = min(a_z, b_z)
    z_top_phys = max(a_z, b_z)

    # Índices
    i0, j0 = phys_to_pixel_indices(X_raw, Z_raw, x_left_phys, z_bottom_phys)
    i1, j1 = phys_to_pixel_indices(X_raw, Z_raw, x_right_phys, z_top_phys)

    # Asegurar orden en índices para slicing tl[i0:i1+1, j0:j1+1]
    i_start, i_end = min(i0, i1), max(i0, i1)
    j_start, j_end = min(j0, j1), max(j0, j1)

    height_px = i_end - i_start + 1
    width_px = j_end - j_start + 1

    extent_px = compute_extent_from_indices(X_raw, Z_raw, i_start, i_end, j_start, j_end)

    # Validaciones
    oob = False
    msgs = []
    if x_left_phys < x_min or x_right_phys > x_max:
        msgs.append(f"X fuera de rango input [{x_min:.2f}, {x_max:.2f}]")
        oob = True
    if z_bottom_phys < z_min or z_top_phys > z_max:
        msgs.append(f"Z fuera de rango input [{z_min:.2f}, {z_max:.2f}]")
        oob = True
    if i_start < 0 or i_end >= rows or j_start < 0 or j_end >= cols:
        msgs.append(f"Indices fuera de bounds tl shape {tl.shape}")
        oob = True

    return {
        'rows_cols': (rows, cols),
        'x_range': (x_min, x_max),
        'z_range': (z_min, z_max),
        'phys_requested': ((x_left_phys, z_top_phys), (x_right_phys, z_bottom_phys)),
        'indices': ((i_start, j_start), (i_end, j_end)),
        'pixels': (height_px, width_px),
        'extent_from_indices': extent_px,
        'oob': oob,
        'messages': msgs,
    }


def main():
    folder = DATA_FOLDER
    if not os.path.isdir(folder):
        print(f"Carpeta de datos no existe: {folder}")
        return

    mats = sorted([p for p in os.listdir(folder) if p.endswith('.mat') and 'PlaneAngle' in p])
    if not mats:
        print("No hay .mat con 'PlaneAngle' en la carpeta de datos.")
        return

    print(f"Usando esquina A (config.ROI_CORNER) = {ROI_CORNER}")
    print(f"Usando esquina B (fijada) = {CORNER_B}\n")

    for fname in mats:
        fpath = os.path.join(folder, fname)
        try:
            info = analyze_mat(fpath)
        except Exception as e:
            print(f"{fname}: ERROR al leer -> {e}")
            continue

        rows, cols = info['rows_cols']
        print(f"{fname}  |  tl.shape={rows}x{cols}")
        print(f"  X range: {info['x_range'][0]:.2f} .. {info['x_range'][1]:.2f}")
        print(f"  Z range: {info['z_range'][0]:.2f} .. {info['z_range'][1]:.2f}")
        (i0, j0), (i1, j1) = info['indices']
        print(f"  Indices corners (i0,j0)={(i0,j0)}  (i1,j1)={(i1,j1)}")
        hpx, wpx = info['pixels']
        print(f"  Pixels cubiertos: height={hpx} px, width={wpx} px")
        print(f"  Extent calculado desde índices: [x_left={info['extent_from_indices'][0]:.2f}, x_right={info['extent_from_indices'][1]:.2f}, z_bottom={info['extent_from_indices'][2]:.2f}, z_top={info['extent_from_indices'][3]:.2f}]")
        print(f"  Modelo espera ROI_HEIGHT x ROI_WIDTH = {ROI_HEIGHT} x {ROI_WIDTH} (px)")
        # sugerencia sencilla
        if (hpx, wpx) != (ROI_HEIGHT, ROI_WIDTH):
            print(f"  -> NOTA: el bloque físico no coincide con la forma del modelo; habría que remuestrear a {ROI_HEIGHT}x{ROI_WIDTH} px")
        if info['oob']:
            print("  [!] Fuera de rango:")
            for m in info['messages']:
                print("    -", m)
        print("\n")


if __name__ == '__main__':
    main()
