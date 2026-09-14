import glob
import os
import re
import time
import shutil

import h5py
import numpy as np
import pyvista as pv

#------ Limpieza de caché al inicio ------#
cache_dir = os.path.join(os.path.dirname(__file__), "__pycache__")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

#------ Configuración ------#


# Ground Truth caso 30: usar todos los .mat del árbol de Interpolation/input.
GROUND_TRUTH_FOLDER = r"/home/j.framinan/TFG_repo/Interpolation/input"
OUTPUT_VTS_PATH = r"/home/j.framinan/TFG_repo/input/VTS/ground_truth_case30.vts"

N_PLANES = None  # Pon None para cargar TODOS los .mat del nivel raíz, o un número en específico

# Configuración de exportación y visualización
SHOW_PYVISTA = False  # True: muestra ventana 3D, False: procesa pero no renderiza
EXPORT_VTS = True     # True: exporta a un grid estructurado .vts

# MUY IMPORTANTE: Dimensiones de cada plano SIN filtrar. (p. ej. Nx, Ny)
# El tamaño total del array original en el .mat debe ser igual a NX * NY
PLANE_GRID_SHAPE = (1920, 504)

# Umbral de intensidad:
ENABLE_TL_THRESHOLD = False
TL_MIN = -99.999
POINT_SIZE = 1.0

# ROI fija para comparar con el resto de métodos.
ROI_HEIGHT = 504
ROI_WIDTH = 1920
ROI_CORNER = (10, -5)

#------ Interpretar ángulo desde el nombre del archivo ------#


angle_regex = re.compile(r"PlaneAngle([-+]?\d+(?:\.\d+)?)")
fallback_number_regex = re.compile(r"([-+]?\d+(?:\.\d+)?)")

def extract_angle(path):
    filename = os.path.basename(path)

    # Caso original: archivos tipo PlaneAngleXX...
    match = angle_regex.search(filename)
    if match is not None:
        return float(match.group(1))

    # Fallback: cualquier número en el nombre (ej. val_generado_+31.46.mat)
    match = fallback_number_regex.search(filename)
    if match is not None:
        return float(match.group(1))

    # Si no hay número, no rotar.
    return 0.0


def collect_mat_files(root_folder):
    """Recoge todos los .mat del árbol, incluyendo validation/."""
    files = glob.glob(os.path.join(root_folder, "**", "*.[mM][aA][tT]"), recursive=True)
    if files:
        return files
    return []


def _safe_get_dataset(f, keys):
    for key in keys:
        if key in f:
            return f[key][:]
    raise KeyError(f"No se encontró ninguno de estos datasets: {keys}. Claves disponibles: {list(f.keys())}")


def _get_z_axis(z_raw):
    z_row = z_raw[0, :].flatten()
    z_col = z_raw[:, 0].flatten()
    if z_row.size > 1 and z_col.size <= 1:
        return z_row
    if z_col.size > 1 and z_row.size <= 1:
        return z_col
    return z_row if z_row.size >= z_col.size else z_col


def phys_to_pixel(r_raw, z_raw, x_phys, z_phys):
    x_axis = r_raw[:, 0]
    col_idx = int(np.argmin(np.abs(x_axis - x_phys)))
    z_axis = _get_z_axis(z_raw)
    row_idx = int(np.argmin(np.abs(z_axis - z_phys)))
    return row_idx, col_idx


def load_and_crop_gt_plane(path, roi_h, roi_w, roi_corner):
    with h5py.File(path, "r") as f:
        tl = np.asarray(_safe_get_dataset(f, ["tl", "TL", "tL", "tl_smooth", "tl_block"]), np.float32)
        r_raw = np.asarray(_safe_get_dataset(f, ["R", "R_block", "X", "x", "r"]), np.float32)
        z_raw = np.asarray(_safe_get_dataset(f, ["Z", "Z_block", "z"]), np.float32)

    tl_t = tl.T
    row_top, col_left = phys_to_pixel(r_raw, z_raw, roi_corner[0], roi_corner[1])
    row_start = row_top - roi_h + 1
    col_end = col_left + roi_w

    if row_start < 0 or col_left < 0 or row_top >= tl_t.shape[0] or col_end > tl_t.shape[1]:
        raise ValueError(
            f"ROI fuera de rango en {os.path.basename(path)}: "
            f"row_start={row_start}, row_top={row_top}, col_left={col_left}, col_end={col_end}, "
            f"shape={tl_t.shape}"
        )

    tl_crop = tl_t[row_start:row_top + 1, col_left:col_end].T.astype(np.float32)
    x_axis = r_raw[:, 0][col_left:col_end].astype(np.float32)
    z_axis = _get_z_axis(z_raw)[row_start:row_top + 1].astype(np.float32)

    if tl_crop.shape != (roi_w, roi_h):
        raise ValueError(
            f"Crop con shape {tl_crop.shape}, esperado {(roi_w, roi_h)} en {os.path.basename(path)}"
        )

    return tl_crop, x_axis, z_axis


def prepare_files(files, n_planes=None):
    """Ordena por timestamp y ángulo, y aplica recorte opcional por N_PLANES."""
    files_with_time = [(f, os.path.getmtime(f)) for f in files]
    files_with_time.sort(key=lambda x: (-x[1], extract_angle(x[0])))

    if n_planes is not None:
        files = [f[0] for f in files_with_time[:n_planes]]
    else:
        files = [f[0] for f in files_with_time]

    return sorted(files, key=extract_angle)


def export_case_vts(case_name, files, output_path):
    if len(files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .mat para el caso '{case_name}'.")

    angles = [extract_angle(p) for p in files]
    use_radians = max(map(abs, angles)) <= 6.5

    plotter = pv.Plotter() if SHOW_PYVISTA else None
    t0 = time.perf_counter()

    nx, ny = PLANE_GRID_SHAPE
    if EXPORT_VTS:
        nz = len(files)
        points_3d = np.empty((nx, ny, nz, 3), dtype=np.float32)
        tl_3d = np.empty((nx, ny, nz), dtype=np.float32)
    else:
        points_3d = None
        tl_3d = None

    _tl0, x_axis0, z_axis0 = load_and_crop_gt_plane(files[0], ROI_HEIGHT, ROI_WIDTH, ROI_CORNER)
    x0 = float(x_axis0[0])
    z0 = float(z_axis0[0])

    print(f"\nCaso: {case_name}")
    print(f"   Archivos totales: {len(files)}")

    for i, path in enumerate(files):

        angle_raw = extract_angle(path)
        t1 = time.perf_counter()

        tl, x_axis, z_axis = load_and_crop_gt_plane(path, ROI_HEIGHT, ROI_WIDTH, ROI_CORNER)

        x_axis = x_axis - x0
        z_axis = z_axis - z0

        X = np.tile(x_axis[:, None], (1, ROI_HEIGHT)).astype(np.float32)
        Z = np.tile(z_axis[None, :], (ROI_WIDTH, 1)).astype(np.float32)

        # ----- Guardar datos de CADA PLANO sin filtrar para Malla Estructurada (.vts) -----
        if EXPORT_VTS:
            if X.shape != (nx, ny) or Z.shape != (nx, ny) or tl.shape != (nx, ny):
                raise ValueError(
                    f"Shape inválido en {os.path.basename(path)}: X={X.shape}, Z={Z.shape}, tl={tl.shape}. "
                    f"Esperado {(nx, ny)} según PLANE_GRID_SHAPE."
                )

            angle_rad = angle_raw if use_radians else np.deg2rad(angle_raw)
            points_3d[:, :, i, 0] = X * np.cos(angle_rad)
            points_3d[:, :, i, 1] = X * np.sin(angle_rad)
            points_3d[:, :, i, 2] = Z
            tl_3d[:, :, i] = tl

        # ----- Filtrado (solo para PyVista Plotter local) -----
        keep = np.isfinite(tl)
        if ENABLE_TL_THRESHOLD:
            keep &= tl >= TL_MIN

        if not keep.any():
            continue

        Xk = X[keep]
        Zk = Z[keep]
        tlk = tl[keep]

        n_kept = Xk.size
        n_total = X.size

        # ----- Conversión cilíndrico -> cartesiano -----
        angle_rad = angle_raw if use_radians else np.deg2rad(angle_raw)

        r = Xk
        theta = angle_rad
        z = Zk

        points = np.empty((n_kept, 3), dtype=np.float32)
        points[:, 0] = r * np.cos(theta)
        points[:, 1] = r * np.sin(theta)
        points[:, 2] = z

        cloud = pv.PolyData(points)
        cloud["tl"] = tlk

        if SHOW_PYVISTA:
            plotter.add_mesh(
                cloud,
                scalars="tl",
                cmap="jet",
                point_size=POINT_SIZE,
                render_points_as_spheres=False,
            )

        print(
            "Angle =", angle_raw,
            "Points kept =", n_kept,
            "Percentage =", round(100 * n_kept / n_total, 2), "%",
            "Mesh generada local en", round(time.perf_counter() - t1, 2), "s"
        )

    if EXPORT_VTS and points_3d is not None and points_3d.shape[2] > 0:
        print("\nGenerando malla estructurada (.vts) para ParaView...")
        all_points_vts = points_3d.reshape(-1, 3, order="F")
        all_tl_vts = tl_3d.reshape(-1, order="F")

        nz = points_3d.shape[2]
        expected_points = nx * ny * nz
        if all_points_vts.shape[0] != expected_points:
            print(f"⚠️ ERROR AL EXPORTAR: N_Puntos cargados ({all_points_vts.shape[0]}) "
                  f"≠ Nx*Ny*Nz Esperados ({expected_points}).\n"
                  "Revisa la variable PLANE_GRID_SHAPE.")
        else:
            grid = pv.StructuredGrid()
            grid.points = all_points_vts
            grid.dimensions = (nx, ny, nz)
            grid["tl"] = all_tl_vts

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            grid.save(output_path)
            print(f"✅ VTS exportado a: {output_path} con dimensions ({nx}, {ny}, {nz})")

    total_time = time.perf_counter() - t0
    print("\nTiempo total:", round(total_time, 2), "s")

    if SHOW_PYVISTA:
        plotter.show()

print("🔍 Generando VTS de Ground Truth desde todo el árbol de Interpolation/input...")
files = prepare_files(collect_mat_files(GROUND_TRUTH_FOLDER), n_planes=N_PLANES)
export_case_vts("ground_truth_case30", files, OUTPUT_VTS_PATH)