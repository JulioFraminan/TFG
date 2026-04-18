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


folder = r"/home/j.framinan/TFG_repo/unet_ae_modular/output/generate/MAT"
N_PLANES = None  # Pon None para cargar TODOS los archivos de la carpeta, o un número en específico

# Configuración de exportación y visualización
SHOW_PYVISTA = False  # True: muestra ventana 3D, False: procesa pero no renderiza
EXPORT_VTS = True    # True: exporta a un grid estructurado .vts
VTS_OUTPUT_PATH = os.path.join(folder, "planos_rotados.vts")

# MUY IMPORTANTE: Dimensiones de cada plano SIN filtrar. (p. ej. Nx, Ny)
# El tamaño total del array original en el .mat debe ser igual a NX * NY
PLANE_GRID_SHAPE = (700, 2000) 

# Umbral de intensidad:
ENABLE_TL_THRESHOLD = False
TL_MIN = -99.999
POINT_SIZE = 1.0

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

#------ Cargar archivos, procesar y mostrar con PyVista ------#


all_files = glob.glob(os.path.join(folder, "*.[mM][aA][tT]"))
# Ordenar por timestamp (más recientes primero), luego por ángulo
all_files_with_time = [(f, os.path.getmtime(f)) for f in all_files]
all_files_with_time.sort(key=lambda x: (-x[1], extract_angle(x[0])))

# Si N_PLANES es None, carga todos; si no, toma los N más recientes
if N_PLANES is not None:
    files = [f[0] for f in all_files_with_time[:N_PLANES]]
else:
    files = [f[0] for f in all_files_with_time]

files = sorted(files, key=extract_angle)  # Re-ordenar por ángulo SIEMPRE para que el 3D no se tuerza

if len(files) == 0:
    raise FileNotFoundError(f"No se encontraron archivos .mat en: {folder}")

angles = [extract_angle(p) for p in files]
use_radians = max(map(abs, angles)) <= 6.5

plotter = pv.Plotter() if SHOW_PYVISTA else None
t0 = time.perf_counter()

# Buffers para Structured Grid (exportación completa sin filtrar)
nx, ny = PLANE_GRID_SHAPE
if EXPORT_VTS:
    nz = len(files)
    points_3d = np.empty((nx, ny, nz, 3), dtype=np.float32)
    tl_3d = np.empty((nx, ny, nz), dtype=np.float32)
else:
    points_3d = None
    tl_3d = None

#------ Representacion ------#


with h5py.File(files[0], "r") as f:
    X0 = np.float32(f["X"][0, 0])
    Y0 = np.float32(f["Y"][0, 0])
    Z0 = np.float32(f["Z"][0, 0])

for i, path in enumerate(files):

    angle_raw = extract_angle(path)
    t1 = time.perf_counter()

    with h5py.File(path, "r") as f:
        X, Y, Z, tl = (
            np.asarray(f[k][:], np.float32)
            for k in ("X", "Y", "Z", "tl")
        )

    # ----- Origen común -----
    X -= X0
    Y -= Y0
    Z -= Z0

    # ----- Guardar datos de CADA PLANO sin filtrar para Malla Estructurada (.vts) -----
    if EXPORT_VTS:
        if X.shape != (nx, ny):
            if X.shape == (ny, nx):
                # Corrige archivos transpuestos sin romper el resto del flujo.
                X = X.T
                Y = Y.T
                Z = Z.T
                tl = tl.T
            else:
                raise ValueError(
                    f"Shape inválido en {os.path.basename(path)}: {X.shape}. "
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
    Yk = Y[keep]
    Zk = Z[keep]
    tlk = tl[keep]

    n_kept = Xk.size
    n_total = X.size

    # ----- Conversión cilíndrico -> cartesiano -----
    # X es el radio (r), Z es altura (z), ángulo viene del archivo
    angle_rad = angle_raw if use_radians else np.deg2rad(angle_raw)

    r = Xk  # Radio
    theta = angle_rad
    z = Zk  # Altura (eje común)

    points = np.empty((n_kept, 3), dtype=np.float32)
    points[:, 0] = r * np.cos(theta) 
    points[:, 1] = r * np.sin(theta) 
    points[:, 2] = z  # Z es el eje común

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

# --- EXPORTACIÓN A STRUCTURED GRID (.vts) ---
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
        
        output_dir = os.path.dirname(VTS_OUTPUT_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        grid.save(VTS_OUTPUT_PATH)
        print(f"✅ VTS exportado a: {VTS_OUTPUT_PATH} con dimensions ({nx}, {ny}, {nz})")

total_time = time.perf_counter() - t0
print("\nTiempo total:", round(total_time, 2), "s")

if SHOW_PYVISTA:
    plotter.show()