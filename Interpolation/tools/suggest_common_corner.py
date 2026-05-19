#!/usr/bin/env python3
"""Sugiere una esquina inferior-derecha común (X_right, Z_bottom).

Analiza todos los .mat de input/ y devuelve:
 - min/max X y Z por fichero
 - para una lista de candidatas X_right muestra cobertura (#ficheros que la contienen)
 - sugiere X_right con cobertura >= threshold (por defecto 0.8) eligiendo la mayor X posible
 - sugiere Z_bottom = max(min_Z) para que esté dentro de todos los ficheros
"""
import os
import h5py
import numpy as np

try:
    from config import DATA_FOLDER
except Exception:
    import importlib.util
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, 'config.py')
    if not os.path.isfile(config_path):
        raise SystemExit(f"No se encontró config.py en {config_path}")
    spec = importlib.util.spec_from_file_location('config', config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    DATA_FOLDER = cfg.DATA_FOLDER

CANDIDATES = [-1400, -1325, -1300, -1250, -1200, -1150, -1125, -1100, -1075, -1050, -1025, -1000, -950, -900, -850, -825, -800]
THRESHOLD = 0.8

def analyze():
    folder = DATA_FOLDER
    mats = sorted([p for p in os.listdir(folder) if p.endswith('.mat') and 'PlaneAngle' in p])
    if not mats:
        print('No .mat encontrados en', folder)
        return

    x_mins = []
    x_maxs = []
    z_mins = []
    z_maxs = []

    for fname in mats:
        path = os.path.join(folder, fname)
        try:
            with h5py.File(path, 'r') as f:
                X = f['X'][:]
                Z = f['Z'][:]
        except Exception as e:
            print('ERROR leyendo', fname, e)
            continue
        x_mins.append(float(X.min()))
        x_maxs.append(float(X.max()))
        z_mins.append(float(Z.min()))
        z_maxs.append(float(Z.max()))

    n = len(x_mins)
    print(f'Analizados {n} ficheros\. Rango X global: {min(x_mins):.2f} .. {max(x_maxs):.2f}')
    print(f'Rango Z global: {min(z_mins):.2f} .. {max(z_maxs):.2f}')
    print('\nResumen por fichero (primeros 10):')
    for i, fname in enumerate(mats[:10]):
        print(f'  {i+1:02d} {fname}: X=[{x_mins[i]:.2f},{x_maxs[i]:.2f}] Z=[{z_mins[i]:.2f},{z_maxs[i]:.2f}]')

    print('\nCobertura candidatos X_right:')
    counts = {}
    for c in CANDIDATES:
        cnt = sum(1 for xmax in x_maxs if xmax >= c)
        counts[c] = cnt
        print(f'  X_right={c:7.2f} -> {cnt}/{n} files ({cnt/n:.2%})')

    # encontrar la mayor X_right con coverage >= THRESHOLD
    viable = [c for c in CANDIDATES if counts[c]/n >= THRESHOLD]
    if viable:
        best = max(viable)
        print(f"\nSugerencia (coverage >= {THRESHOLD:.0%}): X_right = {best}")
    else:
        # sugerir el minimo de X_max (presente en todos)
        min_xmax = min(x_maxs)
        print(f"\nNingun candidato alcanza coverage >= {THRESHOLD:.0%}.\nSugerencia conservadora: X_right = min(X_max) = {min_xmax:.2f} (presente en todos)")

    # Z_bottom sugerido: hacerlo conservador para que esté en todos
    suggested_z_bottom = max(z_mins)
    suggested_z_top = min(z_maxs)
    print(f"\nSugerencia Z: usar Z_bottom = {suggested_z_bottom:.2f} y Z_top = {suggested_z_top:.2f} para estar dentro de todos los ficheros.")
    print('\nHecho.')

if __name__ == '__main__':
    analyze()
