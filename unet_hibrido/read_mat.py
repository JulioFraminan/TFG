import h5py
import numpy as np
from data_utils import phys_to_pixel, _get_z_axis
import config as cfg
from pathlib import Path
BASE = Path(__file__).resolve().parent
p = BASE / 'input' / 'Flat_SeabedSand_PlaneAngle0.00_Length299.00.mat'


def _pick_dataset(file_handle, keys):
    for key in keys:
        if key in file_handle:
            return file_handle[key][:], key
    raise KeyError(f'No se encontro ninguno de estos datasets: {keys}')


with h5py.File(p,'r') as f:
    keys = list(f.keys())
    print('keys:', keys)
    for k in keys:
        try:
            arr = f[k][:]
            print(k, arr.shape, arr.dtype)
        except Exception as e:
            print(k, '->', e)
    # check common datasets
    tl_data = None
    tl_name = None
    for name in ('tl', 'TL', 'tL', 'tl_smooth', 'tl_block'):
        if name in f:
            tl_data = f[name][:]
            tl_name = name
            print(f"{name} shape raw: {tl_data.shape}")
            try:
                print(f"{name}.T shape: {tl_data.T.shape}")
            except Exception:
                pass
            break
    if ('R' in f or 'X' in f) and 'Z' in f:
        X_raw = f['R'][:] if 'R' in f else f['X'][:]
        Z_raw = f['Z'][:]
        z_axis = _get_z_axis(Z_raw)
        print('X_raw shape:', X_raw.shape)
        print('Z_raw shape:', Z_raw.shape)
        i_top, j0 = phys_to_pixel(X_raw, Z_raw, cfg.ROI_CORNER[0], cfg.ROI_CORNER[1])
        print('ROI_CORNER in pixels -> i_top=', i_top, 'j0=', j0)
        if tl_data is not None:
            rows, cols = tl_data.T.shape
            print(f'{tl_name}.T rows,cols =', (rows, cols))
            max_h = rows - i_top
            max_w = cols - j0
            print('Max ROI permitida desde ROI_CORNER:')
            print('  altura max (px) =', max_h)
            print('  ancho max (px)  =', max_w)
            print('  ROI_HEIGHT config =', cfg.ROI_HEIGHT)
            print('  ROI_WIDTH config  =', cfg.ROI_WIDTH)
            print('  cabe en altura?   =', cfg.ROI_HEIGHT <= max_h)
            print('  cabe en ancho?    =', cfg.ROI_WIDTH <= max_w)
