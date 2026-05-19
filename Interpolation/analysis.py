import os

import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_interpolation_data, adaptive_figsize
from config import (
    ROI_HEIGHT, ROI_WIDTH,
    ROIS_PER_PLANE,
    ROI_MODE, ROI_CORNER,
    DATA_FOLDER,
    ANALYSIS_PNG_FOLDER,
    create_output_dirs,
)


def _choose_extent(ext_low, ext_high, weight):
    if ext_low is None:
        return ext_high
    if ext_high is None:
        return ext_low
    try:
        if np.allclose(ext_low, ext_high, rtol=1e-6, atol=1e-6):
            return ext_low
    except Exception:
        pass
    return ext_low if weight <= 0.5 else ext_high


def main():
    create_output_dirs(ANALYSIS_PNG_FOLDER)

    # ══════════════════════════════════════════════════════════════════════
    #  CARGAR DATOS Y NORMALIZAR
    # ══════════════════════════════════════════════════════════════════════
    rois_norm, roi_angles, _, norm, roi_extents = \
        load_interpolation_data(DATA_FOLDER,
                                ROI_HEIGHT, ROI_WIDTH,
                                ROIS_PER_PLANE,
                                roi_mode=ROI_MODE,
                                roi_corner=ROI_CORNER)

    # ══════════════════════════════════════════════════════════════════════
    #  INTERPOLACION LINEAL ENTRE DOS PLANOS
    # ══════════════════════════════════════════════════════════════════════
    if len(rois_norm) >= 2:
        angles_arr = np.array(roi_angles, dtype=np.float32)
        idx_low = int(np.argmin(angles_arr))
        idx_high = int(np.argmax(angles_arr))

        angle_low = float(roi_angles[idx_low])
        angle_high = float(roi_angles[idx_high])
        plane_low = rois_norm[idx_low]
        plane_high = rois_norm[idx_high]

        print("\nInterpolacion lineal entre plano inf y plano sup...")
        n_interp = 8
        ext = _choose_extent(roi_extents[idx_low], roi_extents[idx_high], 0.5)
        fig, axes = plt.subplots(
            1, n_interp,
            figsize=adaptive_figsize(ROI_HEIGHT, ROI_WIDTH, 1, n_interp),
        )
        for k, alpha in enumerate(np.linspace(0, 1, n_interp)):
            interp_norm = (1.0 - alpha) * plane_low + alpha * plane_high
            interp_tl = norm.denormalize_tl(np.clip(interp_norm, 0, 1))
            ang_mix = (1.0 - alpha) * angle_low + alpha * angle_high

            axes[k].imshow(
                interp_tl, cmap="jet", aspect="auto", origin="lower",
                vmin=norm.tl_min, vmax=norm.tl_max, extent=ext,
            )
            axes[k].set_title(f"a={alpha:.2f}\n{ang_mix:.1f}", fontsize=8)
            axes[k].set_xlabel("X"); axes[k].set_ylabel("Z")

        fig.suptitle(
            f"Interpolacion lineal: {angle_low:.1f}° → {angle_high:.1f}°",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(ANALYSIS_PNG_FOLDER, "interpolacion_lineal.png"), dpi=150)
        plt.close(fig)
        print("Interpolacion lineal guardada.")
    else:
        print("Se necesitan al menos 2 planos para interpolar.")

    print("\n" + "=" * 60)
    print(f"  ¡Análisis completado! Figuras en:\n  {ANALYSIS_PNG_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()
