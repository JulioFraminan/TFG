import os

import numpy as np
import matplotlib.pyplot as plt
from data_utils import (
    load_interpolation_data, save_mat, adaptive_figsize,
    interpolate_planes, compute_error_metrics,
)
from config import (
    ROI_HEIGHT, ROI_WIDTH,
    ROIS_PER_PLANE, GENERATE_ANGLES,
    ROI_MODE, ROI_CORNER,
    DATA_FOLDER,
    GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER,
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
    create_output_dirs(GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER)

    # ══════════════════════════════════════════════════════════════════════
    #  CARGAR DATOS Y NORMALIZAR
    # ══════════════════════════════════════════════════════════════════════
    rois_norm, roi_angles, _, norm, roi_extents = \
        load_interpolation_data(DATA_FOLDER,
                                ROI_HEIGHT, ROI_WIDTH,
                                ROIS_PER_PLANE,
                                roi_mode=ROI_MODE,
                                roi_corner=ROI_CORNER)

    if len(rois_norm) < 2:
        print("Se necesitan al menos 2 planos para interpolar.")
        return

    # ══════════════════════════════════════════════════════════════════════
    #  1. GENERACIÓN CONDICIONADA POR ÁNGULO
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  GENERANDO PLANOS PARA {len(GENERATE_ANGLES)} ÁNGULOS")
    print("=" * 60)

    for target_angle in GENERATE_ANGLES:
        interp_norm, idx_low, idx_high, weight, clipped = interpolate_planes(
            rois_norm, roi_angles, target_angle
        )
        interp_tl = norm.denormalize_tl(interp_norm)

        angle_low = float(roi_angles[idx_low])
        angle_high = float(roi_angles[idx_high])
        plane_low = norm.denormalize_tl(rois_norm[idx_low])
        plane_high = norm.denormalize_tl(rois_norm[idx_high])

        error_low = np.abs(interp_tl - plane_low)
        error_high = np.abs(interp_tl - plane_high)
        error_between = np.abs(plane_high - plane_low)

        stats_low = compute_error_metrics(plane_low, interp_tl)
        stats_high = compute_error_metrics(plane_high, interp_tl)

        ext = _choose_extent(roi_extents[idx_low], roi_extents[idx_high], weight)
        mat_path = os.path.join(GENERATE_MAT_FOLDER, f"plano_angulo_{target_angle:+.1f}.mat")
        save_mat(mat_path, interp_tl, extent=ext)

        # --- Figura con 6 subplots ---
        fig, axes = plt.subplots(
            1, 6,
            figsize=adaptive_figsize(interp_tl.shape[0], interp_tl.shape[1], 1, 6),
        )

        pcm0 = axes[0].imshow(interp_tl, cmap="jet", aspect="auto", origin="lower",
                              vmin=norm.tl_min, vmax=norm.tl_max, extent=ext)
        fig.colorbar(pcm0, ax=axes[0], label="TL (dB)", shrink=0.8)
        axes[0].set_title(f"Interpolado -- objetivo {target_angle:.1f}", fontsize=11)
        axes[0].set_xlabel("X"); axes[0].set_ylabel("Z")

        pcm1 = axes[1].imshow(plane_low, cmap="jet", aspect="auto", origin="lower",
                              vmin=norm.tl_min, vmax=norm.tl_max, extent=ext)
        fig.colorbar(pcm1, ax=axes[1], label="TL (dB)", shrink=0.8)
        axes[1].set_title(f"Plano inf -- {angle_low:.2f}", fontsize=11)
        axes[1].set_xlabel("X"); axes[1].set_ylabel("Z")

        pcm2 = axes[2].imshow(plane_high, cmap="jet", aspect="auto", origin="lower",
                              vmin=norm.tl_min, vmax=norm.tl_max, extent=ext)
        fig.colorbar(pcm2, ax=axes[2], label="TL (dB)", shrink=0.8)
        axes[2].set_title(f"Plano sup -- {angle_high:.2f}", fontsize=11)
        axes[2].set_xlabel("X"); axes[2].set_ylabel("Z")

        pcm3 = axes[3].imshow(error_low, cmap="turbo", aspect="auto", origin="lower",
                              vmin=0, vmax=max(stats_low['max_error'], 1e-6), extent=ext)
        fig.colorbar(pcm3, ax=axes[3], label="|Error| (dB)", shrink=0.8)
        axes[3].set_title(
            f"|Interp - inf| MAE={stats_low['mae']:.2f}, Max={stats_low['max_error']:.2f}",
            fontsize=9,
        )
        axes[3].set_xlabel("X"); axes[3].set_ylabel("Z")

        pcm4 = axes[4].imshow(error_high, cmap="turbo", aspect="auto", origin="lower",
                              vmin=0, vmax=max(stats_high['max_error'], 1e-6), extent=ext)
        fig.colorbar(pcm4, ax=axes[4], label="|Error| (dB)", shrink=0.8)
        axes[4].set_title(
            f"|Interp - sup| MAE={stats_high['mae']:.2f}, Max={stats_high['max_error']:.2f}",
            fontsize=9,
        )
        axes[4].set_xlabel("X"); axes[4].set_ylabel("Z")

        pcm5 = axes[5].imshow(error_between, cmap="turbo", aspect="auto", origin="lower",
                              vmin=0, vmax=max(np.max(error_between), 1e-6), extent=ext)
        fig.colorbar(pcm5, ax=axes[5], label="|Dif| (dB)", shrink=0.8)
        axes[5].set_title(
            f"|Sup - Inf| gap={abs(angle_high - angle_low):.2f}°", fontsize=9,
        )
        axes[5].set_xlabel("X"); axes[5].set_ylabel("Z")

        fig.suptitle(
            f"Objetivo: {target_angle:.1f}°  |  Vecinos: {angle_low:.2f}° / {angle_high:.2f}°",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()
        png_path = os.path.join(GENERATE_PNG_FOLDER, f"plano_angulo_{target_angle:+.1f}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        clip_note = " (clamped)" if clipped else ""
        print(
            f"  angulo={target_angle:+7.1f}°  vecinos {angle_low:.2f}°/{angle_high:.2f}°"
            f"  w={weight:.3f}{clip_note}  MAE_inf={stats_low['mae']:.2f} dB"
            f"  MAE_sup={stats_high['mae']:.2f} dB  →  {mat_path}"
        )

    print("\n" + "=" * 60)
    print(f"  ¡LISTO! Resultados en:")
    print(f"  PNG: {GENERATE_PNG_FOLDER}")
    print(f"  MAT: {GENERATE_MAT_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()
