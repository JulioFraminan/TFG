import os

import numpy as np
import matplotlib.pyplot as plt

from data_utils import (
    load_all_rois, load_validation_rois,
    Normalizer, save_mat, adaptive_figsize,
    interpolate_planes, compute_error_metrics,
)
from config import (
    ROI_HEIGHT, ROI_WIDTH,
    ROIS_PER_PLANE,
    ROI_MODE, ROI_CORNER,
    DATA_FOLDER,
    VALIDATION_FOLDER,
    TRAIN_MAT_FOLDER,
    VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER,
    create_output_dirs,
)


def main():
    create_output_dirs(TRAIN_MAT_FOLDER)

    # ------------------------------------------------------------------
    # 1. Cargar datos y extraer ROIs
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  CARGANDO PLANOS Y EXTRAYENDO ROIs")
    print("=" * 60)
    rois, roi_angles, roi_extents = load_all_rois(
        DATA_FOLDER, ROI_HEIGHT, ROI_WIDTH, ROIS_PER_PLANE,
        roi_mode=ROI_MODE, roi_corner=ROI_CORNER,
    )
    if rois is None or len(rois) == 0:
        print("\n[ERROR] No se encontraron ROIs. Revisa DATA_FOLDER y config.py.")
        return
    print(f"\nROIs base: {len(rois)}  |  Forma: {rois.shape}")

    # ------------------------------------------------------------------
    # 2. Normalizacion
    # ------------------------------------------------------------------
    try:
        norm = Normalizer(rois, roi_angles)
    except ValueError as e:
        print(f"\n  [ERROR] No se pudieron inicializar normalizadores: {e}")
        print("  Revisa DATA_FOLDER, la estructura de los .mat y los parametros ROI en config.py.")
        return
    rois_norm = norm.normalize_tl(rois)
    print(f"Rango TL: [{norm.tl_min:.2f}, {norm.tl_max:.2f}]")
    print(f"Rango angulos: [{norm.angle_min:.2f} deg, {norm.angle_max:.2f} deg]")

    # ------------------------------------------------------------------
    # 3. Exportar ROIs originales
    # ------------------------------------------------------------------
    print("Exportando ROIs originales como .mat...")
    for j in range(len(rois_norm)):
        roi_tl = norm.denormalize_tl(rois_norm[j])
        roi_mat_path = os.path.join(TRAIN_MAT_FOLDER, f"roi_original_{roi_angles[j]:+.2f}.mat")
        save_mat(roi_mat_path, roi_tl, extent=roi_extents[j])
    print(f"  {len(rois_norm)} ROIs exportadas en {TRAIN_MAT_FOLDER}")

    # ------------------------------------------------------------------
    # 4. Validacion con planos no vistos (interpolacion lineal)
    # ------------------------------------------------------------------
    val_rois, val_angles, val_extents = load_validation_rois(
        VALIDATION_FOLDER, ROI_HEIGHT, ROI_WIDTH,
        roi_mode=ROI_MODE, roi_corner=ROI_CORNER,
    )
    if val_rois is None or len(val_rois) == 0:
        print("\n  [i] No se encontraron planos de validacion en "
              f"{VALIDATION_FOLDER}")
        print("      Para usarlo, coloca .mat con 'PlaneAngle' en el nombre.")
        return

    create_output_dirs(VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER)
    val_rois_norm = norm.normalize_tl(val_rois)
    print(f"\nValidando con {len(val_rois)} planos no vistos...")

    n_val = len(val_rois)
    n_rows_per_page = min(n_val, 6)
    n_pages = int(np.ceil(n_val / n_rows_per_page))

    all_maes = []
    all_maxes = []
    all_gaps = []
    all_val_angles_plot = []
    all_mapes = []
    all_rmses = []

    for page in range(n_pages):
        start = page * n_rows_per_page
        end = min(start + n_rows_per_page, n_val)
        rows_this = end - start

        fig, axes = plt.subplots(
            rows_this, 3,
            figsize=adaptive_figsize(ROI_HEIGHT, ROI_WIDTH, rows_this, 3),
        )
        if rows_this == 1:
            axes = axes[np.newaxis, :]

        for r, j in enumerate(range(start, end)):
            val_angle = float(val_angles[j])
            ext = val_extents[j]

            interp_norm, idx_low, idx_high, weight, clipped = interpolate_planes(
                rois_norm, roi_angles, val_angle
            )
            ang_low = float(roi_angles[idx_low])
            ang_high = float(roi_angles[idx_high])
            diff_low = abs(val_angle - ang_low)
            diff_high = abs(ang_high - val_angle)
            gap = abs(ang_high - ang_low)

            interp_tl = norm.denormalize_tl(interp_norm)
            real_tl = norm.denormalize_tl(val_rois_norm[j])
            diff_map = np.abs(interp_tl - real_tl)

            errors = compute_error_metrics(real_tl, interp_tl)
            mae_val = errors['mae']
            max_val = errors['max_error']
            rmse_val = errors['rmse']
            mape_val = errors['mape']

            lo_str = f"{ang_low:+.2f} deg (dif={diff_low:.2f} deg)"
            hi_str = f"{ang_high:+.2f} deg (dif={diff_high:.2f} deg)"
            gap_str = f"{gap:.2f} deg"
            mape_str = f"{mape_val:.2f}%" if mape_val != np.inf else "undef"
            clip_note = " (clamped)" if clipped else ""
            print(
                f"  Angulo val {val_angle:+.2f} deg  |  "
                f"vecino inf: {lo_str}  |  vecino sup: {hi_str}  |  "
                f"gap: {gap_str}  |  w={weight:.3f}{clip_note}  |  "
                f"MAE: {mae_val:.2f} dB  |  RMSE: {rmse_val:.2f} dB  |  "
                f"MAPE: {mape_str}  |  Max: {max_val:.2f} dB"
            )

            all_maes.append(mae_val)
            all_maxes.append(max_val)
            all_gaps.append(gap)
            all_val_angles_plot.append(val_angle)
            all_mapes.append(mape_val)
            all_rmses.append(rmse_val)

            mat_path = os.path.join(
                VALIDATION_MAT_FOLDER,
                f"val_interpolado_{val_angle:+.2f}.mat",
            )
            save_mat(mat_path, interp_tl, extent=ext)

            neigh_txt = (
                f"Vecino inf: {ang_low:+.1f} deg (d={diff_low:.1f})"
                f"  |  Vecino sup: {ang_high:+.1f} deg (d={diff_high:.1f})"
                f"  |  Gap total: {gap:.1f} deg"
            )

            angle_str = f"{val_angle:.2f} deg"

            im0 = axes[r, 0].imshow(
                real_tl, cmap="jet", aspect="auto", origin="lower",
                vmin=norm.tl_min, vmax=norm.tl_max, extent=ext,
            )
            fig.colorbar(im0, ax=axes[r, 0], shrink=0.6, label="TL (dB)")
            axes[r, 0].set_title(
                f"Real -- {angle_str}\n{neigh_txt}", fontsize=8,
            )
            axes[r, 0].set_xlabel("X (m)"); axes[r, 0].set_ylabel("Z (m)")

            im1 = axes[r, 1].imshow(
                interp_tl, cmap="jet", aspect="auto", origin="lower",
                vmin=norm.tl_min, vmax=norm.tl_max, extent=ext,
            )
            fig.colorbar(im1, ax=axes[r, 1], shrink=0.6, label="TL (dB)")
            axes[r, 1].set_title(
                f"Interpolado -- {angle_str}\n(w={weight:.2f})", fontsize=8,
            )
            axes[r, 1].set_xlabel("X (m)"); axes[r, 1].set_ylabel("Z (m)")

            im2 = axes[r, 2].imshow(
                diff_map, cmap="turbo", aspect="auto", origin="lower",
                vmin=0, vmax=max_val or 1e-6, extent=ext,
            )
            fig.colorbar(im2, ax=axes[r, 2], shrink=0.6, label="|Error| (dB)")
            mape_str = f"{mape_val:.1f}%" if mape_val != np.inf else "undef"
            axes[r, 2].set_title(
                f"|Error| -- {angle_str}\nMAE={mae_val:.2f} dB  "
                f"MAPE={mape_str}  RMSE={rmse_val:.2f} dB  Max={max_val:.2f} dB",
                fontsize=7,
            )
            axes[r, 2].set_xlabel("X (m)"); axes[r, 2].set_ylabel("Z (m)")

        fig.suptitle(
            f"Validacion: Real vs Interpolado (lineal)  "
            f"(pag. {page + 1}/{n_pages})", fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                VALIDATION_PNG_FOLDER,
                f"validacion_real_vs_interpolado_{page + 1:02d}.png",
            ),
            dpi=150,
        )
        plt.close(fig)
        print(f"  Validacion pag. {page + 1}/{n_pages} guardada ({rows_this} planos).")

    plot_data = [
        (g, m, mx, r, mp, a)
        for g, m, mx, r, mp, a in zip(
            all_gaps, all_maes, all_maxes, all_rmses, all_mapes, all_val_angles_plot
        )
        if g is not None
    ]
    if plot_data:
        plot_data.sort(key=lambda d: d[0])
        gaps_arr = np.array([d[0] for d in plot_data])
        maes_arr = np.array([d[1] for d in plot_data])
        maxes_arr = np.array([d[2] for d in plot_data])
        rmses_arr = np.array([d[3] for d in plot_data])
        mapes_arr = np.array([d[4] for d in plot_data])
        labels_arr = [f"{d[5]:+.2f} deg" for d in plot_data]

        fig_err, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        color_mae = "#1f77b4"
        color_rmse = "#ff7f0e"
        color_max = "#d62728"

        ax1.plot(gaps_arr, maes_arr, color=color_mae, linewidth=1.2, alpha=0.6, zorder=2)
        ax1.plot(gaps_arr, rmses_arr, color=color_rmse, linewidth=1.2, alpha=0.6, zorder=2)
        ax1.plot(gaps_arr, maxes_arr, color=color_max, linewidth=1.2, alpha=0.6, linestyle="--", zorder=2)

        ax1.scatter(gaps_arr, maes_arr, color=color_mae, s=50, zorder=3, label="MAE (dB)")
        ax1.scatter(gaps_arr, rmses_arr, color=color_rmse, s=50, zorder=3, label="RMSE (dB)")
        ax1.scatter(gaps_arr, maxes_arr, color=color_max, s=50, marker="^", zorder=3, label="Max error (dB)")

        ax1.set_xlabel("Gap entre vecinos de entrenamiento (deg)", fontsize=10)
        ax1.set_ylabel("Error (dB)", fontsize=10)
        ax1.set_title("Errores Absolutos vs Distancia Angular", fontsize=11)
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(True, alpha=0.3)

        color_mape = "#2ca02c"
        ax2.plot(gaps_arr, mapes_arr, color=color_mape, linewidth=1.2, alpha=0.6, zorder=2)
        ax2.scatter(gaps_arr, mapes_arr, color=color_mape, s=50, zorder=3, label="MAPE (%)")

        for xi, yi, lbl in zip(gaps_arr, mapes_arr, labels_arr):
            ax2.annotate(lbl, (xi, yi), textcoords="offset points",
                         xytext=(5, 5), fontsize=7, color=color_mape)

        ax2.set_xlabel("Gap entre vecinos de entrenamiento (deg)", fontsize=10)
        ax2.set_ylabel("MAPE (%)", fontsize=10)
        ax2.set_title("Error Relativo (MAPE) vs Distancia Angular", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig_err.suptitle(
            "Validacion (interpolacion lineal) - Error vs gap",
            fontsize=12,
        )
        fig_err.tight_layout()
        err_path = os.path.join(VALIDATION_PNG_FOLDER, "error_vs_gap.png")
        fig_err.savefig(err_path, dpi=150)
        plt.close(fig_err)
        print(f"  Grafico error vs gap guardado en: {err_path}")

    mean_mae = np.mean(all_maes) if all_maes else float("nan")
    mean_rmse = np.mean(all_rmses) if all_rmses else float("nan")
    mapes_ok = [m for m in all_mapes if m != np.inf]
    mean_mape = np.mean(mapes_ok) if mapes_ok else float("nan")

    print("\n  Validacion completada:")
    print(f"    MAE medio   = {mean_mae:.2f} dB")
    print(f"    RMSE medio  = {mean_rmse:.2f} dB")
    print(f"    MAPE medio  = {mean_mape:.1f}%")
    print(f"    sobre {len(all_maes)} planos.")
    print(f"  Resultados en: {VALIDATION_PNG_FOLDER}")
    print(f"                 {VALIDATION_MAT_FOLDER}")

    print("\n" + "=" * 60)
    print(f"  LISTO! Resultados en:\n  {TRAIN_MAT_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()