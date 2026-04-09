import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from data_utils import (
    load_inference_pipeline, save_mat, adaptive_figsize,
    generate_from_seed,
)
from config import (
    ROI_HEIGHT, ROI_WIDTH,
    ROIS_PER_PLANE, NOISE_STD, GENERATE_ANGLES,
    ROI_MODE, ROI_CORNER,
    DATA_FOLDER, MODEL_PATH,
    GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER,
    create_output_dirs,
)


def main():
    create_output_dirs(GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER)

    # ══════════════════════════════════════════════════════════════════════
    #  CARGAR MODELO Y DATOS (normalizer desde checkpoint)
    # ══════════════════════════════════════════════════════════════════════
    device, model, rois_norm, roi_angles, angles_norm, norm, roi_extents = \
        load_inference_pipeline(MODEL_PATH, DATA_FOLDER,
                                ROI_HEIGHT, ROI_WIDTH,
                                ROIS_PER_PLANE,
                                roi_mode=ROI_MODE,
                                roi_corner=ROI_CORNER)

    # ══════════════════════════════════════════════════════════════════════
    #  1. GENERACIÓN CONDICIONADA POR ÁNGULO
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  GENERANDO PLANOS PARA {len(GENERATE_ANGLES)} ÁNGULOS")
    print("=" * 60)

    with torch.no_grad():
        base_samples = torch.from_numpy(rois_norm[:, np.newaxis, :, :]).to(device)

        roi_angles_arr = np.array(roi_angles)

        for target_angle in GENERATE_ANGLES:
            dists = np.abs(roi_angles_arr - target_angle)
            idx = np.argmin(dists)
            seed = base_samples[idx:idx+1]

            gen_tl = generate_from_seed(
                model, seed, target_angle, norm, NOISE_STD, device,
            )

            seed_norm = rois_norm[idx]
            seed_tl = norm.denormalize_tl(seed_norm)

            error_tl = np.abs(gen_tl - seed_tl)
            mae_val = np.mean(error_tl)
            max_err = np.max(error_tl)

            mat_path = os.path.join(GENERATE_MAT_FOLDER, f"plano_angulo_{target_angle:+.1f}.mat")
            save_mat(mat_path, gen_tl)

            # Reconstrucción de la semilla (sin ruido)
            cond_seed = torch.tensor(
                [[angles_norm[idx]]], dtype=torch.float32, device=device,
            )
            recon_seed_norm = model(seed, cond_seed).cpu().squeeze().numpy()
            recon_seed_norm = np.clip(recon_seed_norm, 0, 1)
            recon_seed_tl = norm.denormalize_tl(recon_seed_norm)

            error_recon_gen = np.abs(gen_tl - recon_seed_tl)
            mae_rg = np.mean(error_recon_gen)
            max_rg = np.max(error_recon_gen)

            error_recon = np.abs(seed_tl - recon_seed_tl)
            mae_recon = np.mean(error_recon)
            max_recon = np.max(error_recon)

            # --- Figura con 6 subplots ---
            ext = roi_extents[idx]   # extension fisica de la semilla
            fig, axes = plt.subplots(
                1, 6,
                figsize=adaptive_figsize(gen_tl.shape[0], gen_tl.shape[1], 1, 6),
            )

            pcm0 = axes[0].imshow(gen_tl, cmap="jet", aspect="auto", origin="lower",
                                  vmin=norm.tl_min, vmax=norm.tl_max, extent=ext)
            fig.colorbar(pcm0, ax=axes[0], label="TL (dB)", shrink=0.8)
            axes[0].set_title(f"Generado -- angulo objetivo = {target_angle:.1f}", fontsize=11)
            axes[0].set_xlabel("X"); axes[0].set_ylabel("Z")

            pcm1 = axes[1].imshow(seed_tl, cmap="jet", aspect="auto", origin="lower",
                                  vmin=norm.tl_min, vmax=norm.tl_max, extent=ext)
            fig.colorbar(pcm1, ax=axes[1], label="TL (dB)", shrink=0.8)
            axes[1].set_title(f"Semilla original -- {roi_angles[idx]:.2f}", fontsize=11)
            axes[1].set_xlabel("X"); axes[1].set_ylabel("Z")

            pcm2 = axes[2].imshow(recon_seed_tl, cmap="jet", aspect="auto", origin="lower",
                                  vmin=norm.tl_min, vmax=norm.tl_max, extent=ext)
            fig.colorbar(pcm2, ax=axes[2], label="TL (dB)", shrink=0.8)
            axes[2].set_title(f"Semilla reconstruida -- {roi_angles[idx]:.2f}", fontsize=11)
            axes[2].set_xlabel("X"); axes[2].set_ylabel("Z")

            pcm3 = axes[3].imshow(error_recon, cmap="turbo", aspect="auto", origin="lower",
                                  vmin=0, vmax=max(max_recon, 1e-6), extent=ext)
            fig.colorbar(pcm3, ax=axes[3], label="|Error| (dB)", shrink=0.8)
            axes[3].set_title(f"|Seed orig - Seed recon| MAE={mae_recon:.2f}, Max={max_recon:.2f}", fontsize=9)
            axes[3].set_xlabel("X"); axes[3].set_ylabel("Z")

            pcm4 = axes[4].imshow(error_tl, cmap="turbo", aspect="auto", origin="lower",
                                  vmin=0, vmax=max(max_err, 1e-6), extent=ext)
            fig.colorbar(pcm4, ax=axes[4], label="|Error| (dB)", shrink=0.8)
            axes[4].set_title(f"|Gen - Seed orig| MAE={mae_val:.2f}, Max={max_err:.2f}", fontsize=9)
            axes[4].set_xlabel("X"); axes[4].set_ylabel("Z")

            pcm5 = axes[5].imshow(error_recon_gen, cmap="turbo", aspect="auto", origin="lower",
                                  vmin=0, vmax=max(max_rg, 1e-6), extent=ext)
            fig.colorbar(pcm5, ax=axes[5], label="|Error| (dB)", shrink=0.8)
            axes[5].set_title(f"|Gen - Seed recon| MAE={mae_rg:.2f}, Max={max_rg:.2f}", fontsize=9)
            axes[5].set_xlabel("X"); axes[5].set_ylabel("Z")

            fig.suptitle(
                f"Ángulo objetivo: {target_angle:.1f}°  |  Semilla: {roi_angles[idx]:.2f}°",
                fontsize=14, fontweight="bold",
            )
            fig.tight_layout()
            png_path = os.path.join(GENERATE_PNG_FOLDER, f"plano_angulo_{target_angle:+.1f}.png")
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(
                f"  ángulo={target_angle:+7.1f}°  (semilla {roi_angles[idx]:.2f}°)"
                f"  MAE={mae_val:.2f} dB  Máx={max_err:.2f} dB  →  {mat_path}"
            )

    print("\n" + "=" * 60)
    print(f"  ¡LISTO! Resultados en:")
    print(f"  PNG: {GENERATE_PNG_FOLDER}")
    print(f"  MAT: {GENERATE_MAT_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()
