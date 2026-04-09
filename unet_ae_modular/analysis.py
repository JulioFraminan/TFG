import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from data_utils import load_inference_pipeline, adaptive_figsize
from config import (
    ROI_HEIGHT, ROI_WIDTH,
    ROIS_PER_PLANE,
    ROI_MODE, ROI_CORNER,
    DATA_FOLDER, MODEL_PATH,
    ANALYSIS_PNG_FOLDER,
    create_output_dirs,
)


def main():
    create_output_dirs(ANALYSIS_PNG_FOLDER)

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
    #  INTERPOLACIÓN Y CROSSOVER DE FEATURES
    # ══════════════════════════════════════════════════════════════════════
    if len(rois_norm) >= 2:
        with torch.no_grad():
            sa = torch.from_numpy(rois_norm[0][np.newaxis, np.newaxis, :, :]).to(device)
            sb = torch.from_numpy(rois_norm[1][np.newaxis, np.newaxis, :, :]).to(device)
            ca = torch.tensor([[angles_norm[0]]], dtype=torch.float32, device=device)
            cb = torch.tensor([[angles_norm[1]]], dtype=torch.float32, device=device)

            e1a, e2a, e3a, ba = model.encode(sa)
            e1b, e2b, e3b, bb = model.encode(sb)

            # --- Interpolación latente ---
            print("\nInterpolación latente entre plano 1 y plano 2...")
            n_interp = 8
            ext_a = roi_extents[0]
            ext_b = roi_extents[1]
            ext_interp = [
                min(ext_a[0], ext_b[0]), max(ext_a[1], ext_b[1]),
                min(ext_a[2], ext_b[2]), max(ext_a[3], ext_b[3]),
            ]
            fig, axes = plt.subplots(
                1, n_interp,
                figsize=adaptive_figsize(ROI_HEIGHT, ROI_WIDTH, 1, n_interp),
            )
            for k, alpha in enumerate(np.linspace(0, 1, n_interp)):
                e1_mix = (1 - alpha) * e1a + alpha * e1b
                e2_mix = (1 - alpha) * e2a + alpha * e2b
                e3_mix = (1 - alpha) * e3a + alpha * e3b
                b_mix  = (1 - alpha) * ba  + alpha * bb
                c_mix  = (1 - alpha) * ca  + alpha * cb

                img = model.decode(e1_mix, e2_mix, e3_mix, b_mix, c_mix)
                img_tl = norm.denormalize_tl(
                    np.clip(img.cpu().squeeze().numpy(), 0, 1)
                )
                ang_mix = (1 - alpha) * roi_angles[0] + alpha * roi_angles[1]

                axes[k].imshow(
                    img_tl, cmap="jet", aspect="auto", origin="lower",
                    vmin=norm.tl_min, vmax=norm.tl_max, extent=ext_interp,
                )
                axes[k].set_title(f"a={alpha:.2f}\n{ang_mix:.1f}", fontsize=8)
                axes[k].set_xlabel("X"); axes[k].set_ylabel("Z")

            fig.suptitle(
                f"Interpolación: {roi_angles[0]:.1f}° → {roi_angles[1]:.1f}°",
                fontsize=13,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(ANALYSIS_PNG_FOLDER, "interpolacion_latente.png"), dpi=150)
            plt.close(fig)
            print("Interpolación latente guardada.")

            # --- Crossover de features ---
            print("\nMezcla de planos (crossover de features)...")
            combos = [
                (f"Feat_A + áng_B ({roi_angles[1]:.1f}°)", e1a, e2a, e3a, ba, cb),
                (f"Feat_B + áng_A ({roi_angles[0]:.1f}°)", e1b, e2b, e3b, bb, ca),
                (f"BN_A + skip_B + áng_A",                 e1b, e2b, e3b, ba, ca),
                (f"BN_B + skip_A + áng_B",                 e1a, e2a, e3a, bb, cb),
            ]

            n_plots = len(combos) + 2
            fig, axes = plt.subplots(
                1, n_plots,
                figsize=adaptive_figsize(ROI_HEIGHT, ROI_WIDTH, 1, n_plots),
            )

            orig_a_tl = norm.denormalize_tl(rois_norm[0])
            orig_b_tl = norm.denormalize_tl(rois_norm[1])

            axes[0].imshow(
                orig_a_tl, cmap="jet", aspect="auto", origin="lower",
                vmin=norm.tl_min, vmax=norm.tl_max, extent=roi_extents[0],
            )
            axes[0].set_title(f"A ({roi_angles[0]:.1f})", fontsize=8)
            axes[0].set_xlabel("X"); axes[0].set_ylabel("Z")

            axes[-1].imshow(
                orig_b_tl, cmap="jet", aspect="auto", origin="lower",
                vmin=norm.tl_min, vmax=norm.tl_max, extent=roi_extents[1],
            )
            axes[-1].set_title(f"B ({roi_angles[1]:.1f})", fontsize=8)
            axes[-1].set_xlabel("X"); axes[-1].set_ylabel("Z")

            for idx, (label, e1, e2, e3, b, c) in enumerate(combos):
                img = model.decode(e1, e2, e3, b, c).cpu().squeeze().numpy()
                img_tl = norm.denormalize_tl(np.clip(img, 0, 1))
                axes[idx + 1].imshow(
                    img_tl, cmap="jet", aspect="auto", origin="lower",
                    vmin=norm.tl_min, vmax=norm.tl_max, extent=roi_extents[0],
                )
                axes[idx + 1].set_title(label, fontsize=7)
                axes[idx + 1].set_xlabel("X"); axes[idx + 1].set_ylabel("Z")

            fig.suptitle("Crossover de features — Cond. UNet AE", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(ANALYSIS_PNG_FOLDER, "crossover_features.png"), dpi=150)
            plt.close(fig)
            print("Crossover de features guardado.")
    else:
        print("Se necesitan al menos 2 planos para interpolar y crossover.")

    print("\n" + "=" * 60)
    print(f"  ¡Análisis completado! Figuras en:\n  {ANALYSIS_PNG_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()
