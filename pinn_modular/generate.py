import os

import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_inference_pipeline, save_mat, adaptive_figsize, predict_on_grid, compute_error_metrics
from config import (
    ROI_HEIGHT, ROI_WIDTH,
    ROIS_PER_PLANE, GENERATE_ANGLES,
    ROI_MODE, ROI_CORNER,
    DATA_FOLDER, MODEL_PATH,
    GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER,
    INFER_BATCH_SIZE,
    create_output_dirs,
)


def main():
    create_output_dirs(GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER)

    device, model, rois, roi_angles, norm, coord_norm, roi_extents = load_inference_pipeline(
        MODEL_PATH, DATA_FOLDER,
        ROI_HEIGHT, ROI_WIDTH,
        ROIS_PER_PLANE,
        roi_mode=ROI_MODE,
        roi_corner=ROI_CORNER,
    )

    roi_angles_arr = np.array(roi_angles)

    print("\n" + "=" * 60)
    print(f"  Generating planes for {len(GENERATE_ANGLES)} angles")
    print("=" * 60)

    for target_angle in GENERATE_ANGLES:
        idx = int(np.argmin(np.abs(roi_angles_arr - target_angle)))
        extent = roi_extents[idx]
        seed_tl = rois[idx]

        pred_norm = predict_on_grid(
            model, coord_norm, target_angle, extent, seed_tl.shape, device,
            batch_size=INFER_BATCH_SIZE,
        )
        pred_tl = norm.denormalize_tl(np.clip(pred_norm, 0, 1))

        errors = compute_error_metrics(seed_tl, pred_tl)
        err_map = np.abs(seed_tl - pred_tl)

        mat_path = os.path.join(GENERATE_MAT_FOLDER, f"plane_angle_{target_angle:+.1f}.mat")
        save_mat(mat_path, pred_tl, extent=extent)

        fig, axes = plt.subplots(
            1, 3,
            figsize=adaptive_figsize(ROI_HEIGHT, ROI_WIDTH, 1, 3),
        )

        im0 = axes[0].imshow(
            pred_tl, cmap="jet", aspect="auto", origin="lower",
            vmin=norm.tl_min, vmax=norm.tl_max, extent=extent,
        )
        fig.colorbar(im0, ax=axes[0], label="TL", shrink=0.8)
        axes[0].set_title(f"Predicted - {target_angle:.1f} deg", fontsize=10)

        im1 = axes[1].imshow(
            seed_tl, cmap="jet", aspect="auto", origin="lower",
            vmin=norm.tl_min, vmax=norm.tl_max, extent=extent,
        )
        fig.colorbar(im1, ax=axes[1], label="TL", shrink=0.8)
        axes[1].set_title(f"Nearest train - {roi_angles[idx]:.2f} deg", fontsize=10)

        im2 = axes[2].imshow(
            err_map, cmap="turbo", aspect="auto", origin="lower",
            vmin=0, vmax=errors["max_error"] or 1e-6, extent=extent,
        )
        fig.colorbar(im2, ax=axes[2], label="abs error", shrink=0.8)
        axes[2].set_title(
            f"Error | MAE={errors['mae']:.2f} MAPE={errors['mape']:.1f}%",
            fontsize=9,
        )

        fig.suptitle(f"Target angle: {target_angle:.1f} deg", fontsize=12)
        fig.tight_layout()
        png_path = os.path.join(GENERATE_PNG_FOLDER, f"plane_angle_{target_angle:+.1f}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(
            f"  angle={target_angle:+7.1f} deg | nearest={roi_angles[idx]:.2f} deg "
            f"| MAE={errors['mae']:.2f} | {mat_path}"
        )

    print("\n" + "=" * 60)
    print("  Done")
    print(f"  PNG: {GENERATE_PNG_FOLDER}")
    print(f"  MAT: {GENERATE_MAT_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()
