import os

# -----------------------------
# MODEL AND DATA PARAMETERS
# -----------------------------
ROI_HEIGHT = 480
ROI_WIDTH = 1520
ROIS_PER_PLANE = 1

# ROI definition mode:
# "center_max" -> use max value as center (original behavior)
# "corner_fixed" -> use ROI_CORNER as top-left corner in physical coords
ROI_MODE = "corner_fixed"
ROI_CORNER = (0, 10)
USE_BLOCK_VARIABLES = True

# -----------------------------
# TRAINING PARAMETERS
# -----------------------------
BATCH_SIZE = 6
EPOCHS = 800
LEARNING_RATE = 5e-5
USE_AUGMENTATION = True

# -----------------------------
# FNO PARAMETERS
# -----------------------------
FNO_MODES1 = 24
FNO_MODES2 = 24
FNO_WIDTH = 96
FNO_DEPTH = 5
FNO_USE_COORDS = True
FNO_DROPOUT = 0.0

# -----------------------------
# GENERATION PARAMETERS
# -----------------------------
NOISE_STD = 0.02
GENERATE_ANGLES = [-180]

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(BASE_DIR, "input")
VALIDATION_FOLDER = os.path.join(DATA_FOLDER, "validation")

OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

TRAIN_PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "train", "PNG")
TRAIN_MAT_FOLDER = os.path.join(OUTPUT_FOLDER, "train", "MAT")
GENERATE_PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "generate", "PNG")
GENERATE_MAT_FOLDER = os.path.join(OUTPUT_FOLDER, "generate", "MAT")
ANALYSIS_PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "analysis", "PNG")
VALIDATION_PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "validation", "PNG")
VALIDATION_MAT_FOLDER = os.path.join(OUTPUT_FOLDER, "validation", "MAT")

MODEL_PATH = os.path.join(OUTPUT_FOLDER, "fno_model.pt")


def create_output_dirs(*folders):
    """Create folders if they do not exist."""
    for f in folders:
        os.makedirs(f, exist_ok=True)


def verify_config(checkpoint):
    """Compare saved parameters in .pt with config.py."""
    checks = [
        ("ROI_HEIGHT", checkpoint.get("roi_height"), ROI_HEIGHT),
        ("ROI_WIDTH", checkpoint.get("roi_width"), ROI_WIDTH),
        ("FNO_MODES1", checkpoint.get("fno_modes1"), FNO_MODES1),
        ("FNO_MODES2", checkpoint.get("fno_modes2"), FNO_MODES2),
        ("FNO_WIDTH", checkpoint.get("fno_width"), FNO_WIDTH),
        ("FNO_DEPTH", checkpoint.get("fno_depth"), FNO_DEPTH),
        ("FNO_USE_COORDS", checkpoint.get("fno_use_coords"), FNO_USE_COORDS),
    ]
    for name, saved, current in checks:
        if saved is not None and saved != current:
            print(
                f"  [WARN] {name} in config ({current}) != "
                f"{name} in .pt ({saved})"
            )

    for key, label in [
        ("angle_min", "ANGLE_MIN"),
        ("angle_max", "ANGLE_MAX"),
        ("tl_min", "TL_MIN"),
        ("tl_max", "TL_MAX"),
    ]:
        val = checkpoint.get(key)
        if val is not None:
            print(f"  [i] {label} saved in .pt: {val:.4f}")
