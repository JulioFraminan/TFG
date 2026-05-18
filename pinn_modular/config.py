import math
import os

# ROI parameters
ROI_HEIGHT = 480
ROI_WIDTH = 480
ROIS_PER_PLANE = 1

# ROI mode: "center_max" or "corner_fixed"
ROI_MODE = "corner_fixed"
ROI_CORNER = (50, -5)
USE_BLOCK_VARIABLES = True

# Training parameters
SEED = 42
BATCH_SIZE = 2048
EPOCHS = 400
LEARNING_RATE = 1e-4
USE_AUGMENTATION = False
POINTS_PER_ROI = 20000

# Physics loss parameters
PHYSICS_BATCH_SIZE = 4096
DATA_WEIGHT = 1.0
PHYSICS_WEIGHT = 0.01
INTERFACE_BATCH_SIZE = 2048
INTERFACE_WEIGHT = 0.1
PDE_TYPE = "two_layer_helmholtz"  # "none", "laplace", "helmholtz", "two_layer_helmholtz"

# Two-medium acoustics (air/water)
FREQUENCY_HZ = 1000.0
AIR_SOUND_SPEED = 343.0
AIR_DENSITY = 1.225
WATER_SOUND_SPEED = 1480.0
WATER_DENSITY = 1000.0
AIR_ABOVE_INTERFACE = True
INTERFACE_Z = 0.0  # None -> auto midpoint of ROI bounds
INTERFACE_EPS = 0.5
TL_REF_PRESSURE = 1.0
OUTPUT_IS_TL = True
TL_DB_SIGN = -1.0
TL_DB_OFFSET = 0.0
TL_DB_CLAMP_MIN = 0.0
TL_DB_CLAMP_MAX = 100.0

# Single-medium Helmholtz (PDE_TYPE = "helmholtz")
if INTERFACE_Z is None:
    _interface_z_ref = 0.0
else:
    _interface_z_ref = float(INTERFACE_Z)
HELMHOLTZ_K_MEDIUM = "water" if ROI_CORNER[1] < _interface_z_ref else "air"
HELMHOLTZ_K_AIR = 2.0 * math.pi * FREQUENCY_HZ / AIR_SOUND_SPEED
HELMHOLTZ_K_WATER = 2.0 * math.pi * FREQUENCY_HZ / WATER_SOUND_SPEED
HELMHOLTZ_K = HELMHOLTZ_K_AIR if HELMHOLTZ_K_MEDIUM == "air" else HELMHOLTZ_K_WATER

# PINN model parameters
PINN_HIDDEN_DIM = 128
PINN_NUM_LAYERS = 6
PINN_ACTIVATION = "tanh"
PINN_DROPOUT = 0.0

# Inference parameters
INFER_BATCH_SIZE = 20000
GENERATE_ANGLES = [-180]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "input")
VALIDATION_FOLDER = os.path.join(DATA_FOLDER, "validation")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

TRAIN_PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "train", "PNG")
TRAIN_MAT_FOLDER = os.path.join(OUTPUT_FOLDER, "train", "MAT")
GENERATE_PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "generate", "PNG")
GENERATE_MAT_FOLDER = os.path.join(OUTPUT_FOLDER, "generate", "MAT")
VALIDATION_PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "validation", "PNG")
VALIDATION_MAT_FOLDER = os.path.join(OUTPUT_FOLDER, "validation", "MAT")

MODEL_PATH = os.path.join(OUTPUT_FOLDER, "pinn_model.pt")


def create_output_dirs(*folders):
    for f in folders:
        os.makedirs(f, exist_ok=True)


def verify_config(checkpoint):
    checks = [
        ("ROI_HEIGHT", checkpoint.get("roi_height"), ROI_HEIGHT),
        ("ROI_WIDTH", checkpoint.get("roi_width"), ROI_WIDTH),
    ]
    for name, saved, current in checks:
        if saved is not None and saved != current:
            print(f"  [warn] {name} in config ({current}) != {name} in checkpoint ({saved})")

    pde_type = checkpoint.get("pde_type")
    if pde_type is not None and pde_type != PDE_TYPE:
        print(f"  [warn] PDE_TYPE in config ({PDE_TYPE}) != checkpoint ({pde_type})")
