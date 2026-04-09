import os

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE DATOS
# ══════════════════════════════════════════════════════════════════════════
ROI_HEIGHT     = 700
ROI_WIDTH      = 2000
ROIS_PER_PLANE = 1
ROI_MODE = "corner_fixed"
ROI_CORNER = (0, 10)

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE DIFUSIÓN
# ══════════════════════════════════════════════════════════════════════════
DIFFUSION_STEPS    = 1000
DIFFUSION_SCHEDULE = "linear"
BETA_START = 0.0001
BETA_END   = 0.02

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DEL MODELO TRANSFORMER (DiT)
# ══════════════════════════════════════════════════════════════════════════
MODEL_HIDDEN_SIZE = 320
MODEL_DEPTH       = 8
MODEL_NUM_HEADS   = 8
MLP_RATIO         = 4.0
MODEL_PATCH_SIZE  = 3
ATTENTION_CHUNK_SIZE = 256

# Codec latente determinista (downsample/upsample)
# Se mantiene el prefijo VAE_* para compatibilidad con checkpoints previos.
VAE_LATENT_CHANNELS = 1
VAE_COMPRESSION_RATIO = 6

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════
BATCH_SIZE         = 2
GRAD_ACCUM_STEPS   = 2
EPOCHS             = 400
LEARNING_RATE      = 2e-4
WARMUP_STEPS       = 400
WEIGHT_DECAY       = 1e-4
EMA_DECAY          = 0.9995
USE_AUGMENTATION   = True

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE VALIDACIÓN Y GENERACIÓN
# ══════════════════════════════════════════════════════════════════════════
VALIDATION_EVERY   = 50
NUM_VALIDATION_ANGLES = 10
VALIDATION_DDIM_STEPS = 220
GENERATION_DDIM_STEPS = 280
GENERATE_ANGLES    = [95, 100, 110, 120, 130, 140, 150, 160, 170, 180]

# ══════════════════════════════════════════════════════════════════════════
#  RUTAS Y DIRECTORIOS
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "input")
VALIDATION_FOLDER = os.path.join(DATA_FOLDER, "validation")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

TRAIN_PNG_FOLDER      = os.path.join(OUTPUT_FOLDER, "train", "PNG")
TRAIN_MAT_FOLDER      = os.path.join(OUTPUT_FOLDER, "train", "MAT")
GENERATE_PNG_FOLDER   = os.path.join(OUTPUT_FOLDER, "generate", "PNG")
GENERATE_MAT_FOLDER   = os.path.join(OUTPUT_FOLDER, "generate", "MAT")
VALIDATION_PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "validation", "PNG")
VALIDATION_MAT_FOLDER = os.path.join(OUTPUT_FOLDER, "validation", "MAT")

DIT_MODEL_PATH      = os.path.join(OUTPUT_FOLDER, "dit_model_latest.pt")
# Ruta legacy (no usada por el pipeline DiT actual).
VAE_ENCODER_PATH    = os.path.join(OUTPUT_FOLDER, "vae_encoder.pt")


def create_output_dirs(*folders):
    for f in folders:
        os.makedirs(f, exist_ok=True)


def create_all_dirs():
    create_output_dirs(
        TRAIN_PNG_FOLDER, TRAIN_MAT_FOLDER,
        GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER,
        VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER
    )


def verify_config(checkpoint):
    checks = [
        ("ROI_HEIGHT", checkpoint.get("roi_height"), ROI_HEIGHT),
        ("ROI_WIDTH",  checkpoint.get("roi_width"),  ROI_WIDTH),
    ]
    for name, saved, current in checks:
        if saved is not None and saved != current:
            print(f"  [!] AVISO: {name} mismatched")
