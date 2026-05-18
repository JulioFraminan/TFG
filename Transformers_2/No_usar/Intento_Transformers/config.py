import os

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE DATOS
# ══════════════════════════════════════════════════════════════════════════
ROI_HEIGHT     = 700
ROI_WIDTH      = 2000
ROIS_PER_PLANE = 1
ROI_MODE = "corner_fixed"
ROI_CORNER = (0, 10)
USE_BLOCK_VARIABLES = True

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
MODEL_HIDDEN_SIZE = 448        # Intermediate: 384 → 512 sweet spot
MODEL_DEPTH       = 11         # 10.5 equivalent
MODEL_NUM_HEADS   = 14         # 32 dim/head (good for 448)
MLP_RATIO         = 3.0        # Between 3.0 and 4.0
MODEL_PATCH_SIZE  = 2
ATTENTION_CHUNK_SIZE = 448     # Match hidden size
GRADIENT_CHECKPOINTING = True  # ↑ Enabled (saves ~30% intermediate activations)

# Codec latente determinista (downsample/upsample)
# Se mantiene el prefijo VAE_* para compatibilidad con checkpoints previos.
USE_VAE = True
VAE_LATENT_CHANNELS = 1
VAE_COMPRESSION_RATIO = 6      # Balance: detail vs tokens (39k tokens)

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════
BATCH_SIZE         = 4         # Safe batch size
GRAD_ACCUM_STEPS   = 1         # No accumulation (batch 4 is good enough)
EPOCHS             = 100
LEARNING_RATE      = 1.5e-4
WARMUP_STEPS       = 400
WEIGHT_DECAY       = 1e-4
EMA_DECAY          = 0.9997
USE_AUGMENTATION   = False

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE VALIDACIÓN Y GENERACIÓN
# ══════════════════════════════════════════════════════════════════════════
VALIDATION_EVERY   = 50
NUM_VALIDATION_ANGLES = 10
VALIDATION_DDIM_STEPS = 150    # Balance: quality vs speed (10 angles × 150 = 1500 steps total)
GENERATION_DDIM_STEPS = 350    # Higher for final generation (better quality)
GENERATE_ANGLES    = [95, 100, 110, 120, 130, 140, 150, 160, 170, 180]

# ══════════════════════════════════════════════════════════════════════════
#  RUTAS Y DIRECTORIOS
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAE_CHECKPOINT_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "unet_ae_modular", "output", "unet_ae_model.pt")
)
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
