# DiT condicional para campos TL (estructura propia + datos .mat)

Este proyecto entrena un **Diffusion Transformer (DiT)** condicionado por ángulo usando tus datos de `input/`.

La implementación actual esta pensada para que `train.py` pueda ejecutarse de forma robusta en cluster:

- Carga y extrae ROIs desde `.mat`.
- Normaliza TL y angulo.
- Entrena difusion en espacio latente compacto.
- Valida contra `input/validation/` (si existe).
- Guarda checkpoint y figuras de entrenamiento/validacion.

## 1) Estructura relevante

```text
Intento_Transformers/
├── config.py
├── data_utils.py
├── model.py
├── train.py
├── generate.py
├── quick_test.py
├── train_transformer.sh
├── gen_transformer.sh
├── input/
│   ├── PlaneAngle*.mat
│   └── validation/           # opcional para validacion
└── output/
    ├── dit_model_latest.pt
    ├── train/PNG/
    ├── generate/{PNG,MAT}/
    └── validation/{PNG,MAT}/
```

## 2) Formato esperado de datos

Cada archivo `.mat` debe incluir:

- `tl` (o `TL`, `tL`)
- `X` (o `x`, `R`, `r`)
- `Z` (o `z`)

Y el nombre debe contener `PlaneAngle...` para extraer el angulo.

Ejemplos validos:

- `PlaneAngle-145.00_Length499.00.mat`
- `PlaneAngle100.00_Length499.00.mat`

## 3) Como funciona el pipeline

### 3.1 Entrenamiento (`train.py`)

1. Carga ROIs desde `input/` con los parametros de `config.py`.
2. Normaliza TL a `[0,1]` y luego a `[-1,1]`.
3. Normaliza angulos a `[0,1]`.
4. Codifica imagen a espacio latente con `LatentCodec` (determinista, sin VAE externo).
5. Aplica difusion forward `q(x_t|x_0)`.
6. DiT predice el ruido y optimiza `MSE(noise_pred, noise_target)`.
7. Guarda curva de entrenamiento en `output/train/PNG/training_curve_dit.png`.
8. Si hay `input/validation/`, genera muestras condicionadas por angulo y compara con reales.
9. Guarda checkpoint final en `output/dit_model_latest.pt`.

### 3.2 Validacion (integrada en `train.py`)

Si existe `input/validation/`:

- Se leen planos de validacion con la misma logica ROI.
- Se genera un plano por cada angulo de validacion (hasta `NUM_VALIDATION_ANGLES`).
- Se calculan metricas: MAE, RMSE, MAPE y error maximo.
- Se guardan:
  - `output/validation/PNG/validation_real_vs_generated.png`
  - `output/validation/MAT/val_generated_angle_*.mat`

### 3.3 Generacion (`generate.py`)

1. Carga checkpoint (`dit_model_latest.pt`).
2. Reconstruye arquitectura desde metadatos guardados.
3. Genera para angulos en `GENERATE_ANGLES` con DDIM.
4. Guarda:
   - `output/generate/MAT/plano_angulo_*.mat`
   - `output/generate/PNG/plano_angulo_*.png`

## 4) Cambios importantes respecto a versiones anteriores

- Ya no depende de `vae_encoder/vae_decoder` externos dentro de `train.py`.
- DiT usa **tokenizacion por parches** (`MODEL_PATCH_SIZE`) para evitar explosion de memoria por atencion cuadratica.
- `generate.py` y `quick_test.py` usan el mismo formato de checkpoint que `train.py`.
- Se elimina la incoherencia entre firmas/argumentos del modelo y scripts.

## 5) Parametros clave en `config.py`

### Datos

- `ROI_HEIGHT`, `ROI_WIDTH`
- `ROI_MODE`: `"center_max"` o `"corner_fixed"`
- `ROI_CORNER` (si `corner_fixed`)
- `ROIS_PER_PLANE`

### Difusion

- `DIFFUSION_STEPS`
- `BETA_START`, `BETA_END`

### DiT

- `MODEL_HIDDEN_SIZE`
- `MODEL_DEPTH`
- `MODEL_NUM_HEADS`
- `MLP_RATIO`
- `MODEL_PATCH_SIZE`
- `ATTENTION_CHUNK_SIZE` (atencion por bloques para reducir pico de memoria en ROCm)

### Espacio latente

- `VAE_COMPRESSION_RATIO`
- `VAE_LATENT_CHANNELS`

Nota: se mantiene el prefijo `VAE_*` por compatibilidad, pero el codec actual es determinista (downsample/upsample).

### Entrenamiento

- `BATCH_SIZE`
- `GRAD_ACCUM_STEPS`
- `EPOCHS`
- `LEARNING_RATE`
- `WARMUP_STEPS`
- `WEIGHT_DECAY`
- `EMA_DECAY`
- `USE_AUGMENTATION`

### Validacion / generacion

- `NUM_VALIDATION_ANGLES`
- `VALIDATION_DDIM_STEPS`
- `GENERATION_DDIM_STEPS`
- `GENERATE_ANGLES`

## 6) Ejecucion en cluster (Slurm)

No ejecutes en local si tu flujo va por Slurm.

### Entrenar

```bash
sbatch train_transformer.sh
```

### Generar

```bash
sbatch gen_transformer.sh
```

Los logs quedan en `logs/` segun la configuracion de los scripts.

## 7) Archivos de salida esperados

Tras entrenamiento:

- `output/dit_model_latest.pt`
- `output/train/PNG/training_curve_dit.png`
- `output/validation/PNG/validation_real_vs_generated.png` (si hay validacion)
- `output/validation/MAT/val_generated_angle_*.mat` (si hay validacion)

Tras generacion:

- `output/generate/PNG/plano_angulo_*.png`
- `output/generate/MAT/plano_angulo_*.mat`

## 8) Troubleshooting rapido

### No encuentra ROIs

- Revisa `DATA_FOLDER`.
- Revisa que los `.mat` contengan `tl`, `X`/`R`, `Z`.
- Revisa `ROI_MODE`, `ROI_CORNER`, `ROI_HEIGHT`, `ROI_WIDTH`.

### OOM en GPU

- Baja `BATCH_SIZE`.
- Baja `MODEL_HIDDEN_SIZE` o `MODEL_DEPTH`.
- Sube `MODEL_PATCH_SIZE` (menos tokens).
- Sube `VAE_COMPRESSION_RATIO` (latente mas pequeno).

### Generacion incoherente

- Verifica que el checkpoint sea de este pipeline (`dit_model_latest.pt` reciente).
- Asegura que el rango angular de entrenamiento cubra los angulos objetivo.
