# Diffusion Transformer (DiT) para generación condicionada de planos TL

**Generador de campos de *Transmission Loss* (TL) acústico** usando **Transformers de Difusión** condicionados por ángulo.

## Características principales

- 🎯 **Condicionamiento por ángulo** para control preciso
- ⚡ **Optimizado para AMD MI210** (ROCm): 80% VRAM, ~6h × 220 épocas
- 🖼️ **Alta calidad visual** con DDIM iterativo (150-350 pasos)
- 🔄 **Tokenización por parches** para estabilidad de memoria
- 💾 **Determinístico en latent space** (sin VAE externo)
- ✅ **Validación integrada** contra planos reales

---

## 1) Estructura del proyecto

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

- ✅ Ya no depende de `vae_encoder/vae_decoder` externos
- ✅ DiT usa **tokenización por parches** para estabilidad de memoria
- ✅ `LatentCodec` determinístico (downsample/upsample simple)
- ✅ `generate.py` y `quick_test.py` usan mismo checkpoint que `train.py`
- ✅ **OOM recovery automático** con reducción dinámica de attention chunk
- ✅ **Gradient checkpointing** activado por defecto para ahorrar VRAM
- ✅ **Optimización de batch** y acumulación de gradientes

---

## 9) Guía práctica rápida

## 5) Parámetros actuales optimizados

**Configuración de Abril 2026** - Optimizado para **AMD MI210 (68.7 GB VRAM)**

### Datos

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `ROI_HEIGHT` | 700 | Alto del ROI extraído |
| `ROI_WIDTH` | 2000 | Ancho del ROI extraído |
| `ROI_MODE` | "corner_fixed" | Esquina fija vs. centrado en máximo |
| `ROI_CORNER` | (0, 10) | Esquina física (X, Z) |
| `ROIS_PER_PLANE` | 1 | ROIs por plano |

### Diffusion

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `DIFFUSION_STEPS` | 1000 | Pasos en schedule lineal |
| `BETA_START` | 1e-4 | Beta inicio |
| `BETA_END` | 0.02 | Beta final |

### Arquitectura DiT (OPTIMIZADA)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `MODEL_HIDDEN_SIZE` | 448 | Dimensión oculta (↑ desde 384) |
| `MODEL_DEPTH` | 11 | Número de bloques (↑ desde 10) |
| `MODEL_NUM_HEADS` | 14 | Heads de atención | 
| `MLP_RATIO` | 3.5 | FFN hidden size ratio |
| `MODEL_PATCH_SIZE` | 2 | Patch size para tokenización |
| `ATTENTION_CHUNK_SIZE` | 448 | Chunk size (↑ desde 64) |
| `GRADIENT_CHECKPOINTING` | True | Ahorro 30% activaciones |

### Espacio Latente

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `VAE_COMPRESSION_RATIO` | 6 | Downsample (700×2000 → 117×334) |
| `VAE_LATENT_CHANNELS` | 1 | Canales latentes |

### Entrenamiento

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `BATCH_SIZE` | 4 | Tamaño batch (↑ desde 1) |
| `GRAD_ACCUM_STEPS` | 1 | Acumulación gradientes |
| `EPOCHS` | 220 | Épocas totales |
| `LEARNING_RATE` | 1.5e-4 | Initial LR |
| `WARMUP_STEPS` | 400 | Warmup steps |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `EMA_DECAY` | 0.9997 | EMA decay para validación |

### Validación / Generación

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `VALIDATION_EVERY` | 50 | Validar cada N épocas |
| `NUM_VALIDATION_ANGLES` | 10 | Ángulos a validar |
| `VALIDATION_DDIM_STEPS` | 150 | Pasos DDIM validación (↓ desde 400) |
| `GENERATION_DDIM_STEPS` | 350 | Pasos DDIM generación |

### Rendimiento Actual

```
Modelo:              38.5M parámetros (↓ desde 57M)
VRAM utilizado:      ~80% en MI210 (55-56 GB)
Tiempo por época:    ~1.6 minutos
Tiempo 220 épocas:   ~5h 55m (↓ desde 9h+)
Batchs por época:    10 (38 ROIs / batch 4)
Latent shape:        117×334×1 (39k tokens)

Speedup:             +50-70% vs baseline (batch=1, hidden=384)
```

---

## 6) Validación e Iteración Rápida

### Si quieres iterar rápido (prototipado):

```python
# Reducir en config.py para pruebas:
EPOCHS = 20                    # vs 220
NUM_VALIDATION_ANGLES = 3      # vs 10
VALIDATION_DDIM_STEPS = 50     # vs 150
```

### Si necesitas máxima calidad:

```python
# Aumentar para mejor resultado:
GENERATION_DDIM_STEPS = 500    # vs 350
MODEL_HIDDEN_SIZE = 512        # vs 448
MODEL_DEPTH = 12               # vs 11
```

---

## 7) OOM Recovery automático

Si durante entrenamiento se produce `OutOfMemoryError`:

1. El código intenta reducir `ATTENTION_CHUNK_SIZE` automáticamente
2. Si falla, activa `GRADIENT_CHECKPOINTING`
3. Si sigue fallando, reducir `BATCH_SIZE` o aumentar `GRAD_ACCUM_STEPS` en config.py

**Para evitar OOM desde el inicio:**
```python
BATCH_SIZE = 2              # Menor
ATTENTION_CHUNK_SIZE = 320  # Menor
GRADIENT_CHECKPOINTING = True
```

---

## 8) Cambios importantes respecto a versiones anteriores

## 9) Guía práctica rápida

### Paso 1: Preparar datos

```bash
# Copiar datos a input/
cp /tu/path/*.mat input/

# (Opcional) Crear data de validación
mkdir -p input/validation
cp /tu/path/validation/*.mat input/validation/
```

### Paso 2: Entrenar

```bash
# Entrenar con configuración actual (220 épocas, ~6h)
python train.py

# O entrenar rápido para pruebas (20 épocas, ~15min)
# Editar config.py: EPOCHS = 20, VALIDATION_DDIM_STEPS = 50
python train.py
```

### Paso 3: Generar predicciones

```bash
# Generar con modelo entrenado
python generate.py

# Resultados en output/generate/{PNG,MAT}/
```

### Paso 4: Visualizar resultados

```bash
# Ver curva de entrenamiento
open output/train/PNG/training_curve_dit.png

# Ver validación (si se ejecutó)
open output/validation/PNG/validation_real_vs_generated.png

# Ver generación
open output/generate/PNG/plano_angulo_*.png
```

---

## 10) Comparativa con unet_ae_modular

## 10) Comparativa: DiT vs UNet Autoencoder

| Característica | **DiT (este modelo)** | **UNet AE** |
|---|---|---|
| **Tipo** | Generativo (Diffusion) | Determinista (Autoencoder) |
| **Velocidad entrenamiento** | Lenta (6h × 220 épocas) | Rápida (30min × 70 épocas) |
| **Velocidad generación** | LENTA (150-350 pasos DDIM) | **Muy rápida** (1 forward) |
| **Calidad visual** | ⭐⭐⭐⭐⭐ Excelente | ⭐⭐⭐⭐ Muy buena |
| **Pixelado** | Bajo con DDIM steps altos | Posible (interpolación) |
| **Variabilidad** | Genera variantes diferentes | Determinista |
| **VRAM** | 80% (alto) | < 30% (bajo) |
| **Mejor para** | Máxima calidad, investigación | Prototipado, producción rápida |

**Recomendación:**
- Usa **DiT** si: Necesitas máxima calidad, tiempo no es limitante
- Usa **UNet AE** si: Necesitas iterar rápido, generar muchas muestras

---

## 11) Archivos de salida esperados

Tras entrenamiento:

```
output/
├── dit_model_latest.pt                          ← Checkpoint entrenado
├── train/
│   └── PNG/
│       ├── training_curve_dit.png               ← Curva pérdida/LR
│       └── ...
├── validation/                                  (si hay datos en input/validation/)
│   ├── PNG/
│   │   └── validation_real_vs_generated.png     ← Comparativa real vs gen
│   └── MAT/
│       └── val_generated_angle_*.mat
└── generate/                                    (después de python generate.py)
    ├── PNG/
    │   └── plano_angulo_*.png
    └── MAT/
        └── plano_angulo_*.mat
```

---

## 12) Troubleshooting

### No encuentra ROIs

- Revisa `DATA_FOLDER`.
- Revisa que los `.mat` contengan `tl`, `X`/`R`, `Z`.
- Revisa `ROI_MODE`, `ROI_CORNER`, `ROI_HEIGHT`, `ROI_WIDTH`.

### OOM en GPU

- Baja `BATCH_SIZE`.
- Baja `MODEL_HIDDEN_SIZE` o `MODEL_DEPTH`.
- Sube `MODEL_PATCH_SIZE` (menos tokens).
- Sube `VAE_COMPRESSION_RATIO` (latente mas pequeno).
- Activa `GRADIENT_CHECKPOINTING=True`.
- Baja `ATTENTION_CHUNK_SIZE`.
- El entrenamiento ahora tiene recovery automatico de OOM en forward y backward:
  - reduce `attention runtime chunk` en caliente,
  - activa checkpointing si hace falta,
  - y divide el batch en micro-batches de forma adaptativa.
- En Slurm ROCm usa `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` para reducir fragmentacion.

Perfil recomendado (punto medio):

- `BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=3`
- `ATTENTION_CHUNK_SIZE=320` (suele estabilizarse en runtime alrededor de `40` tras recovery)
- `GRADIENT_CHECKPOINTING=False` para mas velocidad, con fallback automatico a `True` si hay OOM

### Salida pixelada

- Baja `MODEL_PATCH_SIZE` (2 suele verse bastante mejor que 3-4).
- Sube `GENERATION_DDIM_STEPS` y `VALIDATION_DDIM_STEPS`.
- Mantén `VAE_COMPRESSION_RATIO` moderado (demasiada compresion aumenta artefactos).
- Usa checkpoint con EMA (ya habilitado automaticamente en generate.py).

### Generacion incoherente

- Verifica que el checkpoint sea de este pipeline (`dit_model_latest.pt` reciente).
- Asegura que el rango angular de entrenamiento cubra los angulos objetivo.
