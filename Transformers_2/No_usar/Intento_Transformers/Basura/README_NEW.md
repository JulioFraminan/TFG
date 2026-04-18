# Diffusion Transformer para Generación Condicionada de Planos Acústicos

Proyecto de **Trabajo Fin de Grado** (UPM) para generar planos de *Transmission Loss* (TL) acústica de **700×2000 píxeles** condicionados por ángulo de incidencia, empleando un **Diffusion Transformer (DiT)** con **encoder/decoder custom** para resoluciones ultrasónicas.

**Cambio principal respecto a UNet FiLM**: Reemplazo de arquitectura autorrecurrente (UNet AE) por modelo generativo con **proceso de difusión gaussiana** (1000 pasos) y **transformers** de 28 bloques con **attention adaptativo condicionado** (adaLN-Zero).

---

## Estructura del proyecto

```
Intento_Transformers/                  ← Nueva implementación con DiT
├── config.py                          ← Parámetros DiT: timesteps, hidden_size, depth, etc
├── model.py                           ← DiT adaptado: PatchEmbed (700x2000), AngularEmbedder, DiTBlocks
├── vae.py                             ← Encoder/Decoder custom para 700x2000 → latents comprimidos
├── data_utils.py                      ← Carga .mat, AngularDataset, Normalizer (compatible UNet)
├── train.py                           ← Training loop DiT + difusión + angular conditioning
├── generate.py                        ← Sampling: reverse diffusion con ángulo condicionado
├── README.md
├── train_transformer.sh               ← Script de entrenamiento (DDP compatible)
├── input/                             ← Datos de entrada (mismo formato que UNet)
│   ├── PlaneAngle*.mat
│   └── validation/
├── output/                            ← (se crea automáticamente)
│   ├── dit_model_latest.pt            ← Checkpoint DiT entrenado
│   ├── vae_encoder.pt                 ← VAE encoder (fijo o entrenado)
│   ├── train/
│   │   ├── PNG/                       ← Curva de loss, samples durante training
│   │   └── MAT/
│   ├── generate/
│   │   ├── PNG/                       ← Planos generados por ángulo
│   │   └── MAT/
│   └── validation/
│       ├── PNG/
│       └── MAT/
└── DiT-main/                          ← Referencia (NO EDITAR)
    ├── models.py
    ├── diffusion/
    └── ...
```

**Nota**: La carpeta `DiT-main/` se mantiene intacta como referencia. Los archivos adaptados están en raíz.

---

## Requisitos

- **Python** ≥ 3.9
- **PyTorch** ≥ 2.0 (con CUDA recomendado para entrenamiento)
- **NVIDIA CUDA** ≥ 11.8 (recomendado para DDP)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib h5py einops tensorboard timm
```

### Datasets

Los archivos **HDF5 `.mat`** se colocan en `input/`. El nombre debe contener `PlaneAngle` seguido del ángulo:

```
PlaneAngle95.00_Length499.00.mat
PlaneAngle100.00_Length499.00.mat
PlaneAngle-170.00_Length499.00.mat
```

Cada `.mat` debe contener:
- `tl`: Matriz de Transmission Loss (altura variable, ancho variable)
- `X`: Array de coordenadas horizontales
- `Z`: Array de coordenadas verticales
- Rango típico: TL en dB, ángulos entre 95° y 180°

**ROIs extraídas**: 700×2000 (configurable en `config.py`) desde la esquina fija `(X=-1320, Z=10)`.

---

## Guía rápida

### 1. Configurar parámetros

Editar **`config.py`**:

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `ROI_HEIGHT` | Alto del parche extraído | 700 |
| `ROI_WIDTH` | Ancho del parche extraído | 2000 |
| `ROI_CORNER` | Esquina superior-izquierda (X, Z) en coords físicas | (-1320, 10) |
| `ROIS_PER_PLANE` | ROIs por archivo `.mat` | 1 |
| **DiT específicos** | | |
| `MODEL_HIDDEN_SIZE` | Dimensión del espacio oculto (DiT) | 1152 |
| `MODEL_DEPTH` | Número de bloques Transformer | 28 |
| `MODEL_NUM_HEADS` | Número de attention heads | 16 |
| `DIFFUSION_STEPS` | Pasos de difusión (T) | 1000 |
| `DIFFUSION_SCHEDULE` | "linear" o "cosine" | "linear" |
| `BATCH_SIZE` | Tamaño del batch | 8 |
| `EPOCHS` | Número de épocas | 500 |
| `LEARNING_RATE` | Tasa de aprendizaje (AdamW) | 1e-4 |
| `WARMUP_STEPS` | Steps de warmup | 1000 |
| `WEIGHT_DECAY` | L2 regularization | 1e-4 |

### 2. Entrenar desde cero (single GPU)

```bash
python train.py
```

Ejecuta en orden:
1. Carga los `.mat` de `input/` y extrae ROIs de 700×2000
2. Normaliza TL (escala [0,1]) y ángulos ([0,1])
3. Crea AngularDataset (balanceado por ángulo)
4. Entrena DiT con loss de difusión + VAE encoder
5. Guarda checkpoints en `output/dit_model_latest.pt`
6. Valida con planos de `input/validation/` cada N epochs
7. Exporta muestras de validación como PNG y `.mat`

### 2b. Entrenar con DDP (multi-GPU)

```bash
torchrun --nproc_per_node=4 --master_port=29500 train.py
```

### 3. Generar planos

```bash
python generate.py --angles 95 100 110 120 130 140 150 160 170 180
```

- Carga `output/dit_model_latest.pt` y VAE encoder
- Genera planos por cada ángulo especificado
- Ejecuta reverse diffusion (1000 pasos)
- Exporta `.mat` y comparativas PNG

### 4. Análisis avanzado (futuro)

```bash
python analysis.py --mode interpolation --angle1 95 --angle2 180 --steps 20
```

Interpolación suave en espacio de ángulos condicionados.

---

## Resumen de uso

| Situación | Comando |
|---|---|
| Primera vez | `python train.py` → esperar 500 epochs → `python generate.py --angles ...` |
| Ya tienes `.pt` | `python generate.py --angles 95 100 ... 180` |
| Reentrenar desde checkpoint | `python train.py --resume output/dit_model_latest.pt` |
| Multi-GPU | `torchrun --nproc_per_node=4 train.py` |

---

## Arquitectura del modelo

### Diffusion Transformer (DiT) Condicionado por Ángulo

#### Pipeline general
```
1. CODIFICACIÓN (Offline, una sola vez):
   ROI (700×2000) → VAE_Encoder → Latent (z) [C×H'×W' comprimido]

2. ENTRENAMIENTO (Forward diffusion + Reverse con Transformer):
   
   t ~ Uniform(0, T)
   z_t = √(ᾱ_t) · z_0 + √(1-ᾱ_t) · ε,  ε ~ N(0,I)
   
   x_t, t, θ_ángulo → [DiT-28 bloques] → ε_θ(x_t, t, θ)
   
   Loss = MSE(ε_θ(x_t, t, θ), ε)

3. GENERACIÓN (Reverse diffusion):
   z_T ~ N(0, I)
   for t = T, T-1, ..., 1:
     z_{t-1} = (1/√α_t) · (z_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(z_t, t, θ)) + σ_t · ε
   return z_0

4. DECODIFICACIÓN (Offline):
   z_0 → VAE_Decoder → ROI (700×2000)
```

#### Arquitectura detallada

**Entrada**: 
- `z_t`: Latent comprimido por VAE [batch, 128, H_lat, W_lat]
- `t`: Timestep diffusion [batch]
- `θ`: Ángulo normalizado [batch, 1]

**PatchEmbed**:
```
z_t (128 canales) → Cada grupo de 2×2 patches
                 → Embedding lineal → [batch, N_patches, hidden_size=1152]
```

**TimestepEmbedder**:
```
t (scalar) → Sinusoidal pos encoding → MLP(1152) → [batch, hidden_size]
```

**AngularEmbedder** (CUSTOMIZADO):
```
θ (1D, normalizado [0,1]) → MLP(1→512→1152) → [batch, hidden_size]
```

**DiTBlocks** (28 bloques):
```
Per block:
  LayerNorm (no elementwise affine)
  ├─ Attention (16 heads, 1152/16=72 dim per head)
  └─ adaLN-Zero (modulation from t_embed + θ_embed):
     γ, β = MLP(t_emb + θ_emb) → scale and shift
     output = (1 + γ) * x + β

  MLP (mlp_ratio=4.0):
     hidden_size → hidden_size*4 → hidden_size (GELU)
     
  + Residual connections + LayerNorm
```

**FinalLayer**:
```
Unpatchify latent patches → [batch, 128, H_lat, W_lat]
Conv2d(128 → 1) → prediction ε_θ (noise)
```

**VAE Encoder/Decoder**:
```
ENCODER (Offline):
700×2000 → Conv(1→64→128→256) + 4x downsampling → [128, 44, 125]
         → Conv to 128 latent channels → z [128, 44, 125] (compression ratio ~8×8)

DECODER (Reverse sampling):
z [128, 44, 125] → Deconv(128→256→128→64) + 4x upsampling → [1, 700, 2000]
                 → Sigmoid/ClipEvaluator → TL normalizado [0, 1]
```

---

## Training Loop

```python
def training_step(batch_idx, rois, angles):
    # 1. Encode ROIs al espacio latente (offline o on-the-fly)
    z_0 = vae_encoder(rois)
    
    # 2. Sample timestep aleatorio
    t = torch.randint(0, diffusion_steps, (batch_size,))
    noise = torch.randn_like(z_0)
    
    # 3. Add noise (forward diffusion)
    z_t = sqrt_alphas_cumprod[t] * z_0 + sqrt_one_minus_alphas_cumprod[t] * noise
    
    # 4. Predict noise with DiT
    predicted_noise = dit_model(z_t, t, angles)
    
    # 5. Loss
    loss = F.mse_loss(predicted_noise, noise)
    loss.backward()
    optimizer.step()
```

---

## Data augmentation

Transformaciones aplicadas únicamente si `USE_AUGMENTATION = True`:
- **Ruido gaussiano**: σ ∈ {0.01, 0.03}
- **Escalado de contraste**: factor ∈ [0.9, 1.1]
- **Offset de nivel**: offset ∈ [-0.05, 0.05]

**Factor**: ×4 multiplicador de dataset.

---

## Validación y Métricas

Durante entrenamiento cada `VALIDATION_EVERY=50` epochs:

1. **Reverse Diffusion**: Genera muestras para 10 ángulos uniformemente distribuidos
2. **Métricas**:
   - MAE (Mean Absolute Error) en dB
   - MAPE (Mean Absolute Percentage Error)
   - RMSE (Root Mean Square Error)
   - Max error en dB
3. **Outputs**:
   - PNG: Original vs Generado vs Error map
   - `.mat`: Plano completo con metadata

---

## Salidas generadas

### train.py

| Archivo | Descripción |
|---|---|
| `output/dit_model_latest.pt` | Pesos DiT + config + normalizer state |
| `output/vae_encoder.pt` | VAE encoder (fijo) |
| `output/train/PNG/loss_curve.png` | Curva de MSE loss durante entrenamiento |
| `output/train/PNG/sample_epoch_*.png` | Muestras generadas cada 50 epochs |
| `output/validation/PNG/val_angle_*.png` | Validación por ángulo |
| `output/validation/MAT/val_*.mat` | Planos de validación |

### generate.py

| Archivo | Descripción |
|---|---|
| `output/generate/PNG/angle_*.png` | 2×2 subplots: original (si existe) vs generado vs error |
| `output/generate/MAT/angle_*.mat` | Plano generado (HDF5: tl, X, Z, angle_label) |

---

## Comparación con UNet FiLM

| Aspecto | UNet FiLM | DiT (Nuevo) |
|--------|-----------|------------|
| **Tipo** | Autorreconstrucción | Modelo generativo (difusión) |
| **Entrenamiento** | L1 loss directo | MSE loss en espacio de ruido |
| **Generación** | Directo (µs) | Iterativo 1000 pasos (segundos) |
| **Condicionamiento** | FiLM (modulación lineal) | adaLN-Zero (modulación adaptativa) |
| **Escalabilidad** | O(1) arquitectura fija | O(log T) dependencia en T |
| **Mejoras esperadas** | Rápida, determinista | Muestras diversas, potencialmente mejor generalización |
| **Tiempo inferencia** | ~10 ms | ~30-60 segundos (1000 pasos) |

---

## Formato de los `.mat` de salida

Archivos **HDF5** con:

| Variable | Descripción | Shape |
|---|---|---|
| `tl` | Transmission Loss (dB, desnormalizado) | [700, 2000] |
| `X` | Coordenadas horizontales (m) | [2000,] |
| `Z` | Coordenadas verticales (m) | [700,] |
| `angle` | Ángulo condicionado (grados) | scalar |

Se abren con MATLAB o Python (`h5py`).

---

## Performance esperado

En NVIDIA A100 40GB (experimental):
- **Training**: ~500 epochs, 8 imgs/batch → 48-72 horas
- **Validation**: 10 ángulos × 1000 steps cada uno → ~2 minutos
- **Generation**: 1 ángulo × 1000 steps → ~10-15 segundos

Sobre GPU más lenta (A6000 48GB observado):
- **Training**: Misma duración, monitor de memoria activo
- **Generation**: ~20-30 segundos por ángulo

---

## Troubleshooting

### Out of Memory (VRAM)
- Reducir `BATCH_SIZE` en config.py
- Reducir `MODEL_HIDDEN_SIZE` (ej: 768 en lugar de 1152)
- Usar gradient accumulation: `GRADIENT_ACCUMULATE_STEPS=2`

### Loss no converge
- Verificar tasa de aprendizaje (intentar 5e-5)
- Aumentar warmup: `WARMUP_STEPS=5000`
- Verificar normalización de ángulos (deben estar en [0, 1])

### Muestras generadas ruidosas
- Incrementar epochs de entrenamiento
- Usar schedule "cosine" en lugar de "linear"
- Verificar VAE encoder convergió correctamente
