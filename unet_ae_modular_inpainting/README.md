# UNet Autoencoder Condicional + Inpainting — Generación rápida con relleno de datos

**Generador robusto con inpainting** de campos de *Transmission Loss* (TL) acústico usando autoencoder **UNet** + modulación **FiLM** condicionada por ángulo, con capacidad de **rellenar regiones faltantes (huecos)** durante entrenamiento e inferencia.

**⚡ Ventajas principales:**
- Entrenamiento rápido (~30 minutos, 70 épocas)
- Generación en 1 forward pass (< 1 segundo)
- **Inpainting integrado**: completa datos incompletos automáticamente
- VRAM bajo (< 30% típicamente)
- Ideal para prototipado e iteración rápida
- Determinista y reproducible

**⚡ Ventajas principais:**
- Entrenamiento rápido (~30 minutos, 70 épocas)
- Generación en 1 forward pass (< 1 segundo)
- VRAM bajo (< 30% típicamente)
- Ideal para prototipado e iteración rápida
- Determinista y reproducible

---

## 📊 Comparativa: UNet AE vs DiT Diffusion

| Característica | **UNet AE (este)** | **DiT (Diffusion)** |
|---|---|---|
| **Tipo** | Determinista | Generativo (iterativo) |
| **Velocidad entrenamiento** | ⚡ Rápida (30min) | Lenta (6 horas) |
| **Velocidad generación** | ⚡⚡ Muy rápida (< 1s) | Lenta (1-2 min) |
| **Calidad** | ⭐⭐⭐⭐ Muy buena | ⭐⭐⭐⭐⭐ Excelente |
| **Pixelado** | Posible (interpolación) | Bajo (refinamiento iterativo) |
| **Variabilidad** | Determinista (misma entrada = misma salida) | Estocástica (variantes diferentes) |
| **VRAM** | Bajo (< 30%) | Alto (80%) |
| **Parámetros** | ~5-10M | 38.5M |

**¿Cuándo usar cada uno?**
- **UNet AE:** Prototipado rápido, producción, usuario con GPU limitada
- **DiT:** Máxima calidad, investigación, tiempo no es problema

---

## 📁 Estructura del proyecto

```
unet_ae_modular/
├── config.py        ← Parámetros centralizados (ÚNICO archivo a editar)
├── model.py         ← Arquitectura del UNet AE Condicional + FiLM
├── data_utils.py    ← Carga de .mat, ROIs, normalización, augmentation, generación
├── train.py         ← Entrenamiento + validación
├── generate.py      ← Generación de planos con un modelo ya entrenado
├── analysis.py      ← Interpolación latente y crossover de features
├── README.md
├── input/                       ← Colocar aquí los .mat originales
│   ├── TurbineX-..._PlaneAngle-104.47.mat
│   ├── TurbineX-..._PlaneAngle0.13.mat
│   ├── ...
│   └── validation/              ← Planos excluidos del entrenamiento (validación)
│       └── TurbineX-..._PlaneAngle....mat
└── output/                      ← (se crea automáticamente)
    ├── unet_ae_model.pt         ← Modelo entrenado
    ├── train/
    │   ├── PNG/                 ← Curva de entrenamiento, comparaciones orig vs recon
    │   └── MAT/                 ← ROIs originales exportadas
    ├── generate/
    │   ├── PNG/                 ← Figuras de planos generados (6 subplots)
    │   └── MAT/                 ← Planos generados en formato .mat
    ├── validation/
    │   ├── PNG/                 ← Real vs generado, gráfico error vs gap
    │   └── MAT/                 ← Planos generados para ángulos de validación
    └── analysis/
        └── PNG/                 ← Interpolación latente, crossover de features
```

---

## Requisitos

- **Python** ≥ 3.9
- **PyTorch** ≥ 2.0 (con CUDA recomendado para entrenamiento)

```bash
pip install numpy matplotlib h5py
```

### Datos de entrada

Los archivos **HDF5 `.mat`** se colocan en `input/`. El nombre debe contener `PlaneAngle` seguido del ángulo:

```
TurbineX-1320.00_Y-311.25_PlaneAngle-104.47.mat
TurbineX-1320.00_Y-311.25_PlaneAngle0.13.mat
```

Cada `.mat` debe contener las variables `tl`, `X` y `Z`.

---

## Guía rápida

### 1. Configurar parámetros

Editar **únicamente** `config.py`:

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `ROI_HEIGHT` | Alto del parche (ROI) extraído | 550 |
| `ROI_WIDTH` | Ancho del parche (ROI) extraído | 1000 |
| `ROI_MODE` | `"center_max"` (centrado en máximo) o `"corner_fixed"` (esquina fija) | `"corner_fixed"` |
| `ROI_CORNER` | Esquina superior-izquierda visual `(X, Z)` en coords físicas | `(-1320, 10)` |
| `ROIS_PER_PLANE` | ROIs extraídas por plano | 1 |
| `BATCH_SIZE` | Tamaño del batch | 8 |
| `EPOCHS` | Número de épocas | 70 |
| `LEARNING_RATE` | Tasa de aprendizaje (Adam) | 1e-4 |
| `USE_AUGMENTATION` | Data augmentation (×4) | `True` |
| `NOISE_STD` | Desviación del ruido latente en generación | 0.15 |
| `GENERATE_ANGLES` | Ángulos objetivo para `generate.py` | `[-170, …, 158.85]` |

### 2. Entrenar desde cero

```bash
python train.py
```

Ejecuta en orden:
1. Carga los `.mat` de `input/` y extrae ROIs
2. Normaliza TL y ángulos a [0, 1]
3. Data augmentation (si `USE_AUGMENTATION = True`)
4. Entrena con pérdida L1 y *mixed precision* (AMP) si hay GPU
5. Guarda curva de entrenamiento y comparaciones original vs reconstrucción
6. Exporta las ROIs originales como `.mat`
7. Guarda el modelo en `output/unet_ae_model.pt`
8. Valida con planos de `input/validation/` (si existen)

### 3. Generar planos

```bash
python generate.py
```

- Carga `output/unet_ae_model.pt` y verifica consistencia con `config.py`
- Genera planos de TL para cada ángulo en `GENERATE_ANGLES`
- Exporta `.mat` y figuras comparativas (6 subplots por ángulo)

### 4. Análisis avanzado

```bash
python analysis.py
```

- **Interpolación latente**: transición suave entre dos planos en el espacio latente
- **Crossover de features**: mezcla de features del encoder de un plano con el ángulo de otro

---

## Resumen de uso

| Situación | Comando |
|---|---|
| Primera vez | `python train.py` → `python generate.py` |
| Ya tienes `.pt` | `python generate.py` |
| Reentrenar | `python train.py` (sobreescribe el `.pt`) |
| Interpolación / crossover | `python analysis.py` |

### 5. Ajuste de hiperparámetros con Optuna

```bash
python optuna_tune.py --trials 20 --epochs 80
```

- Optimiza primero el MAPE de validación y, en segundo lugar, el pico de VRAM.
- El barrido se ejecuta sin inpainting para comparar solo el UNet base.
- Cada trial guarda sus salidas en `results/optuna_unet_ae_no_inpaint_v1/`.

---

## Arquitectura del modelo

### UNet Autoencoder Condicional con FiLM

```
Entrada: imagen TL (1×H×W) + ángulo normalizado (escalar)

ENCODER (sin condición):
  ConvBlock(1→32)   ── skip e1
  ConvBlock(32→64)  ── skip e2
  ConvBlock(64→128) ── skip e3
  ConvBlock(128→256) ── bottleneck b

DECODER (con FiLM condicionado por ángulo):
  FiLM(b) → Up + cat(e3) → ConvBlock → FiLM
           → Up + cat(e2) → ConvBlock → FiLM
           → Up + cat(e1) → ConvBlock → FiLM → Conv2d(32→1)

Salida: imagen TL reconstruida (1×H×W)
```

- **ConvBlock**: 2× Conv2d 3×3 + InstanceNorm + LeakyReLU(0.2)
- **FiLM**: MLP (1→64→2C) genera γ, β por canal → `x·(1+γ)+β`
- **Skip connections**: preservan detalle espacial entre encoder y decoder

### Proceso de generación

1. Seleccionar la **semilla** más cercana al ángulo objetivo
2. Codificar con el encoder → features latentes (e1, e2, e3, b)
3. Añadir **ruido controlado** (mayor en bottleneck, menor en niveles altos)
4. Decodificar con el **ángulo objetivo** como condición FiLM
5. Desnormalizar a valores reales de TL (dB)

---

## Data augmentation

Transformaciones físicamente coherentes (sin flips ni rotaciones):
- 2 niveles de ruido gaussiano (σ = 0.01, 0.03)
- 1 escalado aleatorio de contraste (×0.9–1.1 + offset)

**Factor**: ×4 (1 original + 3 aumentadas). Se activa con `USE_AUGMENTATION = True`.

---

## Validación

Al final del entrenamiento, si `input/validation/` contiene `.mat`:
1. Genera cada plano de validación usando la semilla de entrenamiento más cercana
2. Compara con el plano real: muestra original, generado y mapa de error
3. Genera gráfico `error_vs_gap.png`: MAE y error máximo vs distancia angular entre vecinos de entrenamiento

---

## Salidas generadas

### train.py

| Archivo | Descripción |
|---|---|
| `output/unet_ae_model.pt` | Pesos del modelo + `roi_height/width` + rangos de normalización |
| `output/train/PNG/training_curve_unet_ae.png` | Curva de pérdida L1 |
| `output/train/PNG/comparacion_*.png` | Original vs reconstrucción vs error |
| `output/train/MAT/roi_original_*.mat` | ROIs originales |
| `output/validation/PNG/validacion_*.png` | Real vs generado (validación) |
| `output/validation/PNG/error_vs_gap.png` | Error vs gap angular |
| `output/validation/MAT/val_generado_*.mat` | Planos generados para validación |

### generate.py

| Archivo | Descripción |
|---|---|
| `output/generate/PNG/plano_angulo_*.png` | 6 subplots por ángulo generado |
| `output/generate/MAT/plano_angulo_*.mat` | Plano generado (HDF5 con X, Y, Z, tl) |

### analysis.py

| Archivo | Descripción |
|---|---|
| `output/analysis/PNG/interpolacion_latente.png` | Transición suave entre dos planos |
| `output/analysis/PNG/crossover_features.png` | Mezcla de features entre planos |

---

## Formato de los `.mat` de salida

Archivos **HDF5** con:

| Variable | Descripción |
|---|---|
| `tl` | Matriz de Transmission Loss (dB) |
| `X`, `Z` | Coordenadas meshgrid |
| `Y` | Ceros (plano 2D) |

Se abren con MATLAB (`load(...)`) o Python (`h5py`).
