# TFG: Generación condicionada de planos de Transmission Loss (TL)

Repositorio con **dos modelos alternativos** para generar campos de Transmission Loss acústico condicionados por ángulo. Elige cuál usar según tus necesidades.

---

## 📊 Comparativa rápida

| Aspecto | **Intento_Transformers (DiT)** | **unet_ae_modular (UNet AE)** |
|--------|------|------|
| **Tipo** | Diffusion Transformer | Autoencoder Condicional |
| **Velocidad entrenamiento** | Lenta (220 épocas ≈ 6h) | Rápida (70 épocas ≈ 30min) |
| **Velocidad generación** | Lenta (DDIM: 150-350 pasos) | Rápida (1 forward pass) |
| **Calidad** | ⭐⭐⭐⭐⭐ Alta (iterativo refinamiento) | ⭐⭐⭐⭐ Buena (determinista) |
| **Pixelado** | Bajo (con DDIM steps adecuados) | Posible (interpolación) |
| **VRAM uso** | 80% (38M params, batch=4) | Bajo (< 30% típicamente) |
| **Mejor para** | Máxima calidad, investigación | Producción, iteración rápida |

---

## 🚀 Cómo elegir

### Usa **Intento_Transformers (DiT)** si:
- ✅ Quieres máxima calidad de imágenes
- ✅ El tiempo de entrenamiento no es limitante
- ✅ Puedes esperar 1-2 min por predicción
- ✅ Estudias modelos generativos

### Usa **unet_ae_modular (UNet AE)** si:
- ✅ Necesitas prototipado rápido
- ✅ Quieres entrenar/iterar múltiples veces
- ✅ Necesitas predicciones en < 1 segundo
- ✅ Tienes recursos GPU limitados

---

## 📁 Estructura del repositorio

```
TFG_repo/
├── README.md  ← Estás aquí
├── Intento_Transformers/          ← DiT (Difusión Transformer)
│   ├── config.py                  ← Parámetros optimizados
│   ├── model.py                   ← Arquitectura DiT
│   ├── train.py                   ← Entrenamiento
│   ├── generate.py                ← Generación DDIM
│   ├── data_utils.py              ← Carga datos
│   ├── README.md                  ← Documentación específica
│   ├── input/                     ← Datos de entrada
│   └── output/                    ← Resultados (modelos, gráficos, MAT)
│
└── unet_ae_modular/               ← UNet Autoencoder
    ├── config.py                  ← Parámetros específicos
    ├── model.py                   ← Arquitectura UNet AE
    ├── train.py                   ← Entrenamiento
    ├── generate.py                ← Generación
    ├── data_utils.py              ← Carga datos
    ├── analysis.py                ← Análisis latente
    ├── README.md                  ← Documentación específica
    ├── input/                     ← Datos de entrada
    └── output/                    ← Resultados
```

---

## 🔧 Setup inicial

### 1. Preparar datos

Coloca archivos `.mat` en la carpeta `input/` correspondiente:

```bash
# Para DiT
cp tus_datos/*.mat TFG_repo/Intento_Transformers/input/

# Para UNet AE
cp tus_datos/*.mat TFG_repo/unet_ae_modular/input/
```

**Formato esperado:** Cada `.mat` debe contener:
- `tl` o `TL`: Campo de Transmission Loss
- `X`, `Z`: Coordenadas espaciales
- Nombre con `PlaneAngle`: ej. `PlaneAngle-145.00_Length499.00.mat`

### 2. Configurar parámetros

Edita `config.py` en el directorio del modelo elegido.

**DiT parámetros clave:**
```python
BATCH_SIZE = 4              # Aumentar si tienes VRAM extra
EPOCHS = 220                # Entrenamiento completo
VALIDATION_DDIM_STEPS = 150 # Balance velocidad/calidad
GENERATION_DDIM_STEPS = 350 # Mayor para generación final
```

**UNet AE parámetros clave:**
```python
BATCH_SIZE = 8
EPOCHS = 70
USE_AUGMENTATION = True
NOISE_STD = 0.15
```

### 3. Entrenar

```bash
# DiT (lento, ~6 horas)
cd Intento_Transformers
python train.py

# UNet AE (rápido, ~30 minutos)
cd unet_ae_modular
python train.py
```

### 4. Generar predicciones

```bash
# DiT
cd Intento_Transformers
python generate.py

# UNet AE
cd unet_ae_modular
python generate.py
```

---

## 📊 Output esperado

Ambos modelos guardan en `output/`:

```
output/
├── model.pt                       ← Checkpoint entrenado
├── train/
│   ├── PNG/
│   │   ├── training_curve_dit.png ← Evolución de pérdida
│   │   └── ...
│   └── MAT/
│       └── train_rois_*.mat       ← ROIs originales
├── validation/
│   ├── PNG/
│   │   ├── validation_real_vs_generated.png
│   │   └── error_metrics_*.png
│   └── MAT/
│       └── val_generated_*.mat
└── generate/
    ├── PNG/
    │   └── plano_angulo_*.png
    └── MAT/
        └── plano_angulo_*.mat
```

---

## 🔍 Detalles técnicos

### Intento_Transformers (DiT)

**Arquitectura:**
- Encoder latente determinista (compression ratio 6)
- Diffusion Transformer con 12 bloques, 16 heads, 448 hidden dim
- Conditioning por ángulo normalizado
- Patch tokenization (patch_size=2)
- Gradient checkpointing activado

**Optimización:**
- Mixed precision training (FP16)
- EMA model (decay=0.9997)
- Scheduler cosine con warmup
- Attention chunking (chunk_size=448)
- OOM recovery automático

**Parámetros actuales:**
```
Modelo: 38.5M parámetros
VRAM: ~80% (MI210 68.7GB)
Tokens latente: 117×334
DDIM steps validación: 150
DDIM steps generación: 350
```

### unet_ae_modular (UNet AE)

**Arquitectura:**
- UNet encoder-decoder simétrico
- 4 bloques con skip connections
- Modulación FiLM para conditioning
- Determinista (sin ruido muestreo)

**Características:**
- Data augmentation (×4 rotaciones)
- L1 loss principal + L2 auxiliar
- Batch normalization
- Interpolación bicúbica

---

## 📝 Notas importantes

### Validación

Si quieres datos de validación, crea subdirectorio:
```bash
mkdir input/validation
cp algunos_datos.mat input/validation/
```

La validación se ejecuta automáticamente al final del entrenamiento.

### GPU/ROCm

Ambos modelos están optimizados para **AMD MI210** (ROCm). Si usas NVIDIA CUDA, los comandos funcionan igual pero pueden necesitar ajustes menores.

### Checkpoint

Los checkpoints guardados contienen:
- Pesos del modelo
- Parámetros de configuración
- Normalización (min/max de TL, ángulos)

Puedes cargar un checkpoint en `generate.py` sin reentrenar.

---

## 🐛 Troubleshooting

| Problema | Solución |
|----------|----------|
| **OOM (Out of Memory)** | Reducir BATCH_SIZE en config.py |
| **Imágenes pixeladas (DiT)** | Aumentar GENERATION_DDIM_STEPS a 500+ |
| **Entrenamiento muy lento** | Usar unet_ae_modular, o reducir EPOCHS |
| **Validación no se ejecuta** | Crear `input/validation/` con datos |

---

## 📚 Referencias

- **DiT:** *Scalable Diffusion Models with Transformers* (Peebles & Xie, 2023)
- **UNet AE:** Arquitectura clásica encoder-decoder con FiLM conditioning

---

## 👤 Autor

Trabajo de Fin de Grado - UPM  
Optimizado y refactorizado: abril 2026

**Configuración final optimizada:**
- DiT: 80% VRAM, +50-70% speedup vs baseline
- UNet AE: Entrenamiento rápido con buena calidad

