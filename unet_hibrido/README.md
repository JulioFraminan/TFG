# UNet Hibrido — UNet + MLP + PINN (versión actual recomendada)

Este es el **módulo híbrido principal y recomendado**. No es solo un UNet con física: combina

- **UNet Autoencoder condicional (FiLM)** para reconstrucción rápida de planos TL.
- **Un PINN MLP independiente** para la parte física y la regresión continua en coordenadas.
- **Pérdidas físicas** sobre ese MLP: término PDE + término de interfaz aire/agua.
- **Inpainting opcional** sobre la rama UNet para robustez frente a huecos.

La idea es separar responsabilidades: el **UNet aprende la reconstrucción de imagen** y el **MLP aprende la función física continua**. Eso permite mantener la rapidez del UNet y, a la vez, meter coherencia física y restricciones de Helmholtz en el entrenamiento.

Resumen rapido:
- **Modelo**: UNet AE condicional por angulo (FiLM), con opcion de inpainting.
- **Fisica**: perdida PDE (Laplace/Helmholtz/2 capas) + perdida de interfaz aire/agua.
- **Derivadas**: diferencias finitas en la grilla de cada ROI (no collocation points).

---

## Estructura del proyecto

```
unet_hibrido/
├── config.py        <- Parametros (datos + entrenamiento + fisica + optuna)
├── model.py         <- UNet AE condicional con FiLM + PINN
├── data_utils.py    <- Carga de .mat, ROIs, normalizacion, augment, etc.
├── train.py         <- Entrenamiento con data loss + physics loss
├── generate.py      <- Generacion con modelo entrenado
├── analysis.py      <- Analisis/visualizaciones
├── optuna_tune.py   <- Tuning automatico de hiperparametros
├── input/           <- .mat originales
│   └── validation/  <- .mat reservados para validacion
└── output/          <- Salidas (se crea automaticamente)
```

---

## Idea clave: UNet + MLP + PINN

Aquí hay dos rutas de optimización en paralelo:

```
loss_total = DATA_WEIGHT * data_loss_UNet
       + PINN_DATA_WEIGHT * pinn_data_loss
       + PHYSICS_WEIGHT * (physics_loss + INTERFACE_WEIGHT * interface_loss)
```

- **data_loss_UNet**: reconstrucción entre la salida del UNet y la ROI real.
- **pinn_data_loss**: ajuste del MLP a muestras físicas extraídas de las ROIs.
- **physics_loss**: residual PDE aplicado al MLP sobre puntos interiores / collocation points.
- **interface_loss**: continuidad de presión y velocidad normal en la interfaz aire/agua.

En otras palabras: el UNet resuelve la parte densa en píxeles y el MLP PINN resuelve la parte física continua.

---

## Como se calcula la fisica (diferencias finitas)

### 1) De TL a presion
Si `OUTPUT_IS_TL = True`, el modelo predice TL normalizado y se convierte a presion:

```
tl_db = tl_norm * (tl_max - tl_min) + tl_min
tl_db = tl_db * TL_DB_SIGN + TL_DB_OFFSET
p = TL_REF_PRESSURE * exp(-tl_db * ln(10) / 20)
```

Si `OUTPUT_IS_TL = False`, se asume que la salida ya es presion.

### 2) Residual PDE en el espacio continuo del PINN
El MLP PINN recibe coordenadas normalizadas `(x, z, angulo)` y el residual PDE se calcula con autograd:

```
\partial_{xx} p, \partial_{zz} p \rightarrow residual de Laplace / Helmholtz / dos capas
```

El residual final depende de `PDE_TYPE`:

- `laplace`:   r = p_xx + p_zz
- `helmholtz`: r = p_xx + p_zz + k^2 * p
- `two_layer_helmholtz`: k cambia segun el medio (aire/agua) y la posicion z

**Submuestreo:** si `PHYSICS_BATCH_SIZE > 0`, se selecciona un subconjunto de puntos para el residual. Si no, se usa el conjunto completo disponible.

### 3) Interfaz aire/agua (interface_loss)
Se evalua en dos filas cercanas a `z = INTERFACE_Z`:

- Fila **z_minus** = `INTERFACE_Z - INTERFACE_EPS`
- Fila **z_plus**  = `INTERFACE_Z + INTERFACE_EPS`

En cada ROI:

1) Se buscan los indices de filas mas cercanos a z_minus y z_plus.
2) Se calcula continuidad de presion:
   `res_p = p_plus - p_minus`
3) Se calcula continuidad de velocidad normal:
   `res_v = (1/rho_plus) * dp/dz_plus - (1/rho_minus) * dp/dz_minus`
4) La loss es el promedio de `res_p^2 + res_v^2`.

Si `INTERFACE_Z` cae fuera del rango `[z_min, z_max]` de la ROI, esa muestra **no aporta** a interface_loss (queda 0). Esto explica por que `iface` puede aparecer en cero en el log.

Si `INTERFACE_BATCH_SIZE > 0`, se submuestrean columnas a lo largo de X para calcular la interfaz mas rapido.

---

## Diferencias finitas vs collocation points

**UNet (reconstrucción en grilla):**
- La salida de imagen sí vive en la **grilla discreta** de cada ROI.
- Es la parte rápida y visual del sistema.
- Se optimiza con pérdidas de reconstrucción e inpainting.

**PINN MLP (física continua):**
- Trabaja sobre puntos `(x, z, angulo)`.
- Usa autograd para derivadas, no diferencias finitas.
- Es la parte que impone la física y la interfaz.

**Collocation points (PINN MLP):**
- Se muestrean puntos aleatorios (x, z, angulo) en el dominio continuo.
- Se usa **autograd** para derivadas, no diferencias finitas.
- Mas flexible y continuo, pero mas caro.

En resumen: aqui la fisica **no** se calcula en un continuo infinito de puntos, sino sobre la grilla del ROI (H x W). Si hay submuestreo, se usan menos puntos.

---

## Parámetros importantes (config.py)

### Fisica
- `PDE_TYPE`: "none" | "laplace" | "helmholtz" | "two_layer_helmholtz"
- `PHYSICS_WEIGHT`: peso global de la fisica
- `INTERFACE_WEIGHT`: peso de la interfaz
- `PINN_DATA_WEIGHT`: peso del ajuste de datos del MLP PINN
- `PINN_DATA_BATCH_SIZE`: tamaño del lote de datos del PINN
- `POINTS_PER_ROI`: puntos muestreados por ROI para el PINN
- `PHYSICS_BATCH_SIZE`: submuestreo de puntos PDE (0 = todos)
- `INTERFACE_BATCH_SIZE`: submuestreo de columnas en interfaz (0 = todos)
- `INTERFACE_Z`: z de interfaz (None = punto medio del ROI)
- `INTERFACE_EPS`: separacion +/- alrededor de la interfaz

### Datos
- `ROI_HEIGHT`, `ROI_WIDTH`, `ROI_MODE`, `ROI_CORNER`, `ROIS_PER_PLANE`

### Entrenamiento
- `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `MODEL_DROPOUT`

---

## Como entrenar

```
python train.py
```

El entrenamiento:
1) Carga .mat y extrae ROIs
2) Normaliza TL y angulos
3) Aplica augment (si `USE_AUGMENTATION = True`)
4) Entrena con data_loss + physics_loss
5) Exporta curvas, comparaciones y el modelo en `output/`

---

## Generacion

```
python generate.py
```

Genera planos para los angulos de `GENERATE_ANGLES` y guarda PNG/MAT en `output/generate/`.

---

## Salidas principales

| Ruta | Descripcion |
|---|---|
| `output/unet_ae_model.pt` | Checkpoint con pesos + normalizacion + parametros fisicos |
| `output/train/PNG/training_curve_unet_ae.png` | Curva de entrenamiento (total/data/physics) |
| `output/train/PNG/comparacion_*.png` | Comparacion original vs reconstruccion |
| `output/train/MAT/roi_original_*.mat` | ROIs originales exportadas |

---

## Nota sobre `iface = 0`

Si `INTERFACE_Z` esta fuera del rango vertical de la ROI (por ejemplo, ROI con Z en [-50, -5] y `INTERFACE_Z = 0`), la perdida de interfaz no se calcula y queda en cero. En ese caso:

- Cambia `INTERFACE_Z` a un valor dentro del rango
- o usa `INTERFACE_Z = None` para tomar el punto medio de cada ROI
