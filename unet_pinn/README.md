# UNet PINN Inpainting — UNet AE + Fisica (Helmholtz)

Este proyecto fusiona el **UNet Autoencoder condicional (FiLM)** con la **fisica tipo PINN** (ecuaciones de Helmholtz) del modelo `pinn_modular`. El objetivo es mantener la rapidez del UNet y anadir coherencia fisica en el entrenamiento.

Resumen rapido:
- **Modelo**: UNet AE condicional por angulo (FiLM), con opcion de inpainting.
- **Fisica**: perdida PDE (Laplace/Helmholtz/2 capas) + perdida de interfaz aire/agua.
- **Derivadas**: diferencias finitas en la grilla de cada ROI (no collocation points).

---

## Estructura del proyecto

```
unet_pinn/
├── config.py        <- Parametros (datos + entrenamiento + fisica)
├── model.py         <- UNet AE condicional con FiLM
├── data_utils.py    <- Carga de .mat, ROIs, normalizacion, augment, etc.
├── train.py         <- Entrenamiento con data loss + physics loss
├── generate.py      <- Generacion con modelo entrenado
├── analysis.py      <- Analisis/visualizaciones
├── input/           <- .mat originales
│   └── validation/  <- .mat reservados para validacion
└── output/          <- Salidas (se crea automaticamente)
```

---

## Idea clave: UNet + PINN

En el UNet puro, la loss es solo de reconstruccion (L1). Aqui se usa una loss compuesta:

```
loss = DATA_WEIGHT * data_loss
     + PHYSICS_WEIGHT * (physics_loss + INTERFACE_WEIGHT * interface_loss)
```

- **data_loss**: L1 entre reconstruccion y TL real (o inpainting si aplica).
- **physics_loss**: residual PDE calculado por diferencias finitas en la grilla.
- **interface_loss**: continuidad de presion y velocidad en la interfaz aire/agua.

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

### 2) Residual PDE en la grilla
Para cada ROI (H x W) se calcula el laplaciano con diferencias finitas centradas:

```
p_xx(i,j) = (p(i,j+1) - 2p(i,j) + p(i,j-1)) / dx^2
p_zz(i,j) = (p(i+1,j) - 2p(i,j) + p(i-1,j)) / dz^2
```

Esto solo se puede evaluar en puntos **interiores** (se pierde 1 pixel de borde). El residual final depende de `PDE_TYPE`:

- `laplace`:   r = p_xx + p_zz
- `helmholtz`: r = p_xx + p_zz + k^2 * p
- `two_layer_helmholtz`: k cambia segun el medio (aire/agua) y la posicion z

**Submuestreo:** si `PHYSICS_BATCH_SIZE > 0`, se selecciona un subconjunto aleatorio de puntos interiores para la loss. Si no, se usa todo el interior.

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

**Diferencias finitas (este proyecto):**
- Se calcula la fisica en la **grilla discreta** de cada ROI.
- Es mas barato y estable para un UNet que ya trabaja en pixeles.
- Derivadas limitadas a la resolucion de la grilla (no continuo).
- Residual PDE se evalua en todos los puntos interiores o en un subconjunto (segun `PHYSICS_BATCH_SIZE`).

**Collocation points (PINN MLP):**
- Se muestrean puntos aleatorios (x, z, angulo) en el dominio continuo.
- Se usa **autograd** para derivadas, no diferencias finitas.
- Mas flexible y continuo, pero mas caro.

En resumen: aqui la fisica **no** se calcula en un continuo infinito de puntos, sino sobre la grilla del ROI (H x W). Si hay submuestreo, se usan menos puntos.

---

## Parametros importantes (config.py)

### Fisica
- `PDE_TYPE`: "none" | "laplace" | "helmholtz" | "two_layer_helmholtz"
- `PHYSICS_WEIGHT`: peso global de la fisica
- `INTERFACE_WEIGHT`: peso de la interfaz
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
