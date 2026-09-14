# Intento_FNO

Este modulo implementa un Fourier Neural Operator (FNO) condicional para predecir TL en ROIs. El modelo aprende una transformacion global en el dominio de Fourier y condiciona cada pixel por el angulo del plano (y, opcionalmente, un grid de coordenadas).

## Arquitectura (model.py)

- Entrada por pixel: TL normalizado (1 canal) + angulo (1 canal constante por pixel) + coords (2 canales opcionales).
- Lifting: `fc0` proyecta cada pixel a `FNO_WIDTH`.
- Bloques FNO (`FNO_DEPTH`):
  - `SpectralConv2d`: rfft2, truncado a modos (`FNO_MODES1`, `FNO_MODES2`) con pesos complejos para bandas baja y alta.
  - Ruta 1x1: `Conv2d(width, width, 1)`.
  - Suma, `GELU` y `Dropout2d` (si `FNO_DROPOUT > 0`).
- Proyeccion: `fc1` + `GELU` + `fc2` para volver a 1 canal.

## Condicionamiento por angulo

- `Normalizer` calcula `angle_min` y `angle_max` en entrenamiento.
- El angulo se normaliza a $[0, 1]$ y se pasa como `cond_dim = 1`.
- El `cond_map` se expande a $H \times W$ y se concatena con TL (y coords si `FNO_USE_COORDS=True`).

## Loss y objetivo

La loss de entrenamiento es L1 (MAE) entre la reconstruccion y el TL objetivo:

$$
L = \lVert \hat{T} - T \rVert_1
$$

## Generacion y relacion con el angulo

- Se selecciona una semilla cercana en angulo y se aplica un ruido controlado por `NOISE_STD`.
- El angulo objetivo entra como condicion y dirige la salida hacia el plano deseado.

## Hiperparametros clave (config.py)

- `FNO_MODES1`, `FNO_MODES2`: cantidad de modos espectrales por eje.
- `FNO_WIDTH`: canales internos en cada bloque.
- `FNO_DEPTH`: numero de bloques FNO.
- `FNO_USE_COORDS`: agrega grid (x, y) normalizado.
- `FNO_DROPOUT`: dropout espacial en bloques.
- `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`.
- `NOISE_STD`: intensidad del ruido en generacion.

## Checkpoint

El `.pt` guarda:
- `model_state_dict`.
- `roi_height` / `roi_width`.
- `fno_*` (modes, width, depth, use_coords, dropout).
- Rangos de normalizacion (`tl_min`, `tl_max`, `angle_min`, `angle_max`).
