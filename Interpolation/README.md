# Interpolacion lineal por angulo — Planos TL

Generador determinista de campos *Transmission Loss* (TL) sin machine learning.
El metodo toma dos planos vecinos por angulo y hace una interpolacion lineal
punto a punto para obtener el plano intermedio.

## Idea base

Para un angulo objetivo $\theta$:

1. Se buscan los dos planos vecinos (angulo inferior y superior).
2. Se calcula el peso $w$ segun la distancia angular:

$$
w = \frac{\theta - \theta_{low}}{\theta_{high} - \theta_{low}}
$$

3. Se interpola por pixel:

$$
TL_{interp} = (1 - w) \cdot TL_{low} + w \cdot TL_{high}
$$

Si $\theta$ cae fuera del rango disponible, el peso se clampa a $[0, 1]$.

---

## Estructura del proyecto

```
Interpolation/
├── config.py        ← Parametros (ROI, rutas, angulos)
├── data_utils.py    ← Carga de .mat, ROIs, normalizacion, interpolacion
├── model.py         ← Helper de interpolacion lineal (sin ML)
├── train.py         ← Exporta ROIs y valida con interpolacion lineal
├── generate.py      ← Genera planos para angulos objetivo
├── analysis.py      ← Barrido lineal entre min y max
├── README.md
├── input/           ← Colocar aqui los .mat originales
│   └── validation/  ← Planos de validacion (opcional)
└── output/          ← (se crea automaticamente)
    ├── train/
    │   └── MAT/     ← ROIs originales exportadas
    ├── generate/
    │   ├── PNG/     ← Figuras de planos interpolados
    │   └── MAT/     ← Planos interpolados en .mat
    ├── validation/
    │   ├── PNG/     ← Real vs interpolado, error vs gap
    │   └── MAT/     ← Planos interpolados de validacion
    └── analysis/
        └── PNG/     ← Barrido lineal min→max
```

---

## Requisitos

- Python >= 3.9
- numpy, matplotlib, h5py

```bash
pip install numpy matplotlib h5py
```

---

## Datos de entrada

Los archivos **HDF5 `.mat`** se colocan en `input/`. El nombre debe contener
`PlaneAngle` seguido del angulo:

```
Flat_SeabedSand_PlaneAngle-10.00_Length299.00.mat
Flat_SeabedSand_PlaneAngle0.00_Length299.00.mat
```

Cada `.mat` debe incluir `tl` (o `tl_block` si se usa `USE_BLOCK_VARIABLES`),
y ejes `X`/`Z` o `R`/`Z`.

---

## Guia rapida

### 1) Configurar parametros

Editar solo `config.py`:

- `ROI_HEIGHT`, `ROI_WIDTH`: tamano de la ROI.
- `ROI_MODE`: `corner_fixed` o `center_max`.
- `ROI_CORNER`: esquina superior izquierda fisica (solo para `corner_fixed`).
- `USE_BLOCK_VARIABLES`: usar `tl_block` en vez de `tl`.
- `GENERATE_ANGLES`: angulos objetivo para `generate.py`.

### 2) Exportar ROIs y validar

```bash
python train.py
```

Este script:
1. Carga los `.mat` de `input/` y extrae ROIs.
2. Exporta ROIs originales a `output/train/MAT/`.
3. Si hay `input/validation/`, interpola y calcula errores.

### 3) Generar planos para angulos objetivo

```bash
python generate.py
```

Genera planos interpolados para cada angulo en `GENERATE_ANGLES` y guarda
`.mat` y figuras en `output/generate/`.

### 4) Analisis (barrido lineal)

```bash
python analysis.py
```

Crea un barrido lineal entre el angulo minimo y maximo disponibles.

---

## Validacion

Si `input/validation/` contiene `.mat`, el script `train.py`:

1. Busca los dos vecinos angulares para cada plano de validacion.
2. Interpola linealmente y calcula MAE, RMSE, MAPE y Max.
3. Guarda figuras y un grafico `error_vs_gap.png`.

---

## Salidas generadas

### train.py

| Archivo | Descripcion |
|---|---|
| `output/train/MAT/roi_original_*.mat` | ROIs originales exportadas |
| `output/validation/PNG/validacion_real_vs_interpolado_*.png` | Real vs interpolado |
| `output/validation/PNG/error_vs_gap.png` | Error vs gap angular |
| `output/validation/MAT/val_interpolado_*.mat` | Planos interpolados de validacion |

### generate.py

| Archivo | Descripcion |
|---|---|
| `output/generate/PNG/plano_angulo_*.png` | Figuras de planos interpolados |
| `output/generate/MAT/plano_angulo_*.mat` | Plano interpolado en .mat |

### analysis.py

| Archivo | Descripcion |
|---|---|
| `output/analysis/PNG/interpolacion_lineal.png` | Barrido lineal min→max |

---

## Formato de los `.mat` de salida

Archivos **HDF5** con:

| Variable | Descripcion |
|---|---|
| `tl` | Matriz de Transmission Loss (dB) |
| `X`, `Z` | Coordenadas meshgrid |
| `Y` | Ceros (plano 2D) |

Se abren con MATLAB (`load(...)`) o Python (`h5py`).