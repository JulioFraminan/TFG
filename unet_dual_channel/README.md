# UNet AE condicional para TL

Este modulo implementa un autoencoder UNet condicional para reconstruir campos de TL a partir de ROIs. La condicion es el angulo del plano y se aplica via FiLM en encoder y decoder.

## Arquitectura (model.py)

**ConvBlock**
- 2x `Conv2d(3x3)` + `GroupNorm(8)` + `LeakyReLU(0.2)`.

**Encoder**
- `enc1`: 1 -> 32
- `enc2`: 32 -> 64
- `enc3`: 64 -> 128
- `MaxPool2d` entre niveles.

**Bottleneck**
- `ConvBlock(128 -> 256)` + `Dropout2d(MODEL_DROPOUT)`.

**Decoder**
- `ConvTranspose2d` para upsampling
- concatenacion con skips del encoder
- `ConvBlock` por nivel

**Atencion**
- `AttentionBlock` en cada skip: `att3`, `att2`, `att1`.

**Salida**
- `Conv2d(32 -> 1, kernel=1)`.

## Condicionamiento por angulo

- El angulo se normaliza a $[0, 1]$ con `Normalizer`.
- `FiLM` usa un MLP `1 -> 64 -> 2C` para producir $(\gamma, \beta)$.
- La modulacion es:

$$
\mathrm{FiLM}(x) = x \cdot (1 + \gamma) + \beta
$$

- Se aplica en `enc3`, bottleneck y en cada nivel del decoder.

## Loss y objetivo

La loss usada es L1 (MAE) entre la reconstruccion y el TL objetivo:

$$
L = \lVert \hat{T} - T \rVert_1
$$

## Generacion condicionada

- Se selecciona una semilla cercana al angulo objetivo.
- Se agrega un ruido controlado por `NOISE_STD`.
- El decoder genera el plano condicionado por el angulo objetivo.

## Hiperparametros clave (config.py)

- `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`.
- `MODEL_DROPOUT` (dropout en bottleneck).
- `NOISE_STD` para la generacion.
- Canales fijos en la arquitectura: 32, 64, 128, 256.

## Checkpoint

El `.pt` guarda:
- `model_state_dict`.
- `roi_height` / `roi_width`.
- Rangos de normalizacion (`tl_min`, `tl_max`, `angle_min`, `angle_max`).
