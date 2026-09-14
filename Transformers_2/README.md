# Transformers_2 - DiT / UNet condicionados por angulo

Este flujo entrena un modelo generativo condicional para TL usando un backbone tipo DiT (Vision Transformer) o un UNet. El condicionamiento es un escalar (angulo normalizado) y se integra con guidance en el muestreo.

## Arquitectura DiT (denoising_diffusion_pytorch/dit.py)

**Tokenizacion**
- La imagen $H \times W$ se divide en patches de tamano `patch_size` (definido por `dit_variant`, por ejemplo DiT-S/8).
- `PatchEmbed` convierte cada patch a un token de dimension `hidden_size`.
- Se suma `pos_embed` fijo (sine-cosine 2D).

**Bloques DiT**
- Cada `DiTBlock` aplica:
  - `RMSNorm` + atencion (vanilla / linear / window segun `dit_attn_type`).
  - `RMSNorm` + MLP con ratio `dit_mlp_ratio`.
  - AdaLN-Zero: la condicion modula el bloque con $(shift, scale, gate)$.

**Condicion y tiempo**
- `TimestepEmbedder`: embedding sinusoidal + MLP.
- `ConditionEmbedder`: MLP sobre el angulo con `dit_class_dropout` para guidance.
- La condicion final es $c = t_{emb} + y_{emb}$ y entra en todos los bloques.

**Salida**
- `FinalLayer` proyecta tokens a patches y `unpatchify` reconstruye la imagen.

## Arquitectura UNet (continuous_classifier_free_guidance.py)

- Entrada por imagen con `init_conv` 7x7.
- Down path: bloques ResNet + atencion lineal + downsample segun `unet_dim_mults`.
- Mid block con atencion.
- Up path simetrico con skip connections.
- Condicion y tiempo se inyectan en `ResnetBlock` via MLP (scale/shift/gate).

## Condicionamiento por angulo

- `NormalizationStats` calcula `angle_mean` y `angle_std`.
- El angulo se normaliza con:

$$
\theta_{norm} = \frac{\theta - \mu_{\theta}}{\sigma_{\theta}}
$$

- `cond_dim = 1` y la condicion entra en `ConditionEmbedder` (DiT) o `classes_mlp` (UNet).
- En muestreo, `cond_scale` aplica guidance con `forward_with_cond_scale`.

## Loss en FlowMatching (transport/transport.py)

Cuando `--algorithm flowmatching`:

- Se muestrea un par $(x_0, x_1)$ y un tiempo $t$.
- Se construye un punto intermedio $x_t$ y un objetivo $u_t$ segun el `path_type`.
- La red predice una magnitud definida por `flow_prediction`:
  - `velocity`: MSE entre salida y $u_t$.
  - `noise` / `score`: MSE ponderada con la varianza de la trayectoria.
- Si esta activo, se suma una perdida de coseno.

La `FlowMatching` retorna la loss promedio (y `cos_loss` si existe).

## Hiperparametros clave (config.py)

**Modelo**
- `model_type`: `dit` o `unet`.
- `dit_variant`, `dit_attn_type`, `dit_mlp_ratio`, `dit_qk_norm`, `dit_class_dropout`.
- `unet_dim`, `unet_dim_mults`, `unet_cond_drop_prob`.

**FlowMatching**
- `flow_path_type`, `flow_prediction`, `flow_loss_weight`.
- `flow_num_steps`, `flow_sampling_method`, `flow_atol`, `flow_rtol`.

**Optimizacion**
- `train_batch_size`, `train_lr`, `train_num_steps`, `gradient_accumulate_every`.
- `ema_decay`, `save_and_sample_every`.

## Metadata y checkpoints

Cada experimento guarda `intento_training_metadata.json` con:
- `tl_min`, `tl_max`, `angle_mean`, `angle_std`.
- Configuracion del modelo y del algoritmo.
- Tamano de entrenamiento y firma de configuracion.
