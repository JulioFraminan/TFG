# PINN Modular - TL regression

This folder mirrors the structure of unet_ae_modular but uses a PINN-style MLP
that maps (x, z, angle) to TL values.

## Files

- config.py
- model.py
- data_utils.py
- train.py
- generate.py
- input/ (put .mat files here)
- input/validation/ (optional validation planes)
- output/ (created automatically)

## Quick start

1) Edit config.py
2) Train:

```
python train.py
```

3) Generate:

```
python generate.py
```

## Notes

- PDE_TYPE controls the physics residual (none, laplace, helmholtz, two_layer_helmholtz).
- two_layer_helmholtz enforces Helmholtz in two media plus interface continuity
	of pressure and normal velocity (Snell-style refraction emerges from this).
- Set FREQUENCY_HZ, AIR/WATER properties, INTERFACE_Z, AIR_ABOVE_INTERFACE, and
	INTERFACE_EPS in config.py to match your scene and units.
- If INTERFACE_Z is None, the midpoint of the ROI bounds is used.
- Physics uses TL->pressure conversion when OUTPUT_IS_TL=True.
- If your TL values are negative dB (common in your data), keep TL_DB_SIGN = -1.0
	so the conversion uses a positive attenuation in dB.
- This PINN is data-driven; physics loss is a regularizer and may need tuning.
