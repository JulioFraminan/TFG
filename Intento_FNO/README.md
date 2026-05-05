# Intento_FNO

Este proyecto implementa un modelo Fourier Neural Operator (FNO) condicional para reconstruir y generar planos TL a partir de ROIs y un angulo. Mantiene la misma estructura de datos y pipeline que el modelo anterior, pero con una arquitectura espectral orientada a capturar dependencias globales.

## Diferencias clave frente al UNet AE

- El FNO opera en el dominio de Fourier con convoluciones espectrales y mezcla global, no usa encoder/decoder ni skip connections.
- La entrada se condiciona con el angulo y (opcionalmente) un grid de coordenadas, en lugar de FiLM por nivel.
- La capacidad para interpolar angulos se apoya en la mezcla global de frecuencias (modos truncados) en lugar de multiescala local.
- La generacion agrega ruido al seed en el dominio espacial, no a niveles intermedios del encoder.
- El costo principal es la FFT por bloque, no el downsampling/upsampling.

## Arquitectura del modelo (FNO condicional)

- **Lifting**: `fc0` proyecta la entrada (TL + angulo + coords) a un ancho `FNO_WIDTH`.
- **Bloques FNO**: cada bloque combina una convolucion espectral (FFT -> truncado -> IFFT) con una ruta 1x1.
- **Proyeccion**: `fc1` + `fc2` llevan el ancho interno a 1 canal de salida.
- **Activacion**: GELU en todos los bloques.

## Condicionamiento y entradas

- El angulo se normaliza y se concatena como un canal constante por pixel.
- Si `FNO_USE_COORDS=True`, se agregan dos canales de coordenadas normalizadas (x, y).
- Entrada final a `fc0`: `[TL, angle, coords]`.

## Entrenamiento y generacion

- Entrenamiento con loss L1 (MAE) sobre reconstruccion del mismo ROI.
- La normalizacion TL y de angulos se conserva y se guarda en el checkpoint.
- La generacion usa una semilla (ROI mas cercano en angulo) y ruido gaussiano en el seed.

## Parametros principales

- `FNO_MODES1`, `FNO_MODES2`: modos espectrales retenidos (capacidad global).
- `FNO_WIDTH`: canales internos del FNO.
- `FNO_DEPTH`: numero de bloques espectrales.
- `FNO_USE_COORDS`: activa canales de coordenadas.
- `FNO_DROPOUT`: dropout espacial en bloques.

## Estructura de carpetas

- `input/`: datos .mat de entrenamiento y `input/validation/`.
- `output/`: resultados de train/generate/validation.
- `logs/`: logs de ejecucion.

## Archivos clave

- `config.py`: hiperparametros y rutas.
- `model.py`: definicion del FNO condicional.
- `data_utils.py`: carga de datos, normalizacion, generacion.
- `train.py`: entrenamiento + validacion.
- `generate.py`: generacion condicionada por angulo.
