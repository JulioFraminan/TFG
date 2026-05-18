# PINN Modular - TL regression

Este directorio implementa una versión PINN modular para regresión de TL
(Transmission Loss) a partir de planos 2D. La idea central es simple:

- la red recibe coordenadas físicas normalizadas `(x, z, angle)`;
- devuelve un valor escalar de TL para ese punto;
- el entrenamiento combina ajuste a datos y una restricción física opcional;
- la inferencia reconstruye un plano completo evaluando la red punto por punto.

La diferencia importante respecto a un autoencoder o a un modelo puramente
convolucional es que aquí la red no ve la imagen completa como una matriz de
entrada, sino que aprende una función continua sobre el espacio físico.

## Objetivo del módulo

Este programa está pensado para estudiar cómo cambia el TL con la posición y
con el ángulo del plano. El pipeline toma archivos `.mat`, extrae regiones de
interés ROI, construye muestras puntuales de entrenamiento, entrena un MLP PINN
y luego vuelve a proyectar la predicción sobre una malla completa para comparar
con el plano original o con planos de validación.

La lógica está separada en piezas pequeñas:

- [config.py](config.py): parámetros de entrenamiento, física y rutas.
- [model.py](model.py): red PINN MLP.
- [data_utils.py](data_utils.py): carga de `.mat`, ROIs, normalización,
	muestreo, reconstrucción y guardado.
- [train.py](train.py): entrenamiento completo y guardado del checkpoint.
- [generate.py](generate.py): inferencia/generación a partir del modelo ya entrenado.

## La parte verdaderamente innovadora del PINN

La parte innovadora de este módulo no es solo la red MLP, sino la forma en la
que se convierte una imagen acústica en una función continua con física
inyectada. El núcleo está en [PINN](model.py#L16), que aproxima el campo TL a
partir de `(x, z, angle)`, y en [compute_pde_residual](train.py#L76), que usa
derivadas automáticas para imponer ecuaciones diferenciales sobre la salida de
la red.

La segunda pieza clave es [compute_interface_loss](train.py#L133), que añade la
condición de continuidad en la interfaz entre medios. Eso permite modelar una
escena con dos dominios acústicos sin forzar una discontinuidad artificial en la
predicción.

### Componentes que sostienen esa idea

- [Normalizer](data_utils.py#L275) fija los rangos de TL y ángulo que se guardan
	en el checkpoint.
- [CoordNormalizer](data_utils.py#L321) lleva coordenadas y ángulo a `[-1, 1]`
	para que la red trabaje en un espacio numéricamente estable.
- [sample_points_from_rois](data_utils.py#L505) convierte cada ROI en puntos
	independientes `(x, z, angle, tl)`.
- [predict_on_grid](data_utils.py#L534) reconstruye el plano completo para la
	inferencia.
- [load_inference_pipeline](data_utils.py#L559) vuelve a crear la red, la
	normalización y los datos a partir del checkpoint.

### Parámetros que gobiernan la física

- [DATA_WEIGHT](config.py#L22), [PHYSICS_WEIGHT](config.py#L23) y
	[INTERFACE_WEIGHT](config.py#L25) controlan el balance entre datos y física.
- [PDE_TYPE](config.py#L26) activa `none`, `laplace`, `helmholtz` o
	`two_layer_helmholtz`.
- [FREQUENCY_HZ](config.py#L30), [AIR_SOUND_SPEED](config.py#L31),
	[AIR_DENSITY](config.py#L32), [WATER_SOUND_SPEED](config.py#L33) y
	[WATER_DENSITY](config.py#L34) definen los dos números de onda del problema.
- [AIR_ABOVE_INTERFACE](config.py#L35), [INTERFACE_Z](config.py#L36) y
	[INTERFACE_EPS](config.py#L37) ubican y muestrean la interfaz.
- [TL_REF_PRESSURE](config.py#L38), [OUTPUT_IS_TL](config.py#L39),
	[TL_DB_SIGN](config.py#L40), [TL_DB_OFFSET](config.py#L41),
	[TL_DB_CLAMP_MIN](config.py#L42) y [TL_DB_CLAMP_MAX](config.py#L43) gobiernan
	la conversión de TL a presión cuando la física se calcula sobre la salida de
	la red.

### Qué hace cada pieza, en orden lógico

1. La red [PINN](model.py#L16) produce una estimación escalar de TL por punto.
2. [compute_pde_residual](train.py#L76) convierte esa salida en una cantidad
	 física y calcula derivadas de primer y segundo orden respecto a `x` y `z`.
3. Si `PDE_TYPE = "two_layer_helmholtz"`, el residuo usa el medio que
	 corresponde a cada punto según [INTERFACE_Z](config.py#L36) y
	 [AIR_ABOVE_INTERFACE](config.py#L35).
4. [compute_interface_loss](train.py#L133) evalúa la continuidad a ambos lados
	 de la interfaz usando [INTERFACE_EPS](config.py#L37).
5. La pérdida total combina datos y física con [DATA_WEIGHT](config.py#L22),
	 [PHYSICS_WEIGHT](config.py#L23) e [INTERFACE_WEIGHT](config.py#L25).

### Qué significa esto en práctica

Este PINN no sustituye automáticamente la información de datos. En este código,
la física actúa como regularizador. Si [PHYSICS_WEIGHT](config.py#L23) es muy
alto, puede dominar el ajuste de datos; si es muy bajo, la restricción física
apenas influye. La configuración de red también importa:
[PINN_HIDDEN_DIM](config.py#L46), [PINN_NUM_LAYERS](config.py#L47),
[PINN_ACTIVATION](config.py#L48) y [PINN_DROPOUT](config.py#L49) determinan la
capacidad de aproximación del MLP.

### Soporte operativo del checkpoint

El guardado y la validación de consistencia del modelo dependen de
[create_output_dirs](config.py#L71) y [verify_config](config.py#L76). La primera
crea la estructura de salida y la segunda avisa si el checkpoint no coincide con
la configuración actual.
- la reconstrucción de una malla completa para inferencia;
- la exportación de resultados a `.mat`.

### [train.py](train.py)

Implementa el ciclo completo de entrenamiento:

- carga los datos;
- calcula rangos globales;
- construye la red;
- entrena por épocas;
- evalúa pérdidas de datos y físicas;
- guarda curvas, comparativas y checkpoint.

### [generate.py](generate.py)

Carga el checkpoint, reconstruye la tubería de inferencia y genera planos para
los ángulos definidos en `GENERATE_ANGLES`.

## Formato de los datos de entrada

Cada archivo `.mat` debe contener una rejilla con las matrices habituales del
proyecto:

- `tl` o variantes compatibles como `TL` o `tL`;
- `X` o `R` para coordenadas horizontales;
- `Z` para coordenadas verticales.

El lector toma los datos con la convención del repositorio y luego transpone el
campo TL para trabajar con la orientación esperada por el resto del código.

El ángulo del plano se extrae del nombre del archivo con el patrón
`PlaneAngle...`.

## Idea general del entrenamiento

El entrenamiento no usa la imagen completa como una única muestra. En su lugar,
para cada ROI:

1. Se genera una lista de puntos `(x, z, angle)` dentro del ROI.
2. A cada punto se le asocia el TL real de ese píxel.
3. Esos puntos se mezclan y se entregan a la red como muestras independientes.

Eso permite tratar el problema como una aproximación de una función continua,
no como una clasificación ni como una simple regresión sobre imágenes planas.

### Por qué se hace así

- la red aprende una relación continua entre posición, ángulo y TL;
- se puede evaluar sobre cualquier malla compatible con el ROI;
- la física se puede imponer con derivadas automáticas sobre coordenadas;
- el mismo modelo sirve para reconstruir planos completos y para interpolar.

## Extracción de ROIs

La extracción de ROIs ocurre en [data_utils.py](data_utils.py). El flujo es:

1. Se recorre la carpeta `input/` buscando archivos `.mat` que contengan
	 `PlaneAngle` en el nombre.
2. Para cada archivo se cargan `tl`, `X` y `Z`.
3. Se calcula el tamaño real del ROI en píxeles.
4. Se elige el modo de extracción según `ROI_MODE`.
5. Se obtienen una o varias ROIs por plano.
6. Se calcula el `extent` físico de cada ROI para poder dibujarlo o reconstruirlo.

### Modos de ROI

`ROI_MODE` controla cómo se selecciona la región:

- `center_max`: busca el máximo de TL y centra la ROI sobre esa zona.
- `corner_fixed`: usa una esquina física fija definida por `ROI_CORNER`.

`corner_fixed` es el modo más determinista, porque siempre recorta en la misma
posición física si la geometría de entrada lo permite.

### Parámetros relevantes

- `ROI_HEIGHT` y `ROI_WIDTH`: tamaño de la ventana de trabajo.
- `ROIS_PER_PLANE`: cuántas ROIs se sacan de cada archivo.
- `ROI_CORNER`: esquina física usada cuando `ROI_MODE = "corner_fixed"`.

## Normalización

Hay dos normalizaciones distintas:

### 1. Normalización del TL

La clase `Normalizer` calcula:

- `tl_min` y `tl_max`;
- `angle_min` y `angle_max`.

Luego escala TL a `[0, 1]` y también normaliza el ángulo a `[0, 1]`.

Esto es lo que la red ve como objetivo de regresión durante entrenamiento.

### 2. Normalización de coordenadas

La clase `CoordNormalizer` lleva `x`, `z` y `angle` a `[-1, 1]`.

Esto es importante porque:

- estabiliza la optimización;
- facilita el uso de activaciones como `tanh`;
- hace que las derivadas respecto a coordenadas sean más manejables.

En el checkpoint se guarda toda la información necesaria para reproducir esta
normalización durante inferencia.

## Arquitectura de la red

La red en [model.py](model.py) es un MLP compacto:

- primera capa lineal desde 3 entradas;
- varias capas ocultas de tamaño `PINN_HIDDEN_DIM`;
- activación elegida por `PINN_ACTIVATION`;
- opcionalmente dropout con `PINN_DROPOUT`;
- capa final lineal a un único valor.

La red no tiene convoluciones, atención ni bloques especiales. Eso la hace más
fácil de entender y, sobre todo, más fácil de derivar con respecto a `x` y `z`.

## Cómo funciona el entrenamiento paso a paso

El flujo de [train.py](train.py) es este:

### Paso 1: semillas y dispositivo

Se fijan semillas en NumPy y PyTorch para hacer la ejecución más reproducible.
Después se decide si se usa CPU o GPU.

### Paso 2: cargar ROIs

Se llama a [load_all_rois](data_utils.py#L243) y se obtiene:

- `rois`;
- `roi_angles`;
- `roi_extents`.

Si no hay ROIs válidas, el entrenamiento se detiene.

### Paso 3: construir normalizadores

Se crea el normalizador [Normalizer](data_utils.py#L275) con `rois` y
`roi_angles` para fijar los rangos de TL y ángulo.
Luego se normalizan las ROIs.

### Paso 4: augmentación opcional

Si `USE_AUGMENTATION = True`, se generan copias con:

- ruido gaussiano;
- escalado de contraste;
- desplazamiento pequeño.

Esto multiplica el tamaño del conjunto, pero conserva el ángulo asociado.

### Paso 5: calcular bounds globales

Con todas las extents se calcula un dominio global para muestrear puntos
colocacionales y para construir la normalización espacial.

### Paso 6: interfaz física

Si el parámetro [INTERFACE_Z](config.py#L36) vale `None`, se usa el punto medio
del rango Z como interfaz.
Esto es práctico cuando la geometría es desconocida o variable.

### Paso 7: crear la red y el optimizador

Se instancia [PINN](model.py#L16) y se usa Adam con [LEARNING_RATE](config.py#L15).

### Paso 8: muestreo por época

En cada época el código hace dos cosas:

- [sample_points_from_rois](data_utils.py#L505) para construir el batch de datos;
- [sample_collocation_points](train.py#L53) para los puntos físicos.

El primer muestreo apunta a supervisión directa. El segundo no necesita verdad
terreno: solo necesita evaluar el residuo PDE.

### Paso 9: pérdida total

La pérdida final se compone de:

```text
loss = DATA_WEIGHT * data_loss + PHYSICS_WEIGHT * (physics_loss + INTERFACE_WEIGHT * interface_loss)
```

Si `PDE_TYPE = "none"`, la parte física queda desactivada o trivialmente cero.

### Paso 10: registro de métricas

Se almacenan por época:

- pérdida total;
- pérdida de datos;
- pérdida física;
- pérdida de interfaz.

### Paso 11: guardar resultados

Al terminar:

- se exporta la curva de entrenamiento en PNG;
- se guardan los ROIs originales como `.mat`;
- se comparan originales y predichos sobre entrenamiento;
- se guarda el checkpoint completo en [MODEL_PATH](config.py#L68).

## Qué guarda el checkpoint

El checkpoint no guarda solo pesos. También conserva metadatos importantes:

- arquitectura de la red;
- parámetros físicos;
- límites de normalización del TL;
- límites de normalización de coordenadas;
- configuración geométrica básica.

Esto permite que [generate.py](generate.py) reconstruya el modelo sin depender
de valores externos que podrían desincronizarse.

### Cómo funciona la generación

La inferencia en [generate.py](generate.py) sigue esta lógica:

1. Crea las carpetas de salida.
2. Llama a [load_inference_pipeline](data_utils.py#L559).
3. Obtiene el modelo entrenado, los ROIs y las normalizaciones.
4. Para cada ángulo de [GENERATE_ANGLES](config.py#L53) busca el ROI de
	entrenamiento más cercano.
5. Evalúa la red sobre la malla completa de ese ROI.
6. Desnormaliza el resultado a TL físico.
7. Calcula errores respecto al plano de referencia escogido.
8. Guarda PNG y `.mat`.

### Importante: qué usa como referencia

Para cada ángulo objetivo, el script toma el ROI de entrenamiento cuyo ángulo
sea más cercano. Esa muestra se usa como referencia visual y para calcular
errores. No es una predicción sobre un plano real independiente, sino una
comparación contra el vecino más próximo del conjunto entrenado.

## Validación

Si existe la carpeta `input/validation/`, el entrenamiento también evalúa esos
planos después de guardar el checkpoint.

La validación hace dos cosas:

- compara plano real vs plano predicho;
- analiza el error en función del hueco angular entre vecinos de entrenamiento.

Ese segundo análisis es útil porque ayuda a entender si el modelo falla más
cuando el ángulo de validación está lejos de los ángulos vistos durante
entrenamiento.

## Parámetros que más conviene revisar

### Geometría y extracción

- [ROI_HEIGHT](config.py#L4)
- [ROI_WIDTH](config.py#L5)
- [ROIS_PER_PLANE](config.py#L6)
- [ROI_MODE](config.py#L9)
- [ROI_CORNER](config.py#L10)

### Entrenamiento

- [SEED](config.py#L13)
- [BATCH_SIZE](config.py#L14)
- [EPOCHS](config.py#L15)
- [LEARNING_RATE](config.py#L16)
- [USE_AUGMENTATION](config.py#L17)
- [POINTS_PER_ROI](config.py#L18)

### Física

- [PHYSICS_BATCH_SIZE](config.py#L21)
- [DATA_WEIGHT](config.py#L22)
- [PHYSICS_WEIGHT](config.py#L23)
- [INTERFACE_BATCH_SIZE](config.py#L24)
- [INTERFACE_WEIGHT](config.py#L25)
- [PDE_TYPE](config.py#L26)
- [HELMHOLTZ_K](config.py#L27)

### Parámetros acústicos

- [FREQUENCY_HZ](config.py#L30)
- [AIR_SOUND_SPEED](config.py#L31)
- [AIR_DENSITY](config.py#L32)
- [WATER_SOUND_SPEED](config.py#L33)
- [WATER_DENSITY](config.py#L34)
- [AIR_ABOVE_INTERFACE](config.py#L35)
- [INTERFACE_Z](config.py#L36)
- [INTERFACE_EPS](config.py#L37)
- [TL_REF_PRESSURE](config.py#L38)
- [OUTPUT_IS_TL](config.py#L39)
- [TL_DB_SIGN](config.py#L40)
- [TL_DB_OFFSET](config.py#L41)
- [TL_DB_CLAMP_MIN](config.py#L42)
- [TL_DB_CLAMP_MAX](config.py#L43)

### Arquitectura de red

- [PINN_HIDDEN_DIM](config.py#L46)
- [PINN_NUM_LAYERS](config.py#L47)
- [PINN_ACTIVATION](config.py#L48)
- [PINN_DROPOUT](config.py#L49)

### Inferencia

- [INFER_BATCH_SIZE](config.py#L52)
- [GENERATE_ANGLES](config.py#L53)

## Interpretación práctica de algunos parámetros

### [PDE_TYPE](config.py#L26)

Es el interruptor principal del comportamiento físico. Si quieres una versión
más simple del modelo, empieza por `none`. Si quieres una restricción suave,
usa `laplace` o `helmholtz`. Si tu escena tiene dos medios, `two_layer_helmholtz`
es la opción más completa.

### [OUTPUT_IS_TL](config.py#L39)

Cuando vale `True`, la física no se aplica directamente sobre la salida de la
red, sino sobre una conversión de TL a presión. Esto suele ser más coherente
con el significado físico de las ecuaciones.

### [TL_DB_SIGN](config.py#L40)

En muchos datos de TL los valores vienen negativos. En ese caso `TL_DB_SIGN = -1.0`
permite convertir la escala de forma consistente antes de pasar a presión.

### [TL_DB_CLAMP_MIN](config.py#L42) y [TL_DB_CLAMP_MAX](config.py#L43)

Estos límites recortan el TL en dB antes de convertirlo a presión. El clamp
protege contra valores extremos que, al pasar por la exponencial de la
conversión acústica, pueden volver inestable el gradiente o hacer que una sola
muestra domine la pérdida física.

### [INTERFACE_Z](config.py#L36)

Si no la fijas manualmente, el código usa el punto medio del ROI. Eso evita
errores de configuración cuando la interfaz no está marcada explícitamente,
aunque no siempre será la elección física correcta.
Si el valor queda fuera del ROI, el residuo físico cae automáticamente a
Helmholtz de un solo medio para ese dominio, en vez de forzar una interfaz que
no existe dentro de la muestra.

### [INTERFACE_EPS](config.py#L37)

Controla qué tan separados quedan los dos puntos usados para imponer la
continuidad en la interfaz. Si es demasiado pequeño, la derivada puede ser
numéricamente inestable; si es demasiado grande, ya no estás evaluando justo la
interfaz.

## Salidas generadas

### Durante entrenamiento

- `output/train/PNG/training_curve_pinn.png`
- `output/train/PNG/compare_train_*.png`
- `output/train/MAT/roi_original_*.mat`
- `output/pinn_model.pt`

### Durante generación

- `output/generate/PNG/plane_angle_*.png`
- `output/generate/MAT/plane_angle_*.mat`

### Durante validación

- `output/validation/PNG/validation_*.png`
- `output/validation/PNG/error_vs_gap.png`
- `output/validation/MAT/val_generated_*.mat`

## Qué mirar al revisar el código a fondo

Si el objetivo es auditar la lógica del programa, el orden más útil es este:

1. [config.py](config.py) para entender qué se puede ajustar.
2. [data_utils.py](data_utils.py) para seguir la carga y la construcción de muestras.
3. [model.py](model.py) para ver la red mínima.
4. [train.py](train.py) para entender el ciclo de entrenamiento y la física.
5. [generate.py](generate.py) para ver cómo se reconstruye el campo completo.

## Advertencias prácticas

- Este PINN no es un solucionador físico puro; es una red de datos con un
	término físico regularizador.
- Si los pesos físicos son demasiado agresivos, puede empeorar la fidelidad al
	dato observado.
- Si los `ROI` no están bien alineados con la geometría física, el modelo puede
	aprender relaciones inconsistentes.
- La calidad de la inferencia depende mucho de que el checkpoint y la
	configuración actual coincidan.

## Resumen corto del flujo

1. Leer archivos `.mat` de `input/`.
2. Extraer ROIs y ángulos.
3. Normalizar TL, coordenadas y ángulos.
4. Entrenar el MLP con pérdida de datos.
5. Añadir física opcional mediante PDE e interfaz.
6. Guardar el checkpoint y gráficos.
7. Reabrir el checkpoint para generar planos completos.

Si quieres, el siguiente paso puede ser añadir también una sección de
"explicación línea por línea" en el propio README para las funciones más
importantes de [train.py](train.py) y [data_utils.py](data_utils.py).
