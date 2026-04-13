## Install

```bash
$ pip install denoising_diffusion_pytorch
```

## Intento_Transformers MAT/H5 workflow

This repository now includes a ready pipeline to train and generate on
`PlaneAngle*.mat` files from `../Intento_Transformers/input`.

New scripts:

- `intento_mat_utils.py`: ROI extraction + `.mat/.h5` IO compatible with
    the Intento_Transformers dataset format.
- `train_intento_mat.py`: training with model and algorithm switches.
- `generate_intento_mat.py`: conditioned generation to PNG + MAT.
- `validation.py`: standalone validation report (real vs generated vs error).

### Train (UNet or DiT, GaussianDiffusion or FlowMatching)

Example DiT + GaussianDiffusion:

```bash
python train_intento_mat.py \
    --model-type dit \
    --dit-variant DiT-XXS/8 \
    --algorithm gaussian \
    --amp \
    --mixed-precision-type bf16 \
    --compile-model \
    --train-height 256 \
    --train-width 512 \
    --results-folder results/intento_mat/dit_gaussian
```

Example UNet + FlowMatching:

```bash
python train_intento_mat.py \
    --model-type unet \
    --algorithm flowmatching \
    --flow-num-steps 250 \
    --amp \
    --mixed-precision-type bf16 \
    --compile-model \
    --train-height 256 \
    --train-width 512 \
    --results-folder results/intento_mat/unet_flow
```

Notes:

- Default ROI extraction mirrors Intento_Transformers (`corner_fixed`, corner `(0, 10)`).
- `Trainer` in this codebase requires at least 100 samples; the script applies a
    minimal safe augmentation/replication when needed.
- For DiT with `--dit-attn-type vanilla`, the script now auto-adjusts to a larger
    patch variant when token count would exceed `--max-vanilla-attn-tokens`
    (default: 4096).
- A full validation report is generated automatically at the end of training.
    You can skip it with `--skip-post-validation`.
- You can change ROI and conditioning behavior with `--roi-*` arguments.

### Standalone validation report

```bash
python validation.py \
        --results-folder results/intento_mat/dit_gaussian \
        --prefer-ema \
        --sampler ddim \
        --num-inference-steps 300
```

Validation outputs are saved under `results/.../validation/{PNG,MAT}`.
Plots and exported MAT grids use physical coordinates in meters (`X [m]`, `Z [m]`).

### Generate conditioned planes (PNG + MAT)

```bash
python generate_intento_mat.py \
    --results-folder results/intento_mat/dit_gaussian \
    --prefer-ema \
    --angles "95,100,110,120,130,140,150,160,170,180" \
    --sampler ddim \
    --num-inference-steps 300 \
    --with-reference
```

For Slurm runs in this repository, use `train_transformer.sh`, `gen_transformer.sh`, and `val_transformer.sh`.

Outputs are saved under:

- `results/.../generate/PNG`
- `results/.../generate/MAT`

Generated plots and MAT grids use physical coordinates in meters (`X [m]`, `Z [m]`).

## Usage

```python
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)

training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
```

Or, if you simply want to pass in a folder name and the desired image dimensions, you can use the `Trainer` class to easily train a model.

```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()
```

Samples and model checkpoints will be logged to `./results` periodically

## Multi-GPU Training

The `Trainer` class is now equipped with <a href="https://huggingface.co/docs/accelerate/accelerator">🤗 Accelerator</a>. You can easily do multi-gpu training in two steps using their `accelerate` CLI

At the project root directory, where the training script is, run

```python
$ accelerate config
```

Then, in the same directory

```python
$ accelerate launch train.py
```

## Miscellaneous

### 1D Sequence

By popular request, a 1D Unet + Gaussian Diffusion implementation.

```python
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 128,
    timesteps = 1000,
    objective = 'pred_v'
)

training_seq = torch.rand(64, 32, 128) # features are normalized from 0 to 1

loss = diffusion(training_seq)
loss.backward()

# Or using trainer

dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()

# after a lot of training

sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)

```

`Trainer1D` does not evaluate the generated samples in any way since the type of data is not known.

You could consider adding a suitable metric to the training loop yourself after doing an editable install of this package
`pip install -e .`.