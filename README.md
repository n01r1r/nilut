# nilut (fork, in progress)

This repository is a working fork of NILUT for experiments on color transformation models.
The current focus is replacing direct RGB regression with a geometric parameterization while
keeping the same input/output interface used by NILUT fitting (`[B, 3] -> [B, 3]`).

## Current intent

- Keep the original NILUT/SIREN fitting workflow.
- Add a geometric variant that predicts transformation parameters instead of output RGB directly.
- Implement geometric rotation with pure PyTorch only (no external geometric algebra packages).

## Implemented model variants

`fit.py` currently supports:

- `--arch nilut`: original residual MLP (`NILUT`)
- `--arch siren`: SIREN baseline (RGB regression)
- `--arch geometric`: `GeometricNiLUT` (SIREN backbone + geometric transform head)

## GeometricNiLUT (implemented)

`GeometricNiLUT` is defined in `models/archs.py`.

Backbone:

- Reuses SIREN feature mapping
- Final output dimension is `7` (not RGB)

Predicted parameters:

- `s in R^1`: scaling (positive via `softplus`)
- `t in R^3`: translation
- `b in R^3`: rotor bivector parameters

Forward transformation:

- Convert `b` to a unit quaternion rotor with exponential map:
  - `theta = ||b||`
  - `R = cos(theta) + (b / theta) * sin(theta)`
- Apply sandwich rotation on input color vector `x`:
  - `x_rot = R * x * R_dagger`
- Final output:
  - `x_out = s * x_rot + t`
- Output is clamped to `[0, 1]` to match the existing NILUT pipeline.

All quaternion operations are implemented in PyTorch (`torch`) inside `models/archs.py`.

## Data requirement for fitting

`fit.py` expects a pair of Hald images:

- input Hald image (source RGB map)
- target Hald image (after applying desired LUT/style transform)

No dataloader format changes are required.

## Run

```bash
# Original NILUT
python fit.py --arch nilut --input <input_hald.png> --target <target_hald.png> --steps 1000 --units 128 --layers 2

# SIREN baseline
python fit.py --arch siren --input <input_hald.png> --target <target_hald.png> --steps 1000 --units 128 --layers 2

# Geometric model
python fit.py --arch geometric --input <input_hald.png> --target <target_hald.png> --steps 1000 --units 128 --layers 2
```

Notes:

- `fit.py` uses `--input` and `--target`.
- In environments where `matplotlib` is unavailable, plotting is skipped, and training still runs.

## Upstream reference

Original NILUT paper/repository:

- Paper: https://arxiv.org/abs/2306.11920
- Upstream repo: https://github.com/mv-lab/nilut

