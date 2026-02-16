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
- `--arch geometric_affine`: `GeometricAffineNiLUT` (full 3x3 affine + translation)

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

## Geometric A/B result summary

Latest A/B run compares:

- `geometric` (7-DoF: scale + rotor + translation)
- `geometric_affine` (12-DoF: full affine matrix + translation)

Model intent:

- `geometric`: constrained local conformal transform (rotation + uniform scaling + translation)
- `geometric_affine`: unconstrained local linear transform (allows shear and non-uniform scaling)

Run setup:

- Target transform: `LUT01_ContrastLUT`
- Steps: `3000`
- Hidden size / depth: `256 / 3`
- Script: `run_ab_experiments.py`
- Device: CUDA (`torch 2.0.1+cu118`) on RTX 3090

Final metrics:

| Arch | Params | PSNR | DeltaE | Max Error |
| --- | ---: | ---: | ---: | ---: |
| `geometric` | 200,199 | **52.9287** | **0.4836** | **0.0175** |
| `geometric_affine` | 201,484 | 51.8299 | 0.5132 | 0.0196 |

Training behavior snapshot:

- `geometric`:
  - starts at `23.26 dB` (step 0), ends at `52.93 dB` (step 2999)
  - temporary spike around step `1800` (`41.15 dB`) but recovers
- `geometric_affine`:
  - starts at `23.24 dB` (step 0), ends at `51.83 dB` (step 2999)
  - temporary drop around step `1800` (`49.77 dB`) and partial recovery

Interpretation for this target (`LUT01_ContrastLUT`):

- The constrained conformal model (`geometric`) gave better final metrics than the higher-DoF affine model.
- This suggests the target mapping is sufficiently captured by rotor+scale+translation, and extra affine freedom did not improve final quality in this run.
- This is one target-specific result, not a global conclusion across all LUT styles.

Recommended follow-up checks:

- Repeat on multiple LUT targets (`LUT02`, `LUT03`, `LUT04`, ...).
- Keep seed, steps, and hidden size fixed for fair A/B comparison.
- Evaluate visual artifacts on ramps/noise images in addition to PSNR/DeltaE.

Logs:

- `results/logs/geometric_3000.log`
- `results/logs/geometric_affine_3000.log`

Weights:

- `weights_GeometricNiLUT.pt`
- `weights_GeometricAffineNiLUT.pt`

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

# Geometric affine model
python fit.py --arch geometric_affine --input <input_hald.png> --target <target_hald.png> --steps 1000 --units 128 --layers 2
```

```bash
# Run geometric vs affine A/B in one command
python run_ab_experiments.py --steps 3000 --units 256 --layers 3
```

Notes:

- `fit.py` uses `--input` and `--target`.
- In environments where `matplotlib` is unavailable, plotting is skipped, and training still runs.

## Upstream reference

Original NILUT paper/repository:

- Paper: https://arxiv.org/abs/2306.11920
- Upstream repo: https://github.com/mv-lab/nilut

