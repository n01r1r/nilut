"""
Training script for fitting Hald-to-Hald color transforms.

Supported architectures:
- nilut: residual MLP baseline
- siren: SIREN RGB regression baseline
- geometric: SIREN backbone + 7D geometric transform head
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import gc
from collections import defaultdict
import argparse

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

from utils import start_timer, clean_mem
from utils import save_rgb, plot_all, pt_psnr, np_psnr, deltae_dist, count_parameters
from dataloader import LUTFitting
# Import the new GeometricNiLUT
from models.archs import SIREN, NILUT, GeometricNiLUT


def fit(lut_model, total_steps, model_input, ground_truth, img_size, opt, verbose=200):
    """
    Simple training loop.
    """
    start_timer()
    metrics = defaultdict(list)
    print(f"\n** Start training for {total_steps} iterations\n")

    for step in range(total_steps):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = lut_model(model_input)
            model_output = out[0] if isinstance(out, tuple) else out

            # Loss Function: L1 Loss is generally more stable for color regression than L2
            loss = torch.mean(torch.abs(model_output - ground_truth))
            _psnr = pt_psnr(ground_truth, model_output).item()

        metrics["mse"].append(loss.item())
        metrics["psnr"].append(_psnr)

        if (step % verbose) == 0:
            print(f">> Step {step} , loss={loss:.6f}, psnr={_psnr:.4f}")

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

    try:
        import matplotlib.pyplot as plt

        plt.plot(metrics["psnr"])
        plt.title("PSNR Evolution")
        plt.show()
    except Exception:
        pass

    print("\n**Evaluate and get performance metrics\n")
    eval(model_input, model_output, ground_truth, img_size)

    torch.save(lut_model.state_dict(), "3dlut.pt")
    clean_mem()


def eval(model_input, model_output, ground_truth, img_size):
    """
    Get performance metrics PSNR and DeltaE for the RGB transformation.
    """
    original_inp = model_input.cpu().view(img_size[0], img_size[1], 3).numpy().astype(np.float32)
    np_out = model_output.cpu().view(img_size[0], img_size[1], 3).detach().numpy().astype(np.float32)
    np_gt = ground_truth.cpu().view(img_size[0], img_size[1], 3).detach().numpy().astype(np.float32)
    np_diff = np.abs(np_gt - np_out)

    psnr = np_psnr(np_gt, np_out)
    deltae = deltae_dist(np_gt, np_out)

    print(
        f"Final metrics >> PSNR={psnr:.4f}, DeltaE={deltae:.4f} --- "
        f"min error {np.min(np_diff):.4f}, max error {np.max(np_diff):.4f}"
    )

    # Save visual results
    save_rgb(original_inp, "results/inp.png")
    save_rgb(np_out, "results/out.png")
    save_rgb(np_gt, "results/gt.png")


def main(inp_path, out_path, total_steps, lut_size, arch):
    """
    Fit a color transform from source/target Hald pair.
    """
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Start fitting arch={arch}, hidden={lut_size}")
    print("Input hald image:", inp_path)
    print("Target hald image:", out_path)

    # Define the dataloader
    lut_images = LUTFitting(inp_path, out_path)
    dataloader = DataLoader(lut_images, batch_size=1, pin_memory=True, num_workers=0)
    img_size = lut_images.shape
    print("\nDataloader ready", img_size)

    # Define the model architecture based on arguments
    if arch == "nilut":
        lut_model = NILUT(
            in_features=3,
            out_features=3,
            hidden_features=lut_size[0],
            hidden_layers=lut_size[1],
        )
    elif arch == "siren":
        # Baseline SIREN regression to RGB
        lut_model = SIREN(
            in_features=3,
            out_features=3,
            hidden_features=lut_size[0],
            hidden_layers=lut_size[1],
            outermost_linear=True,
        )
    elif arch == "geometric":
        # New Geometric Architecture
        print(">> Initializing GeometricNiLUT with 7D Geometric Head (Scale, Translation, Rotor)")
        lut_model = GeometricNiLUT(
            in_features=3,
            hidden_features=lut_size[0],
            hidden_layers=lut_size[1],
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")

    lut_model = lut_model.to(device)
    opt = torch.optim.Adam(lr=1e-3, params=lut_model.parameters())

    print(f"\nCreated model arch={arch} {lut_size} -- params={count_parameters(lut_model)}")

    # Load in memory the input and target hald images
    model_input_cpu, ground_truth_cpu = next(iter(dataloader))
    model_input, ground_truth = model_input_cpu.to(device), ground_truth_cpu.to(device)

    lut_model.train()
    fit(lut_model, total_steps, model_input, ground_truth, img_size, opt)


parser = argparse.ArgumentParser(description="NILUT fitting")
parser.add_argument("--input", help="Input RGB map as a hald image", default="./dataset/halds/Original_Image.png", type=str)
parser.add_argument(
    "--target",
    help="Enhanced RGB map as a hald image, after using the desired 3D LUT",
    default="./dataset/halds/LUT01_ContrastLUT.png",
    type=str,
)
parser.add_argument("--steps", help="Number of optimization steps", default=2000, type=int)
parser.add_argument("--units", help="Number of hidden units", default=256, type=int)
parser.add_argument("--layers", help="Number of hidden layers", default=3, type=int)
parser.add_argument(
    "--arch",
    help="Model architecture: nilut | siren | geometric",
    default="nilut",
    type=str,
    choices=["nilut", "siren", "geometric"],
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(
        inp_path=args.input,
        out_path=args.target,
        total_steps=args.steps,
        lut_size=(args.units, args.layers),
        arch=args.arch,
    )
