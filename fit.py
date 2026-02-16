"""
Training script for fitting Hald-to-Hald color transforms.
Supports: nilut, siren, geometric (7-DoF), geometric_affine (12-DoF)
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import gc
from collections import defaultdict
import argparse

# Device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()
if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
else:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

from utils import start_timer, clean_mem
from utils import save_rgb, plot_all, pt_psnr, np_psnr, deltae_dist, count_parameters
from dataloader import LUTFitting
from models.archs import SIREN, NILUT, GeometricNiLUT, GeometricAffineNiLUT

def fit(lut_model, total_steps, model_input, ground_truth, img_size, opt, verbose=200):
    start_timer()
    metrics  = defaultdict(list)
    print(f"\n** Start training for {total_steps} iterations **\n")

    for step in range(total_steps):
        opt.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            model_output, _ = lut_model(model_input)
            # L1 Loss is robust for color regression
            loss = torch.mean(torch.abs(model_output - ground_truth))
            
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        # Logging
        if step % verbose == 0 or step == total_steps - 1:
            with torch.no_grad():
                _psnr = pt_psnr(ground_truth, model_output).item()
                metrics['mse'].append(loss.item())
                metrics['psnr'].append(_psnr)
                print(f">> Step {step:04d} | Loss: {loss:.6f} | PSNR: {_psnr:.2f}dB")

    # Final Eval
    print("\n** Evaluation **")
    eval(model_input, model_output, ground_truth, img_size)
    
    # Save Model
    save_path = f"weights_{lut_model.__class__.__name__}.pt"
    torch.save(lut_model.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    clean_mem()

def eval(model_input, model_output, ground_truth, img_size):
    original_inp = model_input.cpu().view(img_size[0],img_size[1],3).numpy().astype(np.float32)
    np_out       = model_output.cpu().view(img_size[0],img_size[1],3).detach().numpy().astype(np.float32)
    np_gt        = ground_truth.cpu().view(img_size[0],img_size[1],3).detach().numpy().astype(np.float32)
    np_diff      = np.abs(np_gt - np_out)
    
    psnr = np_psnr(np_gt, np_out)
    deltae = deltae_dist(np_gt, np_out)
    
    print(f"Final Metrics >> PSNR: {psnr:.4f} | DeltaE: {deltae:.4f}")
    print(f"Errors >> Min: {np.min(np_diff):.4f} | Max: {np.max(np_diff):.4f}") 
    
    try:
        save_rgb(original_inp, f"results/inp.png")
        save_rgb(np_out,       f"results/out.png")
        save_rgb(np_gt ,       f"results/gt.png")
    except Exception as e:
        print(f"[Warning] Failed to save result images: {e}")

def main(args):
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Mode: {args.arch} | Hidden: ({args.units}, {args.layers})")
    
    # 1. Load Data
    lut_images = LUTFitting(args.input, args.target)
    dataloader = DataLoader(lut_images, batch_size=1, pin_memory=True, num_workers=0)
    img_size = lut_images.shape
    model_input_cpu, ground_truth_cpu = next(iter(dataloader))
    model_input = model_input_cpu.to(device)
    ground_truth = ground_truth_cpu.to(device)

    # 2. Init Model
    if args.arch == "nilut":
        model = NILUT(hidden_features=args.units, hidden_layers=args.layers)
    elif args.arch == "siren":
        model = SIREN(hidden_features=args.units, hidden_layers=args.layers, outermost_linear=True)
    elif args.arch == "geometric":
        model = GeometricNiLUT(hidden_features=args.units, hidden_layers=args.layers)
    elif args.arch == "geometric_affine":
        model = GeometricAffineNiLUT(hidden_features=args.units, hidden_layers=args.layers)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    model = model.to(device)
    print(f"Parameters: {count_parameters(model)}")

    # 3. Optimize
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    fit(model, args.steps, model_input, ground_truth, img_size, opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./dataset/halds/Original_Image.png", type=str)
    parser.add_argument("--target", default="./dataset/halds/LUT01_ContrastLUT.png", type=str)
    parser.add_argument("--steps", default=2000, type=int)
    parser.add_argument("--units", default=256, type=int)
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--arch", default="nilut", type=str, 
                        choices=["nilut", "siren", "geometric", "geometric_affine"])
    args = parser.parse_args()
    
    main(args)
