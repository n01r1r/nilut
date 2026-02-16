import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

from tqdm import tqdm


STEP_RE = re.compile(r">> Step\s+(\d+)")
METRIC_RE = re.compile(r"Final Metrics >> PSNR:\s*([0-9.]+)\s*\|\s*DeltaE:\s*([0-9.]+)")
ERROR_RE = re.compile(r"Errors >> Min:\s*([0-9.]+)\s*\|\s*Max:\s*([0-9.]+)")


def run_one(arch: str, args: argparse.Namespace) -> dict:
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{arch}_{args.steps}.log")

    cmd = [
        sys.executable,
        "fit.py",
        "--arch",
        arch,
        "--input",
        args.input,
        "--target",
        args.target,
        "--steps",
        str(args.steps),
        "--units",
        str(args.units),
        "--layers",
        str(args.layers),
    ]

    print(f"\n=== Running {arch} ===")
    print(" ".join(cmd))

    pbar = tqdm(total=args.steps, desc=arch, unit="step")
    last_step = 0
    psnr = None
    deltae = None
    min_err = None
    max_err = None

    with open(log_path, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None
        for line in process.stdout:
            f.write(line)
            f.flush()

            m = STEP_RE.search(line)
            if m:
                # fit.py logs zero-based step index; convert to completed steps
                completed = int(m.group(1)) + 1
                if completed > last_step:
                    pbar.update(completed - last_step)
                    last_step = completed

            m = METRIC_RE.search(line)
            if m:
                psnr = float(m.group(1))
                deltae = float(m.group(2))

            m = ERROR_RE.search(line)
            if m:
                min_err = float(m.group(1))
                max_err = float(m.group(2))

        return_code = process.wait()

    if last_step < args.steps:
        pbar.update(args.steps - last_step)
    pbar.close()

    if return_code != 0:
        raise RuntimeError(f"{arch} failed with exit code {return_code}. See log: {log_path}")

    return {
        "arch": arch,
        "psnr": psnr,
        "deltae": deltae,
        "min_err": min_err,
        "max_err": max_err,
        "log_path": log_path,
    }


def print_table(results: list[dict]) -> None:
    headers = ["arch", "psnr", "deltae", "min_err", "max_err", "log_path"]
    rows = []
    for r in results:
        rows.append(
            [
                r["arch"],
                f"{r['psnr']:.4f}" if r["psnr"] is not None else "n/a",
                f"{r['deltae']:.4f}" if r["deltae"] is not None else "n/a",
                f"{r['min_err']:.4f}" if r["min_err"] is not None else "n/a",
                f"{r['max_err']:.4f}" if r["max_err"] is not None else "n/a",
                r["log_path"],
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "-+-".join("-" * w for w in widths)
    print("\n" + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run geometric vs geometric_affine A/B experiments with tqdm.")
    parser.add_argument("--input", default="./dataset/halds/Original_Image.png", type=str)
    parser.add_argument("--target", default="./dataset/halds/LUT01_ContrastLUT.png", type=str)
    parser.add_argument("--steps", default=3000, type=int)
    parser.add_argument("--units", default=256, type=int)
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--log-dir", default="./results/logs", type=str)
    args = parser.parse_args()

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started at {started}")
    print(f"Python: {sys.executable}")

    results = []
    for arch in ("geometric", "geometric_affine"):
        results.append(run_one(arch, args))

    print_table(results)


if __name__ == "__main__":
    main()
