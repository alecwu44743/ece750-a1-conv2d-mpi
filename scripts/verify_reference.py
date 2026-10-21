#!/usr/bin/env python3
"""Verify the MPI convolution against independent reference implementations.

For each test configuration this script runs `bin/conv2d_run --dump DIR` under
mpirun, loads the dumped input / weight / bias / output tensors and compares the
output with
  * torch.nn.functional.conv2d          (if PyTorch is installed)
  * scipy.signal.correlate2d            (sum over input channels, + bias, then stride)
Both are cross-correlations, which is what nn.Conv2d (and our code) computes.

Usage:  python3 scripts/verify_reference.py [--np 4] [--mpirun mpirun] [--image file.pgm]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN = os.path.join(ROOT, "bin", "conv2d_run")

# (description, extra conv2d_run arguments)
CONFIGS = [
    ("edge 3x3, p=0, s=1 (minimum requirement)", ["--size", "257x311", "--filter", "edge"]),
    ("gaussian 7x7, p=3 ('same')", ["--size", "200", "--filter", "gaussian", "--kernel-size", "7", "--padding", "3"]),
    ("random 5x5, s=2, p=2, bias", ["--size", "128x96", "--filter", "random", "--kernel-size", "5",
                                     "--stride", "2", "--padding", "2", "--bias"]),
    ("multi-channel 3->8, 3x3, p=1, bias, im2col", ["--size", "64", "--cin", "3", "--cout", "8", "--filter",
                                                   "random", "--padding", "1", "--bias", "--algo", "im2col"]),
    ("multi-channel 4->6, 4x4, s=3, im2col", ["--size", "99x77", "--cin", "4", "--cout", "6", "--filter",
                                              "random", "--kernel-size", "4", "--stride", "3", "--algo", "im2col"]),
]


def ref_scipy(x, w, b, stride, padding):
    from scipy.signal import correlate2d
    cout, cin = w.shape[:2]
    xp = np.pad(x.astype(np.float64), ((0, 0), (padding, padding), (padding, padding)))
    out = []
    for co in range(cout):
        acc = sum(correlate2d(xp[ci], w[co, ci].astype(np.float64), mode="valid") for ci in range(cin))
        if b is not None:
            acc = acc + b[co]
        out.append(acc[::stride, ::stride])
    return np.stack(out)


def ref_torch(x, w, b, stride, padding):
    import torch
    t = torch.nn.functional.conv2d(
        torch.from_numpy(x)[None].double(), torch.from_numpy(w).double(),
        None if b is None else torch.from_numpy(b).double(), stride=stride, padding=padding)
    return t[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--mpirun-args", default="--oversubscribe")
    ap.add_argument("--image", help="optional PGM image to add as an extra test")
    args = ap.parse_args()

    configs = list(CONFIGS)
    if args.image:
        configs.append((f"image {os.path.basename(args.image)} edge 3x3", ["--image", args.image, "--filter", "edge"]))

    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
        print("PyTorch not found - comparing with SciPy only")

    failures = 0
    for name, extra in configs:
        with tempfile.TemporaryDirectory() as d:
            cmd = [args.mpirun, *args.mpirun_args.split(), "-np", str(args.np), RUN, "--repeat", "1",
                   "--dump", d, *extra]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            x = np.load(os.path.join(d, "input.npy"))
            w = np.load(os.path.join(d, "weight.npy"))
            y = np.load(os.path.join(d, "output.npy"))
            prm = json.load(open(os.path.join(d, "params.json")))
            b = np.load(os.path.join(d, "bias.npy")) if prm["bias"] else None

        refs = {"scipy": ref_scipy(x, w, b, prm["stride"], prm["padding"])}
        if have_torch:
            refs["torch"] = ref_torch(x, w, b, prm["stride"], prm["padding"])
        line = []
        for lib, r in refs.items():
            ok = r.shape == y.shape and np.allclose(y, r, rtol=1e-4, atol=1e-4)
            err = np.abs(y - r).max() if r.shape == y.shape else float("inf")
            line.append(f"{lib}: {'PASS' if ok else 'FAIL'} (max|err|={err:.2e})")
            failures += not ok
        print(f"[P={args.np}] {name:45s} out={tuple(y.shape)}  " + "  ".join(line))

    print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
