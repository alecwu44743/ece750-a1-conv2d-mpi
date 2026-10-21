#!/usr/bin/env python3
"""Turn the CSVs written by conv2d_bench into plots and a Markdown summary.

    python3 scripts/plot_results.py results

Metrics (P = number of MPI processes, T = median wall time over repetitions):
  speedup              S(P) = T(1) / T(P)
  parallel efficiency  E(P) = S(P) / P                    (strong scaling)
  weak efficiency      E(P) = T(1) / T(P), work/rank fixed (weak scaling)
  Karp-Flatt metric    e(P) = (1/S - 1/P) / (1 - 1/P)     (experimentally determined serial fraction)
  throughput           images / s                          (batch mode)
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PHASES = [("t_distribute_med", "distribute (bcast+scatter)"),
          ("t_compute_med", "compute"),
          ("t_collect_med", "collect (gather)")]


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if "." in v else int(v)
            except (ValueError, TypeError):
                pass
    return rows


def key(r):
    return (r["label"], r["mode"], r["algo"], r["H"], r["W"], r["cin"], r["cout"], r["kernel"], r["stride"], r["padding"])


def name(r):
    ch = f", {r['cin']}->{r['cout']}ch" if r["cin"] > 1 or r["cout"] > 1 else ""
    # spatial-local / spatial-halo = distributed-input modes; plain spatial = root scatter/gather
    dist = {"spatial-local": " [pre-distributed]", "spatial-halo": " [halo exchange]",
            "spatial": " [root scatter/gather]"}.get(r["mode"], "") if "dist" in str(r["label"]) else ""
    return f"{r['algo']} {r['H']}x{r['W']} k={r['kernel']}{ch}{dist}"


def karp_flatt(S, P):
    return (1 / S - 1 / P) / (1 - 1 / P) if P > 1 else float("nan")


def plot_latency(rows, out, md):
    if not rows:
        return
    Ps = sorted({r["P"] for r in rows})
    fig, axes = plt.subplots(1, len(Ps), figsize=(6 * len(Ps), 4.5), squeeze=False)
    md.append("## Latency vs image and kernel size\n")
    md.append("| P | image | kernel | total (ms) | compute (ms) | GFLOP/s |\n|---|---|---|---|---|---|\n")
    for ax, P in zip(axes[0], Ps):
        by_k = defaultdict(list)
        for r in rows:
            if r["P"] == P:
                by_k[r["kernel"]].append(r)
                md.append(f"| {P} | {r['H']}x{r['W']} | {r['kernel']}x{r['kernel']} | {r['t_total_med']*1e3:.2f} "
                          f"| {r['t_compute_med']*1e3:.2f} | {r['gflops']:.2f} |\n")
        for k, rs in sorted(by_k.items()):
            rs.sort(key=lambda r: r["H"] * r["W"])
            ax.plot([r["H"] * r["W"] / 1e6 for r in rs], [r["t_total_med"] * 1e3 for r in rs], "o-", label=f"{k}x{k}")
        ax.set(xscale="log", yscale="log", xlabel="image size (megapixels)", ylabel="latency (ms)",
               title=f"Latency, P={P}")
        ax.grid(True, which="both", alpha=.3)
        ax.legend(title="kernel")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "latency.png"), dpi=150)
    md.append("\n![latency](latency.png)\n\n")


def plot_scaling(rows, out, md, weak=False, tag=None, title=None):
    if not rows:
        return
    groups = defaultdict(list)
    for r in rows:
        k = key(r) if not weak else (r["label"], r["algo"], r["W"], r["kernel"], r["cin"], r["cout"])
        groups[k].append(r)
    tag = tag or ("weak" if weak else "strong")
    title = title or ("Weak" if weak else "Strong")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    md.append(f"## {title} scaling\n")
    hdr = "| config | P | total (ms) | compute (ms) | speedup | efficiency | compute eff. | Karp-Flatt | imbalance |\n"
    md.append(hdr + "|---|---|---|---|---|---|---|---|---|\n")
    maxP = 1
    for _, rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["P"])
        base = next((r for r in rs if r["P"] == 1), None)
        if base is None:
            continue
        Ps = [r["P"] for r in rs]
        maxP = max(maxP, max(Ps))
        lbl = name(base) if not weak else f"{base['algo']} {base['W']} wide, k={base['kernel']}"
        if weak:
            eff = [base["t_total_med"] / r["t_total_med"] for r in rs]
            ceff = [base["t_compute_med"] / r["t_compute_med"] for r in rs]
            a1.plot(Ps, [r["t_total_med"] * 1e3 for r in rs], "o-", label=lbl)
            a2.plot(Ps, eff, "o-", label=lbl + " (total)")
            a2.plot(Ps, ceff, "x--", label=lbl + " (compute)")
        else:
            S = [base["t_total_med"] / r["t_total_med"] for r in rs]
            Sc = [base["t_compute_med"] / r["t_compute_med"] for r in rs]
            eff = [s / p for s, p in zip(S, Ps)]
            ceff = [s / p for s, p in zip(Sc, Ps)]
            a1.plot(Ps, S, "o-", label=lbl + " (total)")
            if base["t_total_med"] > base["t_compute_med"] * 1.001:  # compute-only modes: same curve
                a1.plot(Ps, Sc, "x--", label=lbl + " (compute)")
            a2.plot(Ps, eff, "o-", label=lbl)
        for i, r in enumerate(rs):
            sp = base["t_total_med"] / r["t_total_med"]
            sp_s = "-" if weak else f"{sp:.2f}"
            kf_s = "-" if weak or r["P"] == 1 else f"{karp_flatt(sp, r['P']):.3f}"
            md.append(f"| {lbl} | {r['P']} | {r['t_total_med']*1e3:.2f} | {r['t_compute_med']*1e3:.2f} | "
                      f"{sp_s} | {eff[i]:.2f} | {ceff[i]:.2f} | {kf_s} | {r['load_imbalance']:.2f} |\n")
        # stacked phase breakdown per config
        fb, ab = plt.subplots(figsize=(6, 4))
        bottom = [0.0] * len(rs)
        for col, lab in PHASES:
            vals = [r[col] * 1e3 for r in rs]
            ab.bar([str(p) for p in Ps], vals, bottom=bottom, label=lab)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ab.set(xlabel="processes P", ylabel="time (ms)", title=f"{tag} scaling breakdown: {lbl}")
        ab.legend()
        fb.tight_layout()
        fname = f"{tag}_breakdown_{''.join(ch if ch.isalnum() or ch in 'x-' else '_' for ch in lbl).strip('_')}.png"
        fb.savefig(os.path.join(out, fname), dpi=150)
        plt.close(fb)
        md.append(f"\n![{fname}]({fname})\n\n")
    ideal = list(range(1, maxP + 1))
    if weak:
        a2.axhline(1.0, color="k", ls=":", label="ideal")
        a1.set(xlabel="processes P", ylabel="time (ms)", title="Weak scaling: time (ideal = flat)")
        a2.set(xlabel="processes P", ylabel="efficiency T(1)/T(P)", title="Weak scaling efficiency", ylim=(0, 1.2))
    else:
        a1.plot(ideal, ideal, "k:", label="ideal")
        a2.axhline(1.0, color="k", ls=":", label="ideal")
        a1.set(xlabel="processes P", ylabel="speedup T(1)/T(P)", title=f"{title} scaling speedup")
        a2.set(xlabel="processes P", ylabel="efficiency S(P)/P", title=f"{title} scaling efficiency", ylim=(0, 1.2))
    for a in (a1, a2):
        a.set_xscale("log", base=2)
        a.grid(True, alpha=.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{tag}_scaling.png"), dpi=150)
    md.append(f"![{tag}]({tag}_scaling.png)\n\n")


def plot_batch(rows, out, md):
    if not rows:
        return
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    md.append("## Batch throughput\n| config | batch | P | images/s | GFLOP/s | speedup | efficiency |\n"
              "|---|---|---|---|---|---|---|\n")
    for _, rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["P"])
        base = next((r for r in rs if r["P"] == 1), rs[0])
        lbl = f"{base['label']}: {name(base)}"
        ax.plot([r["P"] for r in rs], [r["images_per_s"] for r in rs], "o-", label=lbl)
        for r in rs:
            S = r["images_per_s"] / base["images_per_s"]
            md.append(f"| {lbl} | {r['batch']} | {r['P']} | {r['images_per_s']:.0f} | {r['gflops']:.2f} | "
                      f"{S:.2f} | {S / r['P'] * base['P']:.2f} |\n")
    ax.set(xscale="log", yscale="log", xlabel="processes P", ylabel="images / s", title="Batch throughput")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "batch_throughput.png"), dpi=150)
    md.append("\n![batch](batch_throughput.png)\n\n")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results"
    md = ["# Convolution benchmark summary\n\n"]
    plot_latency(load(os.path.join(d, "latency.csv")), d, md)
    plot_scaling(load(os.path.join(d, "strong.csv")), d, md)
    plot_scaling(load(os.path.join(d, "strong_dist.csv")), d, md, tag="strong_dist",
                 title="Strong (root vs distributed-input)")
    plot_scaling(load(os.path.join(d, "weak.csv")), d, md, weak=True)
    plot_batch(load(os.path.join(d, "batch.csv")), d, md)
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.writelines(md)
    print(f"wrote plots and {os.path.join(d, 'summary.md')}")


if __name__ == "__main__":
    main()
