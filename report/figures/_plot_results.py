"""Regenerate result figures from the top-level results_summary.csv.

Usage (from repo root):
    python report/figures/_plot_results.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(os.path.join(HERE, "..", "..", "results_summary.csv"))

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 10,
    "savefig.bbox": "tight",
})


def load():
    with open(CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        methods, rows = [], []
        for row in reader:
            methods.append(row[0])
            rows.append(row[1:])
    return header, methods, np.array(rows, dtype=float)


def out(name):
    return os.path.join(HERE, name)


def main():
    header, methods, arr = load()
    cols = header[1:]
    ds_labels = ["Jinan (mean)", "Hangzhou (mean)", "Xuancheng (full)"]
    ds_idx = [cols.index("JN_mean"), cols.index("HZ_mean"), cols.index("XC_mean")]
    colors = ["#4C72B0", "#DD8452", "#55A467"]

    # ---- bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(methods))
    bar_w = 0.25
    for i, (lbl, c, idx) in enumerate(zip(ds_labels, colors, ds_idx)):
        ax.bar(x + (i - 1) * bar_w, arr[:, idx], bar_w, label=lbl, color=c, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Average travel time (s)")
    ax.set_title("Average travel time per method, by dataset (lower is better)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out("results_bar.png"), dpi=200)
    plt.close(fig)

    # ---- per-method progression line plot
    order = ["FixedTime", "MaxPressure", "CoLight", "MPLight",
             "Adv-MPLight", "Adv-CoLight", "Ensemble (Adv-CoLight + Adv-MPLight)"]
    order_idx = [methods.index(m) for m in order]
    short_labels = ["FT", "MP", "CoLight", "MPLight", "Adv-MP\nLight", "Adv-Co\nLight", "Ensemble"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for lbl, c, idx, marker in zip(ds_labels, colors, ds_idx, ["o", "s", "^"]):
        ax.plot(range(len(order)), arr[order_idx, idx], "-" + marker, label=lbl, color=c, linewidth=1.6, markersize=7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("Average travel time (s)")
    ax.set_title("Per-method progression: each upgrade reduces travel time across all networks")
    ax.grid(linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out("results_progression.png"), dpi=200)
    plt.close(fig)

    # ---- per-step proportional improvement
    deltas = {
        "Adv-MPLight - MPLight": (arr[methods.index("MPLight")] - arr[methods.index("Adv-MPLight")])
            / arr[methods.index("MPLight")] * 100,
        "Adv-CoLight - CoLight": (arr[methods.index("CoLight")] - arr[methods.index("Adv-CoLight")])
            / arr[methods.index("CoLight")] * 100,
        "Ensemble - Adv-CoLight": (arr[methods.index("Adv-CoLight")]
                                   - arr[methods.index("Ensemble (Adv-CoLight + Adv-MPLight)")])
            / arr[methods.index("Adv-CoLight")] * 100,
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    xpos = np.arange(len(deltas))
    w = 0.25
    for i, (lbl, c, idx) in enumerate(zip(ds_labels, colors, ds_idx)):
        vals = [deltas[k][idx] for k in deltas]
        ax.bar(xpos + (i - 1) * w, vals, w, label=lbl, color=c, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(list(deltas.keys()), fontsize=9)
    ax.set_ylabel("Travel-time reduction (%)")
    ax.set_title("Per-step proportional improvement is approximately uniform across networks")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out("results_step_uplift.png"), dpi=200)
    plt.close(fig)

    # ---- relative improvement vs MPLight
    ref = arr[methods.index("MPLight"), :]
    rel = (ref - arr) / ref * 100.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    bar_colors = ["#bbbbbb", "#8a8a8a", "#5a82c2", "#cc8a64", "#3f7fbf", "#2c5e9c", "#15446e"]
    for ax, lbl, idx in zip(axes, ds_labels, ds_idx):
        ax.bar(np.arange(len(methods)), rel[:, idx], color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.set_title(lbl)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=24, ha="right", fontsize=8.5)
        ax.set_ylabel("Reduction vs. MPLight (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Travel-time reduction vs. MPLight baseline (higher is better)", y=1.02)
    fig.tight_layout()
    fig.savefig(out("results_rel_improve.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
