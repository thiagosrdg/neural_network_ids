# %% [markdown]
# # A look at the SYNTHETIC classes (T02)
#
# Compares the six synthetic classes at difficulty 0 (easy) and difficulty 1 (hard).
# Synthetic data tests the code, not detection: these plots show the rules written in
# `neural_ids.synthetic`, not real traffic.
#
# Run: uv run python explore/synthetic_look.py  → 4 PNG files in reports/figures/
#
# Colors: blue = normal class, orange = attack class (two colors that stay distinct for
# colorblind readers). Each class gets its own panel, with all other flows in light gray.
# Panels use the TRUE generating class (label_noise=0), so the overlap at difficulty 1 comes
# from the packets, not from flipped labels.

# %%
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neural_ids.synthetic import ATTACK_CLASSES, NORMAL_CLASSES, make_dataset  # noqa: E402

FIGURES = Path("reports/figures")
CLASSES = NORMAL_CLASSES + ATTACK_CLASSES
COLOR = {"normal": "#2a78d6", "attack": "#eb6834"}
GRAY = "#c9c8c1"
N_FLOWS, SEED = 6000, 42


def group_color(name: str) -> str:
    return COLOR["attack" if name in ATTACK_CLASSES else "normal"]


# %% Two datasets with the same seed: only the difficulty changes.
data = {
    d: make_dataset(N_FLOWS, attack_fraction=0.5, difficulty=d, seed=SEED, label_noise=0.0)
    for d in (0.0, 1.0)
}
for df in data.values():
    df["total_packets"] = df["fwd_packets"] + df["bwd_packets"]


# %%
def small_multiples(x: str, y: str, xlabel: str, ylabel: str, filename: str) -> None:
    """Log-log scatter: rows = difficulty 0 and 1, columns = the six classes."""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5), sharex=True, sharey=True)
    for row, (d, df) in enumerate(data.items()):
        # Log axes cannot show 0, so zero values are moved to a small floor value.
        xs = np.maximum(df[x].to_numpy(), 1e-4)
        ys = np.maximum(df[y].to_numpy(), 1e-4)
        for col, name in enumerate(CLASSES):
            ax = axes[row, col]
            mine = (df["label"] == name).to_numpy()
            ax.scatter(xs[~mine], ys[~mine], s=4, c=GRAY, alpha=0.4, linewidths=0)
            ax.scatter(xs[mine], ys[mine], s=8, c=group_color(name), alpha=0.6, linewidths=0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, color="#e8e7e1", linewidth=0.6)
            ax.set_axisbelow(True)
            if row == 0:
                kind = "attack" if name in ATTACK_CLASSES else "normal"
                ax.set_title(f"{name} ({kind})", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"difficulty {d:g}\n{ylabel}")
            if row == 1:
                ax.set_xlabel(xlabel)
    fig.suptitle(
        f"SYNTHETIC flows, true class — {ylabel} vs {xlabel} (gray: other classes; 0 drawn at 1e-4)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=110)
    plt.close(fig)


def boxes_by_class(column: str, ylabel: str, filename: str, log: bool) -> None:
    """One box per class; left panel difficulty 0, right panel difficulty 1."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (d, df) in zip(axes, data.items(), strict=True):
        values = [df.loc[df["label"] == name, column].to_numpy() for name in CLASSES]
        if log:
            values = [np.maximum(v, 1e-2) for v in values]
        parts = ax.boxplot(values, patch_artist=True, widths=0.6, showfliers=False)
        for box, name in zip(parts["boxes"], CLASSES, strict=True):
            box.set(facecolor=group_color(name), alpha=0.75, edgecolor="#555550")
        for median in parts["medians"]:
            median.set(color="#1a1a19", linewidth=1.5)
        ax.set_xticks(range(1, len(CLASSES) + 1), CLASSES, rotation=20)
        ax.set_title(f"difficulty {d:g}")
        ax.grid(True, axis="y", color="#e8e7e1", linewidth=0.6)
        ax.set_axisbelow(True)
        if log:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"SYNTHETIC flows, true class — {ylabel} (blue: normal, orange: attack)")
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=110)
    plt.close(fig)


# %%
FIGURES.mkdir(parents=True, exist_ok=True)
small_multiples(
    "duration_s", "total_packets", "duration (s)", "packets", "synthetic_packets_vs_duration.png"
)
small_multiples(
    "iat_mean_s", "iat_std_s", "IAT mean (s)", "IAT std (s)", "synthetic_iat_mean_vs_std.png"
)
boxes_by_class("pkt_len_mean", "mean IP packet length (bytes)", "synthetic_pkt_len_mean.png", False)
boxes_by_class(
    "packets_per_s", "packets per second (0 drawn at 0.01)", "synthetic_packets_per_s.png", log=True
)

# %% Numbers behind the plots: medians per class and difficulty.
cols = ["total_packets", "duration_s", "iat_std_s", "pkt_len_mean", "packets_per_s"]
summary = pd.concat({f"d={d:g}": df.groupby("label")[cols].median() for d, df in data.items()})
print("SYNTHETIC data, medians per class:")
print(summary.round(3).to_string())
print(f"saved 4 figures to {FIGURES}/")
