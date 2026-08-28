"""Exp 2.3 figure — discrepancy against ESM embedding distance (Fig. 3, panel c).

Reads ``results/exp2_d_esm.tsv`` and nothing else.

    python experiments/exp2_d_esm/figures.py

Writes ``figures/exp23.svg`` and prints the Spearman correlation to compare with the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
EXPERIMENT = "exp2_d_esm"
FIGURES = REPO_ROOT / "figures"

# Fixes matplotlib's generated SVG element ids, so re-rendering gives a byte-identical file.
HASHSALT = "thermality"

ARMS = [
    ("within", "Within-cluster", "tab:blue"),
    ("across", "Across-cluster", "tab:orange"),
]


def read_table(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(REPO_ROOT)} not found; run "
            f"`python -m thermality.collect_results experiments/{EXPERIMENT}` first"
        )
    return pd.read_csv(path, sep="\t")


def bin_bands(table: pd.DataFrame) -> pd.DataFrame:
    """Per-bin 2.5th–97.5th percentile of the raw pooled values, both arms combined.

    A spread of points, not a confidence interval on the mean. Bin occupancy is uneven: the
    within-cluster arm has a single-pair ``9.0_10.0`` bin and no ``8.0_9.0``, the across-cluster
    arm has ``8.0_9.0`` and no ``0.0_0.5``.
    """
    rows = []
    for label, group in table.groupby("bin"):
        lower, upper = (float(part) for part in str(label).split("_"))
        low, high = np.percentile(group["discrepancy"].to_numpy(), [2.5, 97.5])
        rows.append({"lower": lower, "upper": upper, "low": float(low), "high": float(high)})
    return pd.DataFrame(rows).sort_values("lower")


def main() -> None:
    plt.rcParams["svg.hashsalt"] = HASHSALT
    FIGURES.mkdir(exist_ok=True)
    table = read_table("exp2_d_esm.tsv")

    plt.figure(figsize=(10, 5))
    for arm, label, color in ARMS:
        series = table[table["arm"] == arm]
        plt.scatter(series["d_esm"], series["discrepancy"], color=color, alpha=0.7, s=24, label=label)

    bands = bin_bands(table)
    for index, (lower, upper, low, high) in enumerate(bands.itertuples(index=False)):
        plt.fill_between(
            [lower, upper], [low, low], [high, high],
            color="tab:green", alpha=0.5 / 7,
            label="Binned 95 % spread, both arms combined" if index == 0 else None,
        )

    plt.xlabel("$d_{ESM}$ distance")
    plt.ylabel("Discrepancy")
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    plt.grid(which="both", linestyle="--", alpha=0.5)
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=True, fancybox=True, edgecolor="grey", shadow=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES / "exp23.svg", format="svg", metadata={"Date": None})
    plt.close()

    # Spearman correlation, as in the paper (Fig. 3, panel c).
    rho, p_value = spearmanr(table["d_esm"], table["discrepancy"])
    print(f"  Spearman rho {rho:.4f}, p = {p_value:.3g}, n = {len(table)}")
    print(f"wrote exp23.svg to {FIGURES}")


if __name__ == "__main__":
    main()
