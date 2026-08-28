"""Exp 2.2 figure — discrepancy against sequence similarity (Fig. 3, panel b).

Reads ``results/exp2_bitscore_norm.tsv`` and nothing else.

    python experiments/exp2_bitscore_norm/figures.py

Writes ``figures/exp22.svg`` and prints the Spearman correlation to compare with the paper.
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
EXPERIMENT = "exp2_bitscore_norm"
FIGURES = REPO_ROOT / "figures"

# Fixes matplotlib's generated SVG element ids, so re-rendering gives a byte-identical file.
HASHSALT = "thermality"


def read_table(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(REPO_ROOT)} not found; run "
            f"`python -m thermality.collect_results experiments/{EXPERIMENT}` first"
        )
    return pd.read_csv(path, sep="\t")


def bin_bands(table: pd.DataFrame) -> pd.DataFrame:
    """Per-bin 2.5th–97.5th percentile of the raw pooled per-pair values.

    This is a spread of points, not a confidence interval on the mean, and the thinnest bin
    holds as few as two values (``0.099_0.2``). The figure legend says "spread" for that reason.
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
    table = read_table("exp2_bitscore_norm.tsv")
    scores = table["bitscore_norm"].to_numpy()
    discrepancies = table["discrepancy"].to_numpy()

    plt.figure(figsize=(10, 5))
    plt.scatter(scores, discrepancies, color="tab:blue", alpha=0.7, s=24, label="WT pairs")

    bands = bin_bands(table)
    for index, (lower, upper, low, high) in enumerate(bands.itertuples(index=False)):
        plt.fill_between(
            [lower, upper], [low, low], [high, high],
            color="tab:blue", alpha=0.5 / 7,
            label="Binned 95 % spread of pairs" if index == 0 else None,
        )

    plt.xlabel("$bitscore_{norm}$ distance")
    plt.ylabel("Discrepancy")
    plt.gca().invert_xaxis()  # similarity 1.0 on the left
    ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    plt.grid(which="both", linestyle="--", alpha=0.5)
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=True, fancybox=True, edgecolor="grey", shadow=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_tick_params(which="both", top=False)
    ax.yaxis.set_tick_params(which="both", right=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "exp22.svg", format="svg", metadata={"Date": None})
    plt.close()

    # Spearman correlation, as in the paper (Fig. 3, panel b).
    rho, p_value = spearmanr(scores, discrepancies)
    print(f"  Spearman rho {rho:.4f}, p = {p_value:.3g}, n = {len(table)}")
    print(f"wrote exp22.svg to {FIGURES}")


if __name__ == "__main__":
    main()
