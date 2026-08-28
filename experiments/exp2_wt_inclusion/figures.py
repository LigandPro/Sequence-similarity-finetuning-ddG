"""Exp 2.1 figure — wild-type inclusion (Fig. 3, panel a).

Reads ``results/exp2_wt_inclusion.tsv`` and nothing else.

    python experiments/exp2_wt_inclusion/figures.py

Writes ``figures/exp21.svg`` and prints the endpoints to compare with the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
EXPERIMENT = "exp2_wt_inclusion"
FIGURES = REPO_ROOT / "figures"

# Fixes matplotlib's generated SVG element ids, so re-rendering gives a byte-identical file.
HASHSALT = "thermality"

ARMS = [
    ("matched", "WT-matched fine-tuning", "tab:blue"),
    ("mismatched", "WT-mismatched fine-tuning", "tab:orange"),
]


def read_table(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(REPO_ROOT)} not found; run "
            f"`python -m thermality.collect_results experiments/{EXPERIMENT}` first"
        )
    return pd.read_csv(path, sep="\t")


def main() -> None:
    plt.rcParams["svg.hashsalt"] = HASHSALT
    FIGURES.mkdir(exist_ok=True)
    table = read_table("exp2_wt_inclusion.tsv")

    # N=0 is the pretrained checkpoint evaluated before the first optimiser step. Every run
    # logs it and all ten agree, so both arms share it, and it is the denominator of the
    # paper's -53 % / -33 % claim.
    baseline = float(table[table["n_wt_train"] == 0]["rmse"].mean())

    plt.figure()
    endpoints: dict[str, float] = {}

    for arm, label, color in ARMS:
        series = table[table["arm"] == arm]
        sizes = sorted(series["n_wt_train"].unique())  # six points, not five: N=0 included
        means, errors = [], []
        for size in sizes:
            values = series[series["n_wt_train"] == size]["rmse"].to_numpy()
            means.append(float(np.mean(values)))
            # Standard error across evaluation clusters.
            errors.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)

        means_array, error_array, size_array = np.array(means), np.array(errors), np.array(sizes)
        plt.errorbar(
            size_array, means_array, yerr=error_array, fmt="o-", label=label,
            color=color, capsize=3, elinewidth=1,
        )
        plt.fill_between(
            size_array, means_array - error_array, means_array + error_array, color=color, alpha=0.1
        )
        endpoints[arm] = means[-1]

    plt.xlabel("N sequences from WT")
    plt.ylabel("Per WT RMSE mean")
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
    plt.gcf().set_size_inches(10, 4)
    plt.tight_layout()
    plt.savefig(FIGURES / "exp21.svg", format="svg", metadata={"Date": None})
    plt.close()

    # Compare with Fig. 3, panel a of the paper.
    print(f"  baseline {baseline:.4f} -> matched {endpoints['matched']:.4f} "
          f"({endpoints['matched'] / baseline - 1:+.1%}), "
          f"mismatched {endpoints['mismatched']:.4f} ({endpoints['mismatched'] / baseline - 1:+.1%})")
    print(f"wrote exp21.svg to {FIGURES}")


if __name__ == "__main__":
    main()
