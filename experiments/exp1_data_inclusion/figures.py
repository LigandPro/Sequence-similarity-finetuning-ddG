"""Exp 1 figures — data-inclusion curves (Fig. 2a, 2b) and the three-regime plot (Fig. 2c).

Reads ``results/exp1_target_curves.tsv``, ``results/exp1_control_curves.tsv`` and
``results/exp1_regimes.tsv``, and nothing else.

    python experiments/exp1_data_inclusion/figures.py

Writes ``figures/exp11_target.svg``, ``figures/exp11_control.svg`` and ``figures/exp12.svg``,
and prints the summary numbers to compare with the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401 — registers the "science" style used below
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
EXPERIMENT = "exp1_data_inclusion"
FIGURES = REPO_ROOT / "figures"

# Fixes matplotlib's generated SVG element ids, so re-rendering gives a byte-identical file.
HASHSALT = "thermality"

# Seeds the bootstrap so the interval widths are reproducible run to run.
SEED = 42
N_BOOTSTRAP = 10_000

SIZES = [0, 200, 400, 800]
SIZE_COLORS = {0: "tab:blue", 200: "tab:green", 400: "tab:orange", 800: "tab:red"}

REGIMES = [
    ("pretrain_with_target_train", "Pretraining with target train", "tab:green"),
    ("target_train_only", "Target train from scratch", "tab:blue"),
    ("target_train_after_pretrain", "Target train from pretrained", "tab:orange"),
]


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval on the mean — the error bar of the three-regime figure."""
    rng = np.random.default_rng(SEED)
    draws = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(N_BOOTSTRAP)]
    alpha = 1 - confidence
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def read_table(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(REPO_ROOT)} not found; run "
            f"`python -m thermality.collect_results experiments/{EXPERIMENT}` first"
        )
    return pd.read_csv(path, sep="\t")


def plot_curves(table: pd.DataFrame, title: str, y_top: float, destination: Path) -> None:
    """Exp 1.1 — per-cluster mean RMSE against training step, one curve per pretraining size.

    The band is the per-step standard error across evaluation clusters, drawn where more than
    one cluster reported (``rmse_se`` is empty otherwise).
    """
    plt.figure()
    for size in SIZES:
        series = table[table["n_target_train"] == size].sort_values("step")
        if series.empty:
            continue
        label = f"{size} target-train sequences" + (" (baseline)" if size == 0 else "")
        color = SIZE_COLORS[size]
        plt.plot(series["step"], series["rmse_mean"], "-", color=color, label=label)
        banded = series[series["rmse_se"].notna()]
        if not banded.empty:
            plt.fill_between(
                banded["step"],
                banded["rmse_mean"] - banded["rmse_se"],
                banded["rmse_mean"] + banded["rmse_se"],
                color=color,
                alpha=0.1,
            )

    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Per cluster RMSE")
    plt.ylim(None, y_top)
    plt.minorticks_on()
    plt.grid(which="both", linestyle="--", alpha=0.7)
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=True, fancybox=True, edgecolor="grey", shadow=True)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.gcf().set_size_inches(10, 3)
    plt.tight_layout()
    plt.savefig(destination, format="svg", metadata={"Date": None})
    plt.close()


def plot_regimes(regimes: pd.DataFrame, destination: Path) -> dict[str, float]:
    """Exp 1.2 — final RMSE against target-train size, one series per training regime."""
    baseline_values = regimes[
        (regimes["regime"] == "pretrain_with_target_train") & (regimes["n_target_train"] == 0)
    ]["rmse"].to_numpy()
    baseline = float(np.mean(baseline_values))
    baseline_low, baseline_high = stats.t.interval(
        0.95, len(baseline_values) - 1, loc=baseline, scale=stats.sem(baseline_values)
    )

    plt.figure()
    minima: dict[str, float] = {}
    for regime, label, color in REGIMES:
        series = regimes[regimes["regime"] == regime]
        sizes = sorted(series["n_target_train"].unique())
        means, half_widths = [], []
        for size in sizes:
            values = series[series["n_target_train"] == size]["rmse"].to_numpy()
            means.append(float(np.mean(values)))
            low, high = bootstrap_ci(values)
            half_widths.append(abs(high - low) / 2)
        minima[regime] = min(means)

        means_array, half_array, size_array = np.array(means), np.array(half_widths), np.array(sizes)
        plt.errorbar(
            size_array, means_array, yerr=half_array, fmt="o-", label=label,
            color=color, capsize=3, elinewidth=1,
        )
        plt.fill_between(
            size_array, means_array - half_array, means_array + half_array, color=color, alpha=0.03
        )

    plt.hlines(baseline, 0, 1000, color="black", linestyle="-", alpha=0.7, label="Baseline")
    plt.hlines(baseline_high, 0, 1000, color="black", linestyle="--", alpha=0.5)
    plt.hlines(baseline_low, 0, 1000, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("N target-train sequences")
    plt.ylabel("Per cluster RMSE")
    plt.minorticks_on()
    plt.grid(which="both", linestyle="--", alpha=0.5)
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=True, fancybox=False, edgecolor="grey", shadow=True)
    plt.gcf().set_size_inches(10, 5)
    plt.tight_layout()
    plt.savefig(destination, format="svg", metadata={"Date": None})
    plt.close()

    minima["baseline"] = baseline
    return minima


def main() -> None:
    plt.style.use(["science", "no-latex"])
    plt.rcParams["svg.hashsalt"] = HASHSALT
    FIGURES.mkdir(exist_ok=True)

    target = read_table("exp1_target_curves.tsv")
    control = read_table("exp1_control_curves.tsv")
    regimes = read_table("exp1_regimes.tsv")

    plot_curves(target, "Target clusters", 1.8, FIGURES / "exp11_target.svg")
    plot_curves(control, "Control clusters", 1.2, FIGURES / "exp11_control.svg")
    numbers = plot_regimes(regimes, FIGURES / "exp12.svg")

    print("  final per-cluster RMSE (compare with Fig. 1a of the paper):")
    for key, value in numbers.items():
        print(f"    {key}: {value:.4f}")
    print(f"wrote exp11_target.svg, exp11_control.svg, exp12.svg to {FIGURES}")


if __name__ == "__main__":
    main()
