"""Aggregate one experiment's training runs into the tables its figure script reads.

    python -m thermality.collect_results experiments/<name>

``thermality.train`` and ``thermality.finetune`` write one TSV per run under
``results/runs/<name>/``, with a row per (step, split, cluster).  This module reads those
files, together with each run's config, and writes ``results/<name>*.tsv``:

    exp1_data_inclusion   exp1_target_curves.tsv, exp1_control_curves.tsv, exp1_regimes.tsv
    exp2_wt_inclusion     exp2_wt_inclusion.tsv
    exp2_bitscore_norm    exp2_bitscore_norm.tsv
    exp2_d_esm            exp2_d_esm.tsv

A config without a run file is reported and skipped.  Nothing here compares against any
number; the figure scripts print their summary statistics for comparison with the paper.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from thermality.config import PROJECT_ROOT, run_identity

RUNS = PROJECT_ROOT / "results" / "runs"
RESULTS = PROJECT_ROOT / "results"
RAW = PROJECT_ROOT / "data" / "sources"

# The three fine-tuning regimes of the Exp 1 regimes figure, keyed by config directory.
REGIME_PRETRAIN = "pretrain_with_target_train"
REGIME_AFTER_PRETRAIN = "target_train_after_pretrain"
REGIME_ONLY = "target_train_only"


# --------------------------------------------------------------------------- run files


class Run:
    """One run's metrics TSV plus the config that produced it."""

    def __init__(self, config_path: Path, metrics_path: Path):
        self.config_path = config_path
        self.stem = run_identity(config_path)[2]
        self.cfg = yaml.safe_load(config_path.read_text()) or {}
        self.table = pd.read_csv(metrics_path, sep="\t")

    def rows(self, split: str) -> pd.DataFrame:
        return self.table[self.table["split"] == split]

    def last_step(self, split: str) -> pd.DataFrame:
        """One row per cluster at the last evaluation step of ``split``."""
        rows = self.rows(split)
        if rows.empty:
            return rows
        return rows[rows["step"] == rows["step"].max()]

    def rmse(self, split: str, cluster: str, at: str = "last") -> float | None:
        rows = self.rows(split)
        rows = rows[rows["cluster"] == cluster].sort_values("step")
        if rows.empty:
            return None
        return float(rows["rmse"].iloc[-1 if at == "last" else 0])


def load_runs(experiment_dir: Path) -> list[Run]:
    runs_dir = RUNS / experiment_dir.name
    if not runs_dir.is_dir():
        raise SystemExit(
            f"no run metrics under {runs_dir.relative_to(PROJECT_ROOT)}; "
            f"run `python -m thermality.run_experiment {experiment_dir.relative_to(PROJECT_ROOT)}` first"
        )
    runs = []
    for config_path in sorted((experiment_dir / "configs").rglob("*.yaml")):
        _, run_name, _ = run_identity(config_path)
        metrics_path = runs_dir / f"{run_name}.tsv"
        if not metrics_path.exists():
            print(f"  no metrics for {config_path.relative_to(PROJECT_ROOT)} — skipped")
            continue
        runs.append(Run(config_path, metrics_path))
    return runs


def is_finetune(run: Run) -> bool:
    return "FLEET" in run.cfg


def train_size(run: Run) -> int:
    """The number of training records, from the last path component of the training set."""
    if is_finetune(run):
        return int(Path(run.cfg["FLEET"]["TRAIN_DIR"]).name)
    dataset = Path(run.cfg["TRAIN"]["DATASET_PATH"])
    return 0 if dataset.name == "pretrain.tsv" else int(dataset.parent.name)


# --------------------------------------------------------------------------- shared helpers


def parse_pair(cluster: str) -> tuple[str, str] | None:
    if "_" not in cluster:
        return None
    wt1, wt2 = cluster.split("_", 1)
    return wt1, wt2


def direction_of(run: Run) -> int:
    """1 when the run trains on WT1 (``wt1_train``), 2 when on WT2."""
    return 1 if "wt1_train" in run.stem.name else 2


def discrepancies(cluster: str, wt1: Run, wt2: Run) -> tuple[float, float] | None:
    """The paper's two discrepancies for one WT pair: D(WT1|WT2) and D(WT2|WT1).

    A ``wt1_train`` run trains on WT1, so its ``val`` split is the matched test set and its
    ``test`` split is the mismatched one; ``wt2_train`` is the mirror image.
    """
    mx1_same = wt1.rmse("val", cluster)
    mx1_diff = wt1.rmse("test", cluster)
    mx2_same = wt2.rmse("test", cluster)
    mx2_diff = wt2.rmse("val", cluster)
    if None in (mx1_same, mx1_diff, mx2_same, mx2_diff):
        return None
    return mx1_same - mx2_diff, mx2_same - mx1_diff


def pair_up(runs: list[Run]) -> dict[str, dict[int, Run]]:
    """``{bin: {direction: run}}`` for the ``<lo>_<hi>_wt{1,2}_train`` configs, complete bins only."""
    paired: dict[str, dict[int, Run]] = {}
    for run in runs:
        if not is_finetune(run):
            continue
        label = run.stem.name.rsplit("_wt", 1)[0]
        paired.setdefault(label, {})[direction_of(run)] = run
    return {label: arms for label, arms in paired.items() if len(arms) == 2}


def write_tsv(name: str, columns: list[str], rows: list[dict]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.relative_to(PROJECT_ROOT)}: {len(rows)} rows")
    return path


def report_spearman(rows: list[dict], x: str) -> None:
    if len(rows) < 3:
        return
    rho, p_value = spearmanr([r[x] for r in rows], [r["discrepancy"] for r in rows])
    print(f"  Spearman rho {rho:.4f}, p = {p_value:.3g}, n = {len(rows)}")


# --------------------------------------------------------------------------- lookups


def bitscore_norm_lookup() -> dict[tuple[str, str], float]:
    """``(wt1, wt2) -> bitscore / mean(self-bitscores)``, symmetric."""
    df = pd.read_csv(RAW / "ALL_SEQS_blastp_named.tsv", sep="\t")
    df["qseqid"] = df["qseqid"].str.split("_").str[0]
    df["sseqid"] = df["sseqid"].str.split("_").str[0]

    self_hits = df[df["qseqid"] == df["sseqid"]].drop_duplicates(subset="qseqid")
    self_map = self_hits.set_index("qseqid")["bitscore"]
    denominator = 0.5 * (df["qseqid"].map(self_map) + df["sseqid"].map(self_map))
    df["bitscore_norm"] = df["bitscore"] / denominator.replace(0, np.nan)

    df = df.drop_duplicates(subset=["qseqid", "sseqid"])
    lookup: dict[tuple[str, str], float] = {}
    for q, s, score in zip(df["qseqid"], df["sseqid"], df["bitscore_norm"]):
        lookup[(q, s)] = score
        lookup[(s, q)] = score
    return lookup


def wt_embeddings() -> dict[str, np.ndarray]:
    df = pd.read_csv(RAW / "wt_embeddings.tsv", sep="\t", usecols=["wt_uid", "avg_wt_emb"])
    df["avg_wt_emb"] = df["avg_wt_emb"].map(lambda s: np.fromstring(s.strip()[1:-1], sep=","))
    return df.set_index("wt_uid")["avg_wt_emb"].to_dict()


def d_esm(wt1: str, wt2: str, embeddings: dict[str, np.ndarray]) -> float | None:
    if wt1 not in embeddings or wt2 not in embeddings:
        return None
    return float(np.sum((embeddings[wt1] - embeddings[wt2]) ** 2))


# --------------------------------------------------------------------------- collectors


def collect_exp1(runs: list[Run]) -> None:
    """Exp 1 — training curves (target and control) and the three-regime endpoints."""
    pretrains = [run for run in runs if not is_finetune(run)]

    for split, out in (("target", "exp1_target_curves.tsv"), ("control", "exp1_control_curves.tsv")):
        rows = []
        for run in pretrains:
            size = train_size(run)
            for step, group in run.rows(split).groupby("step"):
                values = group["rmse"].to_numpy()
                rows.append(
                    {
                        "n_target_train": size,
                        "step": int(step),
                        "rmse_mean": round(float(values.mean()), 6),
                        "rmse_se": round(float(values.std(ddof=1) / np.sqrt(len(values))), 6) if len(values) > 1 else "",
                    }
                )
        rows.sort(key=lambda r: (r["n_target_train"], r["step"]))
        write_tsv(out, ["n_target_train", "step", "rmse_mean", "rmse_se"], rows)

    rows = []
    for run in pretrains:
        size = train_size(run)
        for _, row in run.last_step("target").iterrows():
            record = {
                "regime": REGIME_PRETRAIN,
                "n_target_train": size,
                "cluster": row["cluster"],
                "step": int(row["step"]),
                "rmse": round(float(row["rmse"]), 9),
            }
            rows.append(record)
            # The pretrained model is the N=0 point of the fine-tune-after-pretrain regime too;
            # fine-tuning from scratch has no pretraining to fall back on and starts at its first N.
            if size == 0:
                rows.append({**record, "regime": REGIME_AFTER_PRETRAIN})

    for run in runs:
        if not is_finetune(run):
            continue
        regime = run.stem.parent.name
        if regime not in (REGIME_AFTER_PRETRAIN, REGIME_ONLY):
            print(f"  {run.config_path.relative_to(PROJECT_ROOT)}: not under a regime directory — skipped")
            continue
        for _, row in run.last_step("test").iterrows():
            rows.append(
                {
                    "regime": regime,
                    "n_target_train": train_size(run),
                    "cluster": row["cluster"],
                    "step": int(row["step"]),
                    "rmse": round(float(row["rmse"]), 9),
                }
            )

    rows.sort(key=lambda r: (r["regime"], r["n_target_train"], str(r["cluster"])))
    write_tsv("exp1_regimes.tsv", ["regime", "n_target_train", "cluster", "step", "rmse"], rows)


def collect_exp2_wt_inclusion(runs: list[Run]) -> None:
    """Exp 2.1 — per-cluster RMSE against matched and mismatched WT test sets."""
    rows: list[dict] = []
    # Every run evaluates the shared pretrained checkpoint before its first optimiser step,
    # so the N=0 point is one value per (arm, direction, cluster) rather than one per run.
    baseline: dict[tuple[str, int, str], tuple[float, Run]] = {}

    for run in runs:
        if not is_finetune(run):
            continue
        direction = direction_of(run)
        size = train_size(run)
        matched_split, mismatched_split = ("val", "test") if direction == 1 else ("test", "val")
        for arm, split in (("matched", matched_split), ("mismatched", mismatched_split)):
            for cluster in sorted(run.rows(split)["cluster"].unique()):
                first, last = run.rmse(split, cluster, at="first"), run.rmse(split, cluster, at="last")
                key = (arm, direction, cluster)
                if key in baseline and round(baseline[key][0], 9) != round(first, 9):
                    print(
                        f"  warning: N=0 value for {key} differs between "
                        f"{baseline[key][1].stem} and {run.stem} — did both start from the same checkpoint?"
                    )
                baseline.setdefault(key, (first, run))
                rows.append(
                    {"n_wt_train": size, "arm": arm, "direction": direction, "cluster": cluster, "rmse": round(last, 9)}
                )

    rows.extend(
        {"n_wt_train": 0, "arm": arm, "direction": direction, "cluster": cluster, "rmse": round(value, 9)}
        for (arm, direction, cluster), (value, _) in baseline.items()
    )
    rows.sort(key=lambda r: (r["n_wt_train"], r["arm"], r["direction"], r["cluster"]))
    write_tsv("exp2_wt_inclusion.tsv", ["n_wt_train", "arm", "direction", "cluster", "rmse"], rows)


def collect_exp2_bitscore(runs: list[Run]) -> None:
    """Exp 2.2 — discrepancy against normalised blastp bitscore, one point per direction."""
    lookup = bitscore_norm_lookup()
    rows: list[dict] = []
    for label, arms in sorted(pair_up(runs).items()):
        for cluster in sorted(arms[1].table["cluster"].unique()):
            pair = parse_pair(cluster)
            if pair is None or pair not in lookup:
                print(f"  {label}/{cluster}: no bitscore for this pair — skipped")
                continue
            values = discrepancies(cluster, arms[1], arms[2])
            if values is None:
                print(f"  {label}/{cluster}: metrics missing in one direction — skipped")
                continue
            for direction, value in enumerate(values, start=1):
                rows.append(
                    {
                        "pair": cluster,
                        "wt1": pair[0],
                        "wt2": pair[1],
                        "bin": label,
                        "direction": direction,
                        "bitscore_norm": round(float(lookup[pair]), 9),
                        "discrepancy": round(value, 9),
                    }
                )
    rows.sort(key=lambda r: (r["pair"], r["direction"]))
    write_tsv(
        "exp2_bitscore_norm.tsv",
        ["pair", "wt1", "wt2", "bin", "direction", "bitscore_norm", "discrepancy"],
        rows,
    )
    report_spearman(rows, "bitscore_norm")


def collect_exp2_d_esm(runs: list[Run]) -> None:
    """Exp 2.3 — discrepancy against ESM embedding distance, within and across clusters."""
    embeddings = wt_embeddings()
    rows: list[dict] = []
    for arm in ("within", "across"):
        arm_runs = [run for run in runs if run.stem.parts[0] == arm]
        for label, directions in sorted(pair_up(arm_runs).items()):
            for cluster in sorted(directions[1].table["cluster"].unique()):
                pair = parse_pair(cluster)
                if pair is None:
                    continue
                distance = d_esm(pair[0], pair[1], embeddings)
                values = discrepancies(cluster, directions[1], directions[2])
                if distance is None or values is None:
                    print(f"  {arm}/{label}/{cluster}: embedding or metrics missing — skipped")
                    continue
                for direction, value in enumerate(values, start=1):
                    rows.append(
                        {
                            "pair": cluster,
                            "wt1": pair[0],
                            "wt2": pair[1],
                            "arm": arm,
                            "bin": label,
                            "direction": direction,
                            "d_esm": round(distance, 9),
                            "discrepancy": round(value, 9),
                        }
                    )
    rows.sort(key=lambda r: (r["arm"], r["pair"], r["direction"]))
    write_tsv(
        "exp2_d_esm.tsv",
        ["pair", "wt1", "wt2", "arm", "bin", "direction", "d_esm", "discrepancy"],
        rows,
    )
    report_spearman(rows, "d_esm")


COLLECTORS = {
    "exp1_data_inclusion": collect_exp1,
    "exp2_wt_inclusion": collect_exp2_wt_inclusion,
    "exp2_bitscore_norm": collect_exp2_bitscore,
    "exp2_d_esm": collect_exp2_d_esm,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path, help="an experiments/<name>/ directory")
    args = parser.parse_args()

    experiment_dir = args.experiment.resolve()
    collector = COLLECTORS.get(experiment_dir.name)
    if collector is None:
        print(f"no collector for {experiment_dir.name}; known: {', '.join(COLLECTORS)}", file=sys.stderr)
        return 1

    print(f"{experiment_dir.name}:")
    runs = load_runs(experiment_dir)
    if not runs:
        print("  no runs found", file=sys.stderr)
        return 1
    collector(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
