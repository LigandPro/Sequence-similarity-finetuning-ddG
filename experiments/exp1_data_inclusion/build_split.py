"""Rebuild the Exp 1 data-inclusion split (``data/splits/exp1_data_inclusion``).

One seeded global generator is consumed in sequence, so the order of the sampling calls below
is load-bearing: moving one changes every file after it.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "sources"
DEFAULT_OUT = REPO_ROOT / "data" / "splits"
ROOT_NAME = "exp1_data_inclusion"

RANDOM_SEED = 420

SMALL_THRESHOLD = 100
MEDIUM_THRESHOLD = 1_500
BIG_THRESHOLD = 4_000
DOMINANT_THRESHOLD = 0.7
MAX_TARGET_TRAIN_PER_CLUSTER = 1000
MAX_TEST_PER_CLUSTER = 500

TARGET_TRAIN_POPULATIONS = [10, 20, 50, 100, 200, 300, 400, 600, 800, 1000]

SOURCES = ["mega_smdi", "S2648", "fireprot", "Ssym", "tmdbs"]
CLUSTER_FILE = "mega_S2648_S669_Ssym_p53_M_ptmuld_ptmulnr_fireprot_tmdbs_tmdbm_cluster.tsv"
COLUMNS_TO_KEEP = ["wt_uid", "ddG", "mutant_seq", "wt_seq"]


def load_dataset() -> pd.DataFrame:
    """Concatenate the sources, keeping single substitutions, and attach the cluster map."""
    filtered_sources = []
    for source in SOURCES:
        frame = pd.read_csv(RAW / f"{source}.tsv", sep="\t", low_memory=False)
        if "mutation_type" in frame.columns:
            filtered = frame[frame["mutation_type"] == "single"]
        else:
            filtered = frame[
                frame["wt_seq"].astype(str).str.len() == frame["mutant_seq"].astype(str).str.len()
            ]
            filtered = filtered[
                filtered["wt_seq"]
                .astype(str)
                .combine(
                    filtered["mutant_seq"].astype(str),
                    lambda a, b: sum(x != y for x, y in zip(a, b)),
                )
                == 1
            ]
        filtered_sources.append(filtered[COLUMNS_TO_KEEP])

    dataset = pd.concat(filtered_sources)
    dataset["mutation_type"] = "single"
    dataset = dataset.drop_duplicates(subset=["mutant_seq", "wt_uid"])

    clusters = pd.read_csv(RAW / CLUSTER_FILE, sep="\t")
    return dataset.merge(clusters[["wt_uid", "cluster"]], on="wt_uid", how="left")


def annotate(dataset: pd.DataFrame) -> pd.DataFrame:
    """Flag clusters one wt_seq dominates, and bin clusters by size."""
    wtseq_counts = dataset.groupby(["cluster", "wt_seq"]).size().reset_index(name="count")
    cluster_sizes = dataset.groupby("cluster").size().reset_index(name="cluster_size")
    wtseq_counts = wtseq_counts.merge(cluster_sizes, on="cluster")
    wtseq_counts["local_threshold"] = (wtseq_counts["cluster_size"] * DOMINANT_THRESHOLD).astype(int)
    dominated = wtseq_counts[wtseq_counts["count"] > wtseq_counts["local_threshold"]]["cluster"].unique()
    dataset["domination"] = dataset["cluster"].isin(dominated)

    dataset["cluster_size"] = dataset["cluster"].map(dataset["cluster"].value_counts())
    dataset["size_bin"] = pd.cut(
        dataset["cluster_size"],
        bins=[-float("inf"), SMALL_THRESHOLD, MEDIUM_THRESHOLD, BIG_THRESHOLD, float("inf")],
        labels=["small", "medium", "big", "large"],
    )
    return dataset


def pack_clusters(dataset: pd.DataFrame, cluster_names: list) -> tuple[dict, dict, dict]:
    """Split the big clusters into target, control and target-training sets.

    A cluster is target-eligible when one of its wt_seqs has enough mutations to fill the
    largest target-training set and no single wt_seq dominates it; the rest become control
    candidates, then are randomly thinned so the two groups have the same number of clusters.
    """
    target: dict = {}
    target_train: dict = {}
    control: dict = {}

    for cluster_name in cluster_names:
        domination = dataset.loc[dataset["cluster"] == cluster_name, "domination"].iloc[0]
        cluster_df = dataset.groupby("cluster").get_group(cluster_name)
        wt_seq_counts = cluster_df["wt_seq"].value_counts()

        if (wt_seq_counts >= MAX_TARGET_TRAIN_PER_CLUSTER).any() and not domination:
            suitable = wt_seq_counts[wt_seq_counts > MAX_TARGET_TRAIN_PER_CLUSTER].index.tolist()
            target_train_wt_seq = np.random.choice(suitable)
            target_train_seqs = dataset[dataset["wt_seq"].isin([target_train_wt_seq])].copy()
            target_train_seqs = target_train_seqs.loc[
                np.random.choice(list(target_train_seqs.index), size=MAX_TARGET_TRAIN_PER_CLUSTER, replace=False)
            ]
            target_train[cluster_name] = target_train_seqs

            others = [wt_seq for wt_seq in wt_seq_counts.index if wt_seq != target_train_wt_seq]
            target_seqs = dataset[dataset["wt_seq"].isin(others)].copy()
            if len(target_seqs) < MAX_TEST_PER_CLUSTER:
                raise ValueError(f"Cluster {cluster_name} has fewer than {MAX_TEST_PER_CLUSTER} target rows")
            target[cluster_name] = target_seqs.loc[
                np.random.choice(list(target_seqs.index), size=MAX_TEST_PER_CLUSTER, replace=False)
            ]
        else:
            unsuitable = dataset[dataset["wt_seq"].isin(wt_seq_counts.index.tolist())].copy()
            if len(unsuitable) < MAX_TEST_PER_CLUSTER:
                raise ValueError(f"Cluster {cluster_name} has fewer than {MAX_TEST_PER_CLUSTER} control rows")
            control[cluster_name] = unsuitable.loc[
                np.random.choice(list(unsuitable.index), size=MAX_TEST_PER_CLUSTER, replace=False)
            ]

    if len(control) < len(target):
        raise ValueError("more target clusters than control ones; balancing is not implemented")
    keep = np.random.choice(list(control.keys()), len(target), replace=False)
    control = {k: v for k, v in control.items() if k in keep}

    return control, target, target_train


def build(out_root: Path) -> None:
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    dataset = annotate(load_dataset())
    train_dataset = dataset
    big = dataset[dataset["size_bin"] == "big"]

    control, target, target_train = pack_clusters(big, list(big.groupby("cluster").groups.keys()))
    print(f"clusters: {len(target)} target, {len(control)} control, {len(target_train)} target_train")

    pretrain = train_dataset[
        ~train_dataset["cluster"].isin(list(target.keys()) + list(control.keys()))
    ]
    heldout_val = pretrain.sample(n=MAX_TEST_PER_CLUSTER, random_state=RANDOM_SEED, replace=False)
    pretrain = pretrain[~pretrain["mutant_seq"].isin(heldout_val["mutant_seq"])]
    pretrain_mem = pretrain.sample(n=MAX_TEST_PER_CLUSTER, random_state=RANDOM_SEED, replace=False)
    print(f"pretrain rows: {len(pretrain)}")

    output_dir = out_root / ROOT_NAME
    test_path = output_dir / "test"
    target_path = test_path / "target_per_cluster"
    control_path = test_path / "control_per_cluster"
    for path in (output_dir, test_path, target_path, control_path):
        path.mkdir(parents=True, exist_ok=True)

    pretrain.to_csv(output_dir / "pretrain.tsv", sep="\t", index=False)
    pd.concat(list(control.values())).to_csv(test_path / "all_control.tsv", sep="\t", index=False)
    pd.concat(list(target.values())).to_csv(test_path / "all_target.tsv", sep="\t", index=False)
    heldout_val.to_csv(test_path / "heldout_val.tsv", sep="\t", index=False)
    pretrain_mem.to_csv(test_path / "pretrain_mem.tsv", sep="\t", index=False)

    for cluster, rows in target.items():
        rows.to_csv(target_path / f"{cluster}.tsv", sep="\t", index=False)
    for cluster, rows in control.items():
        rows.to_csv(control_path / f"{cluster}.tsv", sep="\t", index=False)

    for n in TARGET_TRAIN_POPULATIONS:
        amount_path = output_dir / "target_train" / str(n)
        amount_path.mkdir(parents=True, exist_ok=True)
        sampled_this_population = pd.DataFrame()
        for cluster, rows in target_train.items():
            if len(rows) < n:
                raise ValueError(f"cluster {cluster} has fewer than {n} target_train rows")
            sampled = rows.sample(n=n, random_state=42, replace=False)
            sampled.to_csv(amount_path / f"{cluster}.tsv", sep="\t", index=False)
            sampled_this_population = pd.concat([sampled_this_population, sampled])
        sampled_this_population.to_csv(amount_path / "all_target_train.tsv", sep="\t", index=False)
        pd.concat([pretrain, sampled_this_population]).to_csv(
            amount_path / "pretrain_with_all_target_train.tsv", sep="\t", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory to write the split root into")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
