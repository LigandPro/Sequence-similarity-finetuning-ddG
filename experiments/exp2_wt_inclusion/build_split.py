"""Rebuild the Exp 2.1 WT-inclusion split (``data/splits/exp2_wt_inclusion``).

One seeded global generator is consumed in sequence, so the order of the sampling calls below
is load-bearing: moving one changes every file after it.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_OUT = REPO_ROOT / "data" / "splits"
ROOT_NAME = "exp2_wt_inclusion"

RANDOM_SEED = 42

SUFFICIENT_POPULATION = 1000
TOO_IMPORTANT_CLUSTER = 10000
TRAIN_SAMPLE_SIZE = 800
TEST_SAMPLE_SIZE = 200
TRAIN_POPULATIONS = [10, 20, 50, 100, 200, 400, 800]

TEST_SOURCES = ["mega_smdi"]


def read_processed(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        sys.exit("run python data/preprocess.py first")
    return pd.read_csv(path, sep="\t", low_memory=False)


def valid_clusters(test_dataset: pd.DataFrame, cluster_names: list) -> dict:
    """Clusters holding two wild types of equal length, each with enough mutations to fine-tune on."""
    valid: dict = {}
    for cluster in cluster_names:
        cluster_slice = test_dataset[test_dataset["cluster"] == cluster]
        if len(cluster_slice) < SUFFICIENT_POPULATION * 2 or len(cluster_slice) > TOO_IMPORTANT_CLUSTER:
            continue
        all_wt = cluster_slice.wt_uid.unique()
        if len(all_wt) < 2:
            continue
        sufficient = {}
        for wt in all_wt:
            wt_slice = cluster_slice[cluster_slice["wt_uid"] == wt]
            if len(wt_slice) < SUFFICIENT_POPULATION:
                continue
            sufficient[wt] = len(wt_slice.wt_seq.iloc[0])
        if len(sufficient) >= 2:
            valid[cluster] = sufficient

    # Equal-length wild types only: pick one length per cluster, then at most two wild types.
    for cluster in list(valid):
        length_counts: dict[int, int] = {}
        for length in valid[cluster].values():
            length_counts[length] = length_counts.get(length, 0) + 1
        length_counts = {length: n for length, n in length_counts.items() if n >= 2}
        if not length_counts:
            del valid[cluster]
            continue
        chosen = random.choice(list(length_counts.keys()))
        valid[cluster] = {wt: length for wt, length in valid[cluster].items() if length == chosen}

    for cluster in valid:
        if len(valid[cluster]) > 2:
            chosen_wts = random.sample(list(valid[cluster].keys()), 2)
            valid[cluster] = {wt: valid[cluster][wt] for wt in chosen_wts}
    return valid


def build(out_root: Path) -> None:
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    train_dataset = read_processed("mutations.tsv")
    test_dataset = train_dataset[train_dataset["source"].isin(TEST_SOURCES)]

    cluster_names = list(train_dataset[["cluster"]].drop_duplicates(subset=["cluster"], keep="first")["cluster"])
    valid = valid_clusters(test_dataset, cluster_names)
    print(f"clusters with a usable WT pair: {len(valid)}")

    train_dataset = train_dataset[~train_dataset["cluster"].isin(list(valid.keys()))]

    output_dir = out_root / ROOT_NAME
    for name in ("wt1_train", "wt1_test", "wt2_train", "wt2_test"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    columns = ["mutant_seq", "wt_uid", "wt_seq", "cluster", "ddG"]
    heldout_val = pd.DataFrame()

    for cluster, wts in valid.items():
        wt1, wt2 = list(wts.keys())
        per_wt = {}
        for tag, wt in (("1", wt1), ("2", wt2)):
            wt_slice = test_dataset[test_dataset["wt_uid"] == wt][columns]
            wt_train = wt_slice.sample(TRAIN_SAMPLE_SIZE, random_state=RANDOM_SEED)
            wt_test = wt_slice.drop(wt_train.index).sample(TEST_SAMPLE_SIZE, random_state=RANDOM_SEED)
            heldout_val = pd.concat([heldout_val, wt_test])
            per_wt[tag] = (wt_train, wt_test)

        for tag, (_, wt_test) in per_wt.items():
            wt_test.to_csv(output_dir / f"wt{tag}_test" / f"{cluster}.tsv", sep="\t", index=False)

        for population in TRAIN_POPULATIONS:
            for tag, (wt_train, _) in per_wt.items():
                directory = output_dir / f"wt{tag}_train" / str(population)
                directory.mkdir(parents=True, exist_ok=True)
                wt_train.sample(population, random_state=RANDOM_SEED).to_csv(
                    directory / f"{cluster}.tsv", sep="\t", index=False
                )

        for tag, (wt_train, _) in per_wt.items():
            wt_train.to_csv(output_dir / f"wt{tag}_train" / f"{cluster}.tsv", sep="\t", index=False)

    train_dataset = train_dataset.assign(mutation_type="single")
    assert train_dataset["mutation_type"].eq("single").all()
    train_dataset.to_csv(output_dir / "pretrain.tsv", sep="\t", index=False)
    heldout_val.to_csv(output_dir / "heldout_val.tsv", sep="\t", index=False)
    print(f"pretrain rows: {len(train_dataset)}, held-out val rows: {len(heldout_val)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory to write the split root into")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
