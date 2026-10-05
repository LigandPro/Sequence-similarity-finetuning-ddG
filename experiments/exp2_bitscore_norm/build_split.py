"""Rebuild the Exp 2.2 bitscore split (``data/splits/exp2_bitscore_norm``).

One seeded global generator is consumed in sequence, so the order of the sampling calls below
is load-bearing: moving one changes every file after it.  In particular ``heldout_val.tsv`` is drawn from the
global generator at the very end, after every per-bin sample.
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
ROOT_NAME = "exp2_bitscore_norm"

RANDOM_SEED = 42

MINIMUM_LENGTH = 1000
PAIRS_TO_SAMPLE = 5
TRAIN_SAMPLE_SIZE = 800
TEST_SAMPLE_SIZE = 200
BINS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

COLUMNS_TO_SAVE = ["mutant_seq", "wt_uid", "wt_seq", "cluster", "ddG"]


def read_processed(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        sys.exit("run python data/preprocess.py first")
    return pd.read_csv(path, sep="\t", low_memory=False)


def sample_pairs(blast: pd.DataFrame, train_dataset: pd.DataFrame) -> dict:
    """Up to five WT pairs per bitscore bin, among pairs whose wild types are both well populated."""
    wt_uid_counts = train_dataset["wt_uid"].value_counts()
    qseqids, sseqids = blast["qseqid"].values, blast["sseqid"].values
    valid = (
        (qseqids != sseqids)
        & (wt_uid_counts.reindex(qseqids, fill_value=0).values > MINIMUM_LENGTH)
        & (wt_uid_counts.reindex(sseqids, fill_value=0).values > MINIMUM_LENGTH)
    )
    scores = blast["bitscore_norm"].values
    pairs = {(qseqids[i], sseqids[i]): scores[i] for i in np.where(valid)[0]}

    pairs_df = pd.DataFrame(
        [(q, s, score) for (q, s), score in pairs.items()], columns=["qseqid", "sseqid", "bitscore_norm"]
    )
    pairs_df["score_bin"] = pd.cut(pairs_df["bitscore_norm"], bins=BINS, include_lowest=True, right=True)

    sampled = {}
    for interval in pairs_df["score_bin"].unique():
        bin_df = pairs_df[pairs_df["score_bin"] == interval]
        if bin_df.empty:
            continue
        chosen = bin_df.sample(n=min(PAIRS_TO_SAMPLE, len(bin_df)), random_state=RANDOM_SEED)
        sampled[str(interval)] = chosen[["qseqid", "sseqid", "bitscore_norm"]].reset_index(drop=True)
    return sampled


def write_bins(sampled_per_bin: dict, train_dataset: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """One directory per bin, holding a train and a test split for each wild type of each pair."""
    wt_tests: list[pd.DataFrame] = []
    for bin_label, pairs_in_bin in sampled_per_bin.items():
        folder = bin_label.replace("(", "").replace("]", "").replace(",", "_").replace(" ", "")
        bin_dir = output_dir / folder
        for name in ("wt1_train", "wt2_train", "wt1_test", "wt2_test"):
            (bin_dir / name).mkdir(parents=True, exist_ok=True)

        for _, row in pairs_in_bin.iterrows():
            wt1, wt2 = row["qseqid"], row["sseqid"]
            per_wt = {}
            skip = False
            for tag, wt in (("1", wt1), ("2", wt2)):
                wt_data = train_dataset[train_dataset["wt_uid"] == wt]
                if len(wt_data) < TRAIN_SAMPLE_SIZE + TEST_SAMPLE_SIZE:
                    print(f"  wt{tag}={wt} has only {len(wt_data)} rows, skipping")
                    skip = True
                    break
                wt_train = wt_data.sample(n=TRAIN_SAMPLE_SIZE, random_state=RANDOM_SEED)
                wt_test = wt_data.drop(wt_train.index).sample(n=TEST_SAMPLE_SIZE, random_state=RANDOM_SEED)
                per_wt[tag] = (wt_train, wt_test)
            if skip:
                continue

            wt_tests.extend([per_wt["1"][1], per_wt["2"][1]])
            file_name = f"{wt1}_{wt2}.tsv"
            for tag, (wt_train, wt_test) in per_wt.items():
                wt_train[COLUMNS_TO_SAVE].to_csv(bin_dir / f"wt{tag}_train" / file_name, sep="\t", index=False)
                wt_test[COLUMNS_TO_SAVE].to_csv(bin_dir / f"wt{tag}_test" / file_name, sep="\t", index=False)
            print(f"  saved {file_name} (bitscore_norm={row['bitscore_norm']:.3f})")
    # heldout_val.tsv keeps this column order, not the order the sampled rows carry.
    return pd.concat(wt_tests)[["mutant_seq", "wt_uid", "wt_seq", "cluster", "ddG"]]


def build(out_root: Path) -> None:
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    train_dataset = read_processed("mutations.tsv")
    assert train_dataset.index.is_unique
    blast = read_processed("wt_pairs.tsv")
    blast = blast[blast["same_cluster"]]

    sampled_per_bin = sample_pairs(blast, train_dataset)

    all_sampled = pd.concat(list(sampled_per_bin.values()))
    sampled_wt = list(set(np.concatenate([all_sampled.qseqid.unique(), all_sampled.sseqid.unique()])))
    sampled_clusters = train_dataset[train_dataset["wt_uid"].isin(sampled_wt)]["cluster"].unique()
    clean_train_dataset = train_dataset[~train_dataset["cluster"].isin(sampled_clusters)]
    print(f"{len(sampled_wt)} wild types over {len(sampled_clusters)} clusters held out")

    output_dir = out_root / ROOT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_train_dataset = clean_train_dataset.assign(mutation_type="single")
    assert clean_train_dataset["mutation_type"].eq("single").all()
    clean_train_dataset.to_csv(output_dir / "pretrain.tsv", sep="\t", index=False)

    all_wt_tests = write_bins(sampled_per_bin, train_dataset, output_dir)
    all_wt_tests.sample(TEST_SAMPLE_SIZE).reset_index(drop=True).to_csv(
        output_dir / "heldout_val.tsv", sep="\t", index=False
    )
    print(f"pretrain rows: {len(clean_train_dataset)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory to write the split root into")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
