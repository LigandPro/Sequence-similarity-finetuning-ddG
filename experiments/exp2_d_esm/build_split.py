"""Rebuild the Exp 2.3 embedding-distance splits.

``--arm within`` keeps wild-type pairs from the same cluster, ``--arm across`` pairs from
different clusters; each arm has its own pretraining corpus and validation set.

One seeded global generator is consumed in sequence, so the order of the sampling calls below
is load-bearing: moving one changes every file after it.  ``heldout_val.tsv`` is drawn from
the global generator at the very end.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "sources"
DEFAULT_OUT = REPO_ROOT / "data" / "splits"

# The two arms were built with these settings; both differences matter for the split:
#   seed — 1218 for the across arm, 42 for the within arm.
#   pretrain_mutation_type_single — whether ``mutation_type`` is set to ``single`` on the
#     pretraining corpus before it is written; only the within arm has it.
ARMS = {
    "within": {
        "root": "exp2_d_esm/within",
        "seed": 42,
        "pretrain_mutation_type_single": True,
    },
    "across": {
        "root": "exp2_d_esm/across",
        "seed": 1218,
        "pretrain_mutation_type_single": False,
    },
}

GLOBAL_SEED = 42

MINIMUM_LENGTH = 1000
PAIRS_TO_SAMPLE = 5
TRAIN_SAMPLE_SIZE = 800
TEST_SAMPLE_SIZE = 200
BINS = np.array([0.001, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])

SOURCES = ["mega_smdi", "fireprot", "Ssym", "tmdbs", "S2648", "S669"]
CLUSTER_FILE = "mega_S2648_S669_Ssym_p53_M_ptmuld_ptmulnr_fireprot_tmdbs_tmdbm_cluster.tsv"
COLUMNS_TO_KEEP = ["wt_uid", "ddG", "mutant_seq", "wt_seq", "mutation_type"]
COLUMNS_TO_SAVE = ["mutant_seq", "wt_uid", "wt_seq", "cluster", "ddG", "mutation_type"]


def preprocess(dataset: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    if "mutation_type" not in dataset.columns:
        dataset["mutation_type"] = "single"
    dataset = dataset[COLUMNS_TO_KEEP]
    dataset = dataset.merge(clusters[["wt_uid", "cluster"]], on="wt_uid", how="left")
    return dataset.drop_duplicates(subset=["mutant_seq", "wt_uid"])


def desymmetrise(blast: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per unordered pair, retaining every self-pair."""
    qseqid = blast["qseqid"].values
    sseqid = blast["sseqid"].values
    pair_df = blast.copy()
    pair_df["dup_key"] = list(zip(np.minimum(qseqid, sseqid), np.maximum(qseqid, sseqid)))
    duplicated = (qseqid != sseqid) & pair_df.duplicated(subset="dup_key", keep="first")
    return blast.loc[~duplicated].reset_index(drop=True)


def load_blast(clusters: pd.DataFrame) -> pd.DataFrame:
    """Unordered WT pairs, each carrying the squared L2 distance between the two ESM embeddings."""
    blast = pd.read_csv(RAW / "ALL_SEQS_blastp_named.tsv", sep="\t")
    blast["qseqid"] = blast["qseqid"].str.split("_").str[0]
    blast["sseqid"] = blast["sseqid"].str.split("_").str[0]
    wt_uid_to_cluster = clusters.set_index("wt_uid")["cluster"]
    blast["qseq_cluster"] = blast["qseqid"].map(wt_uid_to_cluster)
    blast["sseq_cluster"] = blast["sseqid"].map(wt_uid_to_cluster)
    blast.drop_duplicates(subset=["qseqid", "sseqid"], inplace=True)
    blast = desymmetrise(blast)

    embeddings = pd.read_csv(RAW / "wt_embeddings.tsv", sep="\t", usecols=["wt_uid", "avg_wt_emb"])
    lookup = {
        uid: np.fromstring(text.strip()[1:-1], sep=",")
        for uid, text in zip(embeddings["wt_uid"], embeddings["avg_wt_emb"])
    }
    blast["avg_emb_dist"] = [
        float(np.sum((lookup[q] - lookup[s]) ** 2)) for q, s in zip(blast["qseqid"], blast["sseqid"])
    ]
    return blast


def sample_pairs(blast: pd.DataFrame, train_dataset: pd.DataFrame, seed: int) -> dict:
    """Up to five WT pairs per distance bin, among pairs whose wild types are both well populated."""
    wt_uid_counts = train_dataset["wt_uid"].value_counts()
    qseqids, sseqids = blast["qseqid"].values, blast["sseqid"].values
    valid = (
        (qseqids != sseqids)
        & (wt_uid_counts.reindex(qseqids, fill_value=0).values > MINIMUM_LENGTH)
        & (wt_uid_counts.reindex(sseqids, fill_value=0).values > MINIMUM_LENGTH)
    )
    distances = blast["avg_emb_dist"].values
    pairs = {(qseqids[i], sseqids[i]): distances[i] for i in np.where(valid)[0]}

    pairs_df = pd.DataFrame(
        [(q, s, d) for (q, s), d in pairs.items()], columns=["qseqid", "sseqid", "avg_emb_dist"]
    )
    pairs_df["score_bin"] = pd.cut(pairs_df["avg_emb_dist"], bins=BINS, include_lowest=True, right=True)

    sampled = {}
    for interval in pairs_df["score_bin"].unique():
        bin_df = pairs_df[pairs_df["score_bin"] == interval]
        if bin_df.empty:
            continue
        chosen = bin_df.sample(n=min(PAIRS_TO_SAMPLE, len(bin_df)), random_state=seed)
        sampled[str(interval)] = chosen[["qseqid", "sseqid", "avg_emb_dist"]].reset_index(drop=True)
    return sampled


def build(arm: str, out_root: Path) -> None:
    settings = ARMS[arm]
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    clusters = pd.read_csv(RAW / CLUSTER_FILE, sep="\t").drop_duplicates(subset=["wt_uid"], keep="first")
    train_dataset = preprocess(
        pd.concat([pd.read_csv(RAW / f"{s}.tsv", sep="\t", low_memory=False) for s in SOURCES]), clusters
    )

    blast = load_blast(clusters)
    same_cluster = blast["qseq_cluster"] == blast["sseq_cluster"]
    blast = blast[same_cluster if arm == "within" else ~same_cluster]
    blast = desymmetrise(blast)

    sampled_per_bin = sample_pairs(blast, train_dataset, settings["seed"])

    if settings["pretrain_mutation_type_single"]:
        train_dataset["mutation_type"] = "single"

    all_sampled = pd.concat(list(sampled_per_bin.values()))
    sampled_wt = list(set(np.concatenate([all_sampled.qseqid.unique(), all_sampled.sseqid.unique()])))
    sampled_clusters = train_dataset[train_dataset["wt_uid"].isin(sampled_wt)]["cluster"].unique()
    clean_train_dataset = train_dataset[~train_dataset["cluster"].isin(sampled_clusters)]
    print(f"{arm}: {len(sampled_wt)} wild types over {len(sampled_clusters)} clusters held out")

    train_dataset["mutation_type"] = "single"

    output_dir = out_root / settings["root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_train_dataset.to_csv(output_dir / "pretrain.tsv", sep="\t", index=False)

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
                wt_train = wt_data.sample(n=TRAIN_SAMPLE_SIZE, random_state=settings["seed"])
                wt_test = wt_data.drop(wt_train.index).sample(n=TEST_SAMPLE_SIZE, random_state=settings["seed"])
                per_wt[tag] = (wt_train, wt_test)
            if skip:
                continue

            file_name = f"{wt1}_{wt2}.tsv"
            for tag, (wt_train, wt_test) in per_wt.items():
                wt_train[COLUMNS_TO_SAVE].to_csv(bin_dir / f"wt{tag}_train" / file_name, sep="\t", index=False)
                wt_test[COLUMNS_TO_SAVE].to_csv(bin_dir / f"wt{tag}_test" / file_name, sep="\t", index=False)
            wt_tests.extend([per_wt["1"][1], per_wt["2"][1]])
            print(f"  saved {file_name} (d_esm={row['avg_emb_dist']:.3f})")

    # heldout_val.tsv is drawn from the global generator, so it depends on everything above.
    all_wt_tests = pd.concat(wt_tests)[COLUMNS_TO_SAVE]
    all_wt_tests.sample(TEST_SAMPLE_SIZE).reset_index(drop=True).to_csv(
        output_dir / "heldout_val.tsv", sep="\t", index=False
    )
    print(f"pretrain rows: {len(clean_train_dataset)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True, help="which cluster arm to build")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory to write the split root into")
    args = parser.parse_args()
    build(args.arm, args.out)


if __name__ == "__main__":
    main()
