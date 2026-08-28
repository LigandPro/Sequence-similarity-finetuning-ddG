"""Collation, metrics, and the multicluster aggregation callback."""

import csv

import numpy as np
import torch
from scipy import stats
from transformers import TrainerCallback

METRICS = ["rmse", "mae", "spearmanr", "pearsonr", "dispersion_ratio"]
METRIC_COLUMNS = ["step", "split", "cluster", *METRICS]

MAX_SEQUENCE_LENGTH = 800


class MetricAggregationCallback(TrainerCallback):
    """Average a multicluster dataset's per-cluster metrics into one dataset-level metric.

    ``eval_target_clusterA_rmse`` and ``eval_target_clusterB_rmse`` become ``eval_target_rmse``,
    so ``metric_for_best_model`` can name a whole dataset regardless of how many clusters it
    holds.  Aggregation waits until every expected cluster has reported, because the Trainer
    evaluates one dataset at a time.
    """

    def __init__(self, multicluster_info):
        """``multicluster_info`` maps a dataset name to how many clusters it should have."""
        self.multicluster_info = multicluster_info
        self.accumulated_metrics = {}
        self.current_epoch = -1
        self.aggregated_this_epoch = set()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        if state.epoch != self.current_epoch:
            self.current_epoch = state.epoch
            self.accumulated_metrics = {}
            self.aggregated_this_epoch = set()

        self.accumulated_metrics.update(metrics)

        for dataset_name, expected_count in self.multicluster_info.items():
            for metric_name in METRICS:
                aggregated_key = f"eval_{dataset_name}_{metric_name}"
                if aggregated_key in self.aggregated_this_epoch:
                    continue

                prefix, suffix = f"eval_{dataset_name}_", f"_{metric_name}"
                values = [
                    value
                    for key, value in self.accumulated_metrics.items()
                    if key.startswith(prefix)
                    and key.endswith(suffix)
                    and key != aggregated_key
                    # "all" files are pre-aggregated datasets of their own, not clusters
                    and key[len(prefix) : -len(suffix)] != "all"
                ]

                if len(values) == expected_count:
                    aggregated = sum(values) / len(values)
                    metrics[aggregated_key] = aggregated
                    self.accumulated_metrics[aggregated_key] = aggregated
                    self.aggregated_this_epoch.add(aggregated_key)


def custom_collate(batch, tokenizer=None, truncate_length=None):
    # truncate_length is the tokenizer's configured limit; every published run padded and
    # truncated to MAX_SEQUENCE_LENGTH regardless, so that is what is used here.
    def tokenize(sequences):
        return tokenizer(
            sequences,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH + 2,
            truncation=True,
        )

    clusters = [item.get("cluster") for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])
    wt_sequences = [item["wt"] for item in batch]
    mut_sequences = [item["mut"] for item in batch]

    # The model takes gap and difference positions but does not use them; computing the real
    # alignment here dominated the data loader, so they are passed through empty.
    return {
        "labels": labels,
        "encodings": {"wt": tokenize(wt_sequences), "mut": tokenize(mut_sequences)},
        "gap_positions": {"wt": [[] for _ in batch], "mut": [[] for _ in batch]},
        "difference_position": torch.zeros(len(batch), dtype=torch.long),
        "clusters": clusters,
    }


def compute_metrics(p, cfg=None):
    predictions = p.predictions[: len(p.label_ids)]
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    labels = p.label_ids.flatten() if p.label_ids.ndim > 1 else p.label_ids

    return {
        "dispersion_ratio": float(np.std(predictions) / np.std(labels)),
        "spearmanr": float(stats.spearmanr(labels, predictions).correlation),
        "rmse": float(np.sqrt(np.mean((predictions - labels) ** 2))),
        "mae": float(np.mean(np.abs(predictions - labels))),
        "pearsonr": float(stats.pearsonr(labels, predictions).correlation),
    }


def write_metrics(path, log_history, eval_names):
    """Every evaluation in a Trainer's log history as one row per (step, evaluation set).

    ``eval_names`` maps a Trainer evaluation-set key to the ``(split, cluster)`` pair it reports
    under, so the tags in the output are the experiment's vocabulary rather than the Trainer's.
    """
    rows = []
    for entry in log_history:
        step = entry.get("step")
        for key, (split, cluster) in eval_names.items():
            values = {metric: entry.get(f"eval_{key}_{metric}") for metric in METRICS}
            if all(value is None for value in values.values()):
                continue
            rows.append({"step": step, "split": split, "cluster": cluster, **values})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
