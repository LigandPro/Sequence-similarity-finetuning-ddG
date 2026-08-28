"""Pretrain one model on a corpus, evaluating against one or more cluster directories.

One config, one GPU, one model.  Pick the GPU with ``CUDA_VISIBLE_DEVICES`` if the machine has
more than one.  The run is identified by the config's path inside its experiment, so
``experiments/exp1_data_inclusion/configs/pretrain.yaml`` writes checkpoints to
``checkpoints/exp1_data_inclusion/pretrain/`` and metrics to
``results/runs/exp1_data_inclusion/pretrain.tsv``.  The fine-tuning configs of the same
experiment start from the last ``checkpoint-N`` left under that directory.
"""

import argparse
import os
import warnings
from functools import partial
from glob import glob
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from thermality.config import PROJECT_ROOT, get_cfg_defaults, resolve_path, run_identity
from thermality.model.dataset import HierarchicalSyntheticDataset, ListDataset
from thermality.model.utils import (
    MetricAggregationCallback,
    compute_metrics,
    custom_collate,
    write_metrics,
)

TARGET_METRIC = "mae"


def load_eval_datasets(cfg):
    """The evaluation sets named in ``TEST.DATASETS``, plus what each one is called.

    A directory is a multicluster dataset: one evaluation set per ``*.tsv`` in it, whose metrics
    the aggregation callback averages back into a single dataset-level number.  A file is a
    single set.  Returns ``(datasets, names, multicluster_counts)`` where ``names`` maps an
    evaluation set to the ``(split, cluster)`` pair it reports under.
    """
    datasets, names, multicluster = {}, {}, {}

    for dataset_name, dataset_path, reverse in cfg.TEST.DATASETS:
        if not dataset_path:
            raise ValueError(f"TEST.DATASETS entry {dataset_name!r} has no path")
        path = resolve_path(dataset_path)

        def add(key, split, cluster, source, reversed_labels=False):
            datasets[key] = ListDataset(source, reverse=reversed_labels)
            names[key] = (split, cluster)

        if os.path.isdir(path):
            cluster_files = [f for f in sorted(glob(f"{path}/*.tsv")) if not Path(f).stem.startswith("all")]
            print(f"|| {dataset_name}: {len(cluster_files)} clusters in {dataset_path}")
            for cluster_file in cluster_files:
                cluster = Path(cluster_file).stem
                add(f"{dataset_name}_{cluster}", dataset_name, cluster, cluster_file)
                if reverse:
                    add(f"reverse_{dataset_name}_{cluster}", f"reverse_{dataset_name}", cluster, cluster_file, True)
            multicluster[dataset_name] = len(cluster_files)
            if reverse:
                multicluster[f"reverse_{dataset_name}"] = len(cluster_files)
        else:
            print(f"|| {dataset_name}: single evaluation set")
            add(dataset_name, dataset_name, "", path)
            if reverse:
                add(f"reverse_{dataset_name}", f"reverse_{dataset_name}", "", path, True)

    return datasets, names, multicluster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="path to a config file under an experiment's configs/")
    parser.add_argument("--resume", default=None, help="checkpoint directory to resume from")
    args = parser.parse_args()

    warnings.simplefilter("ignore")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    experiment, run_name, stem = run_identity(args.config)
    output_dir = PROJECT_ROOT / "checkpoints" / experiment / stem
    metrics_path = PROJECT_ROOT / "results" / "runs" / experiment / f"{run_name}.tsv"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"|| {experiment} / {run_name} on {device}")

    train_dataset = HierarchicalSyntheticDataset(
        resolve_path(cfg.TRAIN.DATASET_PATH),
        mutation_types=cfg.TRAIN.MUTATION_TYPES,
        reverse_augmentation=True,
    )
    eval_datasets, eval_names, multicluster = load_eval_datasets(cfg)

    np.random.seed(cfg.TRAIN.SEED)
    torch.manual_seed(cfg.TRAIN.SEED)

    ddGRegressor = import_module(f"thermality.model.{cfg.MODEL.ARCHITECTURE}").ddGRegressor
    model = ddGRegressor(cfg=cfg).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL.ESM_CHECKPOINT,
        padding=True,
        max_length=cfg.TRAIN.TRUNCATE_LENGTH,
        use_fast=True,
    )

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        max_steps=cfg.TRAIN.MAX_STEPS,
        per_device_train_batch_size=cfg.TRAIN.BATCH_SIZE,
        per_device_eval_batch_size=cfg.TEST.BATCH_SIZE,
        dataloader_num_workers=2 if device.type == "cuda" else 1,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        lr_scheduler_type=cfg.TRAIN.SCHEDULER,
        lr_scheduler_kwargs=cfg.TRAIN.SCHEDULER_KWARGS,
        warmup_ratio=0.2,
        learning_rate=cfg.TRAIN.LR,
        weight_decay=0.01,
        optim="ademamix",
        metric_for_best_model=f"eval_{cfg.TEST.TARGET_DATASET}_{TARGET_METRIC}",
        greater_is_better=False,
        eval_strategy="steps",
        eval_steps=cfg.TRAIN.EVAL_STEPS,
        log_level="warning",
        logging_strategy="steps",
        logging_steps=100,
        report_to="none",
        include_for_metrics=["inputs"],
        include_tokens_per_second=True,
        bf16=device.type == "cuda",
        seed=cfg.TRAIN.SEED,
        save_strategy="steps",
        save_steps=cfg.TRAIN.EVAL_STEPS,
        save_total_limit=20,
        save_safetensors=True,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        max_grad_norm=cfg.TRAIN.GRAD_NORM,
        gradient_accumulation_steps=cfg.TRAIN.GRAD_ACCUM,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=partial(custom_collate, tokenizer=tokenizer, truncate_length=cfg.TRAIN.TRUNCATE_LENGTH),
        compute_metrics=partial(compute_metrics, cfg=cfg),
        callbacks=[MetricAggregationCallback(multicluster)] if multicluster else [],
    )

    # trainer.train() starts a fresh TrainerState, so the pre-training evaluation has to be
    # kept by hand.  It is the published N=0 baseline, not a diagnostic.
    step_zero = dict(trainer.evaluate(), step=0)
    trainer.train(resume_from_checkpoint=args.resume)

    rows = write_metrics(metrics_path, [step_zero, *trainer.state.log_history], eval_names)
    print(f"|| {rows} metric rows -> {metrics_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
