"""Fine-tune one model per cluster from a shared pretrained checkpoint.

``FLEET.TRAIN_DIR`` holds one ``<cluster>.tsv`` per cluster; each becomes its own model,
evaluated against the matching file in ``FLEET.TEST_DIR`` and, when present, ``FLEET.VAL_DIR``.
The per-cluster metrics land in one TSV — averaging across clusters happens in
``thermality.collect_results``, not here.  One GPU; pick it with ``CUDA_VISIBLE_DEVICES`` if the
machine has more than one.
"""

import argparse
import re
import warnings
from functools import partial
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from thermality.config import PROJECT_ROOT, get_cfg_defaults, resolve_path, run_identity
from thermality.model.dataset import ListDataset
from thermality.model.utils import compute_metrics, custom_collate, write_metrics


def find_clusters(train_dir, val_dir, test_dir):
    """One entry per cluster that has both a training and a test file."""
    train_dir, test_dir = Path(train_dir), Path(test_dir)
    val_dir = Path(val_dir) if val_dir else None

    clusters = []
    for train_path in sorted(train_dir.glob("*.tsv")):
        if train_path.stem.startswith("all"):
            continue
        test_path = test_dir / train_path.name
        if not test_path.exists():
            print(f"  ! no test file for {train_path.stem}, skipping")
            continue
        val_path = val_dir / train_path.name if val_dir else None
        clusters.append(
            {
                "name": train_path.stem,
                "train": train_path,
                "val": val_path if val_path and val_path.exists() else None,
                "test": test_path,
            }
        )
    return clusters


def resolve_checkpoint(directory):
    """The last ``checkpoint-N`` directory under ``FLEET.PRETRAINED_CHECKPOINT``."""
    directory = Path(directory)
    checkpoints = sorted(directory.glob("checkpoint-*"), key=lambda p: int(re.sub(r"\D", "", p.name)))
    if not checkpoints:
        raise FileNotFoundError(
            f"no checkpoint-* directory under {directory} (FLEET.PRETRAINED_CHECKPOINT): "
            "train this experiment's pretrain config first"
        )
    return checkpoints[-1]


def load_model(cfg, device, checkpoint):
    ddGRegressor = import_module(f"thermality.model.{cfg.MODEL.ARCHITECTURE}").ddGRegressor
    model = ddGRegressor(cfg=cfg).to(device)
    if checkpoint is None:
        print("  → random initialisation")
        return model

    weights = Path(checkpoint) / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"no model.safetensors in {checkpoint}")
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(weights), device="cpu"))
    print(f"  → pretrained weights from {checkpoint}")
    return model


def train_cluster(cfg, cluster, run_name, output_dir, device, checkpoint, tokenizer):
    """Train one cluster's model and return its evaluation history."""
    name = cluster["name"]
    print(f"\n=== {name} ===")

    train_dataset = ListDataset(str(cluster["train"]), reverse_augmentation=True)
    eval_datasets = {f"{name}_test": ListDataset(str(cluster["test"]))}
    eval_names = {f"{name}_test": ("test", name)}
    if cluster["val"]:
        eval_datasets[f"{name}_val"] = ListDataset(str(cluster["val"]))
        eval_names[f"{name}_val"] = ("val", name)
    print(f"  train {len(train_dataset)}, " + ", ".join(f"{k} {len(v)}" for k, v in eval_datasets.items()))

    best_metric = f"eval_{name}_{'val' if cluster['val'] else 'test'}_spearmanr"

    training_args = TrainingArguments(
        run_name=f"{run_name}__{name}",
        output_dir=str(output_dir / name),
        do_train=True,
        do_eval=True,
        max_steps=cfg.TRAIN.MAX_STEPS,
        per_device_train_batch_size=cfg.TRAIN.BATCH_SIZE,
        per_device_eval_batch_size=cfg.TEST.BATCH_SIZE,
        learning_rate=cfg.TRAIN.LR,
        lr_scheduler_type=cfg.TRAIN.SCHEDULER,
        lr_scheduler_kwargs=cfg.TRAIN.SCHEDULER_KWARGS,
        warmup_ratio=0.2,
        weight_decay=0.01,
        optim="ademamix",
        eval_strategy="steps",
        eval_steps=cfg.TRAIN.EVAL_STEPS,
        logging_strategy="steps",
        logging_steps=cfg.TRAIN.EVAL_STEPS,
        save_strategy="steps",
        save_steps=cfg.TRAIN.EVAL_STEPS,
        save_total_limit=2,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        bf16=device.type == "cuda",
        seed=cfg.TRAIN.SEED,
        report_to="none",
        dataloader_num_workers=16 if device.type == "cuda" else 1,
        dataloader_prefetch_factor=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        max_grad_norm=cfg.TRAIN.GRAD_NORM,
        gradient_accumulation_steps=cfg.TRAIN.GRAD_ACCUM,
    )

    trainer = Trainer(
        model=load_model(cfg, device, checkpoint),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=partial(custom_collate, tokenizer=tokenizer, truncate_length=cfg.TRAIN.TRUNCATE_LENGTH),
        compute_metrics=partial(compute_metrics, cfg=cfg),
    )

    # trainer.train() starts a fresh TrainerState, so the pre-training evaluation has to be
    # kept by hand.  It is the published N=0 baseline, not a diagnostic.
    step_zero = dict(trainer.evaluate(), step=0)
    trainer.train()
    history = [step_zero, *trainer.state.log_history]

    trainer.model.cpu()
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history, eval_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="path to a config file under an experiment's configs/")
    args = parser.parse_args()

    warnings.simplefilter("ignore")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    if not cfg.FLEET.TRAIN_DIR or not cfg.FLEET.TEST_DIR:
        raise ValueError("FLEET.TRAIN_DIR and FLEET.TEST_DIR are both required")

    experiment, run_name, stem = run_identity(args.config)
    output_dir = PROJECT_ROOT / "checkpoints" / experiment / stem
    metrics_path = PROJECT_ROOT / "results" / "runs" / experiment / f"{run_name}.tsv"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(cfg.TRAIN.SEED)
    torch.manual_seed(cfg.TRAIN.SEED)

    checkpoint = (
        resolve_checkpoint(resolve_path(cfg.FLEET.PRETRAINED_CHECKPOINT))
        if cfg.FLEET.PRETRAINED_CHECKPOINT
        else None
    )

    clusters = find_clusters(
        resolve_path(cfg.FLEET.TRAIN_DIR),
        resolve_path(cfg.FLEET.VAL_DIR),
        resolve_path(cfg.FLEET.TEST_DIR),
    )
    if not clusters:
        raise ValueError(f"no clusters under {cfg.FLEET.TRAIN_DIR}")

    print(f"|| {experiment} / {run_name} on {device}: {len(clusters)} clusters")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL.ESM_CHECKPOINT,
        padding=True,
        max_length=cfg.TRAIN.TRUNCATE_LENGTH,
        use_fast=True,
    )

    history, eval_names = [], {}
    for index, cluster in enumerate(clusters, 1):
        print(f"\n[{index}/{len(clusters)}]", end="")
        cluster_history, cluster_names = train_cluster(
            cfg, cluster, run_name, output_dir, device, checkpoint, tokenizer
        )
        history.extend(cluster_history)
        eval_names.update(cluster_names)

    rows = write_metrics(metrics_path, history, eval_names)
    print(f"\n|| {rows} metric rows over {len(clusters)} clusters -> {metrics_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
