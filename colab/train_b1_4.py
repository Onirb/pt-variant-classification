"""Fine-tuning BERTimbau no Colab sobre divisões previamente congeladas."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

MODEL = "neuralmind/bert-base-portuguese-cased"
REVISION = "94d69c95f98f7d5b2a8700c420230ae10def0baa"
SEEDS = (7, 42, 2026)
LABELS = {0: "PT-PT", 1: "PT-BR"}
PROTOCOL = {
    "model": MODEL, "revision": REVISION, "epochs": 3,
    "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.1,
    "max_length": 256, "batch_size": 8, "gradient_accumulation_steps": 2,
    "evaluation_and_save_steps": 50,
    "selection": "best validation macro_f1 during fixed three-epoch training",
}


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_inputs(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = root / name
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"Arquivo ausente ou alterado: {name}")
    frame = pd.read_parquet(root / "data/development.parquet")
    splits = json.loads((root / "data/splits.json").read_text(encoding="utf-8"))
    if len(frame) != manifest["rows"] or set(frame.label) != {0, 1}:
        raise ValueError("Contagem ou rótulos incompatíveis com o pacote.")
    for seed in SEEDS:
        split = splits[str(seed)]
        train_indexes, validation_indexes = split["train_indices"], split["validation_indices"]
        train_set, validation_set = set(train_indexes), set(validation_indexes)
        if len(train_set) != len(train_indexes) or len(validation_set) != len(validation_indexes):
            raise ValueError(f"Seed {seed}: índices repetidos.")
        if train_set & validation_set or train_set | validation_set != set(range(len(frame))):
            raise ValueError(f"Seed {seed}: índices inválidos ou compartilhados.")
        train, validation = frame.iloc[train_indexes], frame.iloc[validation_indexes]
        if set(train.document_id) & set(validation.document_id):
            raise ValueError(f"Seed {seed}: documento compartilhado.")
        for part in (train, validation):
            if set(part.domain) != set(frame.domain) or set(part.label) != {0, 1}:
                raise ValueError(f"Seed {seed}: falta domínio ou classe.")
    return frame, splits, manifest


class TokenizedTexts(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer):
        self.encodings = tokenizer(frame.text.tolist(), truncation=True, max_length=PROTOCOL["max_length"])
        self.labels = frame.label.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: values[index] for key, values in self.encodings.items()}, "labels": self.labels[index]}


def compute_metrics(evaluation) -> dict[str, float]:
    logits = evaluation.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(evaluation.label_ids, predictions)),
        "macro_f1": float(f1_score(evaluation.label_ids, predictions, labels=[0, 1], average="macro", zero_division=0)),
    }


def detailed_metrics(frame: pd.DataFrame) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(frame.label, frame.prediction, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(frame.label, frame.prediction)),
        "macro_f1": float(f1_score(frame.label, frame.prediction, labels=[0, 1], average="macro", zero_division=0)),
        "per_class": {LABELS[label]: {
            "precision": float(precision[label]), "recall": float(recall[label]),
            "f1": float(f1[label]), "support": int(support[label]),
        } for label in (0, 1)},
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(frame.label, frame.prediction, labels=[0, 1]).tolist(),
        "per_domain": {str(domain): {
            "rows": len(part), "accuracy": float(accuracy_score(part.label, part.prediction)),
            "macro_f1": float(f1_score(part.label, part.prediction, labels=[0, 1], average="macro", zero_division=0)),
        } for domain, part in frame.groupby("domain")},
    }


def training_arguments(output: Path, seed: int, *, smoke: bool = False, cpu_test: bool = False):
    """cpu_test é usado somente pelo teste de integração com uma rede minúscula."""
    interval = 10 if smoke else PROTOCOL["evaluation_and_save_steps"]
    return TrainingArguments(
        output_dir=str(output), num_train_epochs=PROTOCOL["epochs"],
        max_steps=10 if smoke else -1,
        per_device_train_batch_size=PROTOCOL["batch_size"], per_device_eval_batch_size=16,
        gradient_accumulation_steps=PROTOCOL["gradient_accumulation_steps"],
        learning_rate=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"],
        warmup_ratio=PROTOCOL["warmup_ratio"], optim="adamw_torch",
        eval_strategy="steps", eval_steps=interval, save_strategy="steps", save_steps=interval,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,
        fp16=not cpu_test, gradient_checkpointing=not cpu_test,
        logging_steps=5 if smoke else 10, seed=seed, data_seed=seed,
        dataloader_num_workers=0, report_to="none", push_to_hub=False, use_cpu=cpu_test,
    )


def run_seed(root: Path, output_root: Path, frame: pd.DataFrame, splits: dict, seed: int, smoke: bool):
    output = output_root / ("smoke" if smoke else "full") / f"seed_{seed}"
    report_path = output / "report.json"
    expected = {"protocol": PROTOCOL, "seed": seed, "smoke": smoke,
                "package_sha256": digest_file(root / "package_manifest.json")}
    state_path = output / "run_config.json"
    if state_path.exists() and json.loads(state_path.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"Configuração diferente de uma execução existente: {output}")
    if report_path.exists():
        print(f"Seed {seed} já concluída: {report_path}", flush=True)
        return json.loads(report_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    write_json(state_path, expected)
    split = splits[str(seed)]
    train = frame.iloc[split["train_indices"]].reset_index(drop=True)
    validation = frame.iloc[split["validation_indices"]].reset_index(drop=True)
    if smoke:
        validation = validation.groupby(["domain", "label"], group_keys=False).head(4).reset_index(drop=True)
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, num_labels=2, id2label=LABELS,
        label2id={value: key for key, value in LABELS.items()},
    )
    trainer = Trainer(
        model=model, args=training_arguments(output, seed, smoke=smoke),
        train_dataset=TokenizedTexts(train, tokenizer), eval_dataset=TokenizedTexts(validation, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer, compute_metrics=compute_metrics,
    )
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    checkpoint = get_last_checkpoint(str(output))
    if checkpoint:
        print(f"Retomando seed {seed}: {checkpoint}", flush=True)
    trained = trainer.train(resume_from_checkpoint=checkpoint)
    training_seconds = perf_counter() - started
    result = trainer.predict(trainer.eval_dataset)
    predictions = validation[["text", "label", "domain", "document_id", "segment_id", "lp"]].copy()
    predictions.insert(0, "seed", seed)
    predictions["prediction"] = np.argmax(result.predictions, axis=-1)
    predictions["logit_difference_br_minus_pt"] = result.predictions[:, 1] - result.predictions[:, 0]
    predictions.to_parquet(output / "predictions.parquet", index=False)
    trainer.save_model(str(output / "model"))
    tokenizer.save_pretrained(output / "model")
    report = {
        **expected, "experiment": "B1.4 BERTimbau full fine-tuning",
        "scope": "GPU smoke only, not a candidate metric" if smoke else "development validation only",
        "train_rows": len(train), "validation_rows": len(validation),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "training_metrics": trained.metrics, "training_seconds_this_session": training_seconds,
        "global_steps": trainer.state.global_step,
        "planned_full_steps_per_seed": math.ceil(len(train) / 16) * PROTOCOL["epochs"],
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gpu_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "accelerate", "numpy", "pandas", "scikit-learn")},
        "python": platform.python_version(), "cuda": torch.version.cuda,
        **detailed_metrics(predictions),
    }
    write_json(report_path, report)
    print(json.dumps({key: report[key] for key in ("seed", "scope", "macro_f1", "training_seconds_this_session", "peak_allocated_gpu_gib")}, indent=2), flush=True)
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    frame, splits, manifest = load_inputs(args.package_root)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU ausente. No Colab: Ambiente de execução > Alterar tipo de ambiente > GPU. Reconecte e execute novamente.")
    seeds = (args.seed,) if args.seed is not None else ((42,) if args.mode == "smoke" else SEEDS)
    for seed in seeds:
        run_seed(args.package_root, args.output_dir, frame, splits, seed, args.mode == "smoke")
    if args.mode == "full":
        complete = []
        for seed in SEEDS:
            path = args.output_dir / "full" / f"seed_{seed}" / "report.json"
            if path.exists():
                complete.append(json.loads(path.read_text(encoding="utf-8")))
        if len(complete) == len(SEEDS):
            scores = np.array([report["macro_f1"] for report in complete])
            write_json(args.output_dir / "full" / "aggregate.json", {
                "experiment": "B1.4", "seeds": list(SEEDS),
                "macro_f1_mean": float(scores.mean()), "macro_f1_std": float(scores.std()), "reports": complete,
            })


if __name__ == "__main__":
    main()
