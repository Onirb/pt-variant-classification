"""Ajuste final Track B em todo o desenvolvimento, com protocolo congelado."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

if __package__:
    from .train_b1_4 import LABELS, MODEL, REVISION, TokenizedTexts, digest_file, load_inputs, write_json
else:
    from train_b1_4 import LABELS, MODEL, REVISION, TokenizedTexts, digest_file, load_inputs, write_json


PROBES = [
    "Estou a estudar aprendizagem automática e preciso de rever este código.",
    "Estou estudando aprendizado de máquina e preciso revisar esse código.",
    "Olá! A análise contém acentuação: ação, revisão, probabilidade e amanhã.",
]


def final_training_arguments(output: Path, selection: dict, *, cpu_test=False):
    protocol = selection["final_protocol"]
    return TrainingArguments(
        output_dir=str(output), max_steps=protocol["max_steps"],
        per_device_train_batch_size=protocol["batch_size"],
        gradient_accumulation_steps=protocol["gradient_accumulation_steps"],
        learning_rate=protocol["learning_rate"], weight_decay=protocol["weight_decay"],
        warmup_ratio=protocol["warmup_ratio"], optim="adamw_torch",
        eval_strategy="no", save_strategy="steps", save_steps=50, save_total_limit=2,
        load_best_model_at_end=False, fp16=not cpu_test,
        gradient_checkpointing=not cpu_test, logging_steps=10,
        seed=protocol["seed"], data_seed=protocol["seed"],
        dataloader_num_workers=0, report_to="none", push_to_hub=False, use_cpu=cpu_test,
    )


def probe_logits(model, tokenizer, texts=PROBES):
    model.eval()
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    device = next(model.parameters()).device
    with torch.inference_mode():
        return model(**{key: value.to(device) for key, value in tokens.items()}).logits.float().cpu().numpy()


def validate_saved_artifact(artifact: Path) -> dict:
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = (artifact / name).resolve()
        if not path.is_relative_to(artifact.resolve()) or digest_file(path) != expected:
            raise ValueError(f"Artefato alterado ou caminho inválido: {name}")
    if manifest["global_steps"] != manifest["final_protocol"]["max_steps"]:
        raise ValueError("Ajuste final incompleto.")
    model = AutoModelForSequenceClassification.from_pretrained(artifact / "model", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(artifact / "model", local_files_only=True)
    first = probe_logits(model, tokenizer)
    second = probe_logits(model, tokenizer)
    if not np.isfinite(first).all() or not np.allclose(first, second, rtol=1e-5, atol=1e-6):
        raise ValueError("Inferência recarregada instável ou não finita.")
    expected = np.array(manifest["cpu_reload_probe_logits"])
    if not np.allclose(first, expected, rtol=1e-4, atol=1e-4):
        raise ValueError("Inferência recarregada diverge do recibo salvo.")
    if model.config.id2label != LABELS:
        raise ValueError("Mapa de classes incompatível.")
    del model, tokenizer
    return {"files_and_steps_verified": True, "cpu_reload_deterministic": True,
            "labels": [LABELS[int(value)] for value in first.argmax(axis=1)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(validate_saved_artifact(args.output_dir / "artifact"), indent=2))
        return
    frame, _, package_manifest = load_inputs(args.package_root)
    selection = json.loads((args.package_root / "selection.json").read_text(encoding="utf-8"))
    if selection["model"] != MODEL or selection["revision"] != REVISION or len(frame) != 1877:
        raise ValueError("Modelo, revisão ou corpus divergem da seleção final.")
    selection_hash = digest_file(args.package_root / "selection.json")
    run_config = {"selection_sha256": selection_hash,
                  "package_sha256": digest_file(args.package_root / "package_manifest.json")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "run_config.json"
    if state_path.exists() and json.loads(state_path.read_text()) != run_config:
        raise ValueError("Configuração incompatível com execução final existente.")
    artifact = args.output_dir / "artifact"
    if (artifact / "manifest.json").exists():
        print("Modelo final já concluído; verificando o artefato existente.", flush=True)
        print(json.dumps(validate_saved_artifact(artifact), indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Selecione GPU no Colab antes de iniciar o ajuste final.")
    write_json(state_path, run_config)
    protocol = selection["final_protocol"]
    if protocol["world_size"] != 1:
        raise ValueError("O protocolo final foi definido para uma única GPU.")
    set_seed(protocol["seed"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, num_labels=2, id2label=LABELS,
        label2id={value: key for key, value in LABELS.items()},
    )
    trainer = Trainer(
        model=model, args=final_training_arguments(args.output_dir / "checkpoints", selection),
        train_dataset=TokenizedTexts(frame, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
    )
    if trainer.args.world_size != 1:
        raise ValueError("Use uma GPU; o orçamento depende do batch efetivo fixo.")
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = get_last_checkpoint(str(checkpoint_dir))
    if checkpoint:
        print(f"Retomando ajuste final: {checkpoint}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    trained = trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = perf_counter() - started
    if trainer.state.global_step != protocol["max_steps"]:
        raise ValueError("Treino terminou antes do orçamento fixado.")
    # Artefato safetensors + tokenizer, sem pickle, otimizador ou dados.
    trainer.model.save_pretrained(artifact / "model", safe_serialization=True)
    tokenizer.save_pretrained(artifact / "model")
    cuda_probe = probe_logits(trainer.model, tokenizer)
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    trainer.model.to("cpu")
    torch.cuda.empty_cache()
    cpu_probe = probe_logits(trainer.model, tokenizer)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(artifact / "model", local_files_only=True)
    loaded_tokenizer = AutoTokenizer.from_pretrained(artifact / "model", local_files_only=True)
    reload_probe = probe_logits(loaded_model, loaded_tokenizer)
    if not np.allclose(cpu_probe, reload_probe, rtol=1e-5, atol=1e-5):
        raise ValueError("Pesos recarregados divergem da inferência CPU antes da exportação.")
    # Estes textos são sondas de funcionamento, não exemplos rotulados de teste.
    if not np.isfinite(reload_probe).all():
        raise ValueError("Logits não finitos.")
    write_json(artifact / "selection.json", selection)
    files = sorted(path for path in artifact.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "1.0", "artifact_type": "track_b_fine_tuned_classifier",
        "scope": "final fit on all development; external final test still reserved",
        "model": MODEL, "revision": REVISION, "label_map": LABELS,
        "rows_fit": len(frame), "document_count": int(frame.document_id.nunique()),
        "development_data_sha256": package_manifest["sha256"]["data/development.parquet"],
        "final_protocol": protocol, "global_steps": trainer.state.global_step,
        "training_metrics": trained.metrics, "training_seconds_this_session": elapsed,
        "peak_allocated_gpu_gib": peak_allocated, "peak_reserved_gpu_gib": peak_reserved,
        "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "accelerate", "numpy", "pandas")},
        "decision_rule": "argmax of two raw logits; not calibrated probabilities",
        "probe_texts": PROBES, "cpu_reload_probe_logits": reload_probe.tolist(),
        "probe_labels": [LABELS[int(value)] for value in reload_probe.argmax(axis=1)],
        "gpu_cpu_probe_labels_agree": bool(np.array_equal(cuda_probe.argmax(axis=1), reload_probe.argmax(axis=1))),
        "sha256": {path.relative_to(artifact).as_posix(): digest_file(path) for path in files},
    }
    write_json(artifact / "manifest.json", manifest)
    print(json.dumps({"artifact": str(artifact), "rows_fit": len(frame),
                      "steps": trainer.state.global_step, "training_seconds": elapsed,
                      "peak_gpu_gib": peak_allocated, "cpu_reload_verified": True}, indent=2), flush=True)
    del trainer, model, loaded_model, tokenizer, loaded_tokenizer
    gc.collect()


if __name__ == "__main__":
    main()
