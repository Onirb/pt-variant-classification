"""Treina Char-CNN apenas na validação interna de desenvolvimento."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import ID_TO_LABEL, load_corpus
from src.experiments import development_split
from src.neural import CharacterCNN, CharacterDataset, CharacterTokenizer, set_seed


def evaluate(model: CharacterCNN, loader: DataLoader, device: torch.device) -> tuple[float, float, dict[str, float]]:
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for tokens, labels in loader:
            logits = model(tokens.to(device))
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    _, _, f1_per_class, _ = precision_recall_fscore_support(actual, predicted, labels=[0, 1], zero_division=0)
    return (
        float(accuracy_score(actual, predicted)),
        float(f1_score(actual, predicted, average="macro")),
        {ID_TO_LABEL[label]: float(f1_per_class[index]) for index, label in enumerate((0, 1))},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina uma Char-CNN em validação interna.")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6)
    arguments = parser.parse_args()
    seed = 42
    patience = 3
    set_seed(seed)
    torch.set_num_threads(arguments.threads)
    device = torch.device("cpu")

    corpus, _, summary = load_corpus(cache_dir=os.environ.get("HF_HOME"))
    train, validation = development_split(corpus, seed=seed)
    tokenizer = CharacterTokenizer.fit(train["text"], max_length=arguments.max_length)
    train_dataset = CharacterDataset(train["text"], train["label"], tokenizer)
    validation_dataset = CharacterDataset(validation["text"], validation["label"], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True, collate_fn=train_dataset.collate)
    validation_loader = DataLoader(validation_dataset, batch_size=arguments.batch_size, shuffle=False, collate_fn=validation_dataset.collate)

    model = CharacterCNN(
        len(tokenizer.vocabulary), embedding_dim=arguments.embedding_dim, channels=arguments.channels
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    output = Path("runs/char_cnn_v1")
    output.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    stalled = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        losses: list[float] = []
        for tokens, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(tokens.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        accuracy, macro_f1, per_class_f1 = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "accuracy": accuracy, "macro_f1": macro_f1, "per_class_f1": per_class_f1}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            stalled = 0
            torch.save(model.state_dict(), output / "model_state.pt")
        else:
            stalled += 1
            if stalled >= patience:
                break

    (output / "tokenizer.json").write_text(json.dumps(tokenizer.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "model": "CharacterCNN kernels=(3,4,5)",
        "device": str(device),
        "seed": seed,
        "max_length": arguments.max_length,
        "embedding_dim": arguments.embedding_dim,
        "channels": arguments.channels,
        "batch_size": arguments.batch_size,
        "threads": arguments.threads,
        "corpus": summary.to_dict(),
        "best_validation_macro_f1": best_f1,
        "history": history,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_validation_macro_f1": best_f1, "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
