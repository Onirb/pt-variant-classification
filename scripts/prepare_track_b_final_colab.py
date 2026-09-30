"""Empacota seleção final, notebook e desenvolvimento congelado para o Colab."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.prepare_track_b_colab import notebook_cell


SOURCE = Path("runs/track_b/colab_b1_4_bundle_v1/track_b_colab")
OUTPUT = Path("runs/track_b/colab_final_bundle_v1")
RESULTS = Path("runs/track_b/b1_4_colab_import_v1")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def selection_from_results():
    reports = [json.loads((RESULTS / "full" / f"seed_{seed}" / "report.json").read_text(encoding="utf-8"))
               for seed in (7, 42, 2026)]
    best_epochs = []
    evidence = []
    for report in reports:
        best_step = int(Path(report["best_model_checkpoint"]).name.rsplit("-", 1)[1])
        steps_per_epoch = report["global_steps"] / report["training_metrics"]["epoch"]
        epoch_fraction = best_step / steps_per_epoch
        best_epochs.append(epoch_fraction)
        evidence.append({"seed": report["seed"], "best_step": best_step,
                         "steps_per_epoch": steps_per_epoch, "best_epoch_fraction": epoch_fraction})
    median_epochs = statistics.median(best_epochs)
    steps = math.ceil(math.ceil(1877 / 16) * median_epochs)
    return {
        "schema_version": "1.0", "selected_candidate": "B1.4",
        "model": "neuralmind/bert-base-portuguese-cased",
        "revision": "94d69c95f98f7d5b2a8700c420230ae10def0baa",
        "selection_basis": "highest mean development macro F1 and higher mean F1 in all four domains versus B1.3",
        "remaining_limit": "recall gap grew to 0.140704; reduction-of-asymmetry goal not satisfied",
        "epoch_budget_rule": "median best epoch fraction from the three development runs, rescaled to full corpus",
        "evidence": evidence, "median_epoch_fraction": median_epochs,
        "final_protocol": {
            "seed": 42, "seed_rule": "fixed reference seed, not selected by seed F1",
            "rows": 1877, "max_steps": steps, "batch_size": 8,
            "gradient_accumulation_steps": 2, "world_size": 1, "max_length": 256,
            "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.1,
            "precision": "fp16", "decision_rule": "argmax; no threshold calibration",
        },
        "final_test_policy": "DSL-TL test reserved until local artifact verification; no FRMT reuse",
    }


def notebook():
    cells = [
        notebook_cell("markdown", """# Track B — ajuste final no Colab

Este notebook executa **um único ajuste** do BERTimbau original em todos os
1.877 textos de desenvolvimento. Duração congelada: 318 passos, aproximadamente
2,69 épocas; semente 42. Ele não repete os três experimentos anteriores.

Selecione GPU (T4, se disponível) e execute as células em ordem.
Envie **track_b_final_package_v1.zip** na primeira célula.
O modelo salvo inclui pesos e tokenizer; o ZIP final terá aproximadamente
400–450 MB. Checkpoints continuam em uma nova pasta do Drive.
"""),
        notebook_cell("code", """from pathlib import Path
import hashlib, io, json, subprocess, sys, zipfile
from google.colab import files

uploaded = files.upload()
names = [name for name in uploaded if name.lower().endswith('.zip')]
if len(names) != 1:
    raise ValueError('Envie somente o ZIP do ajuste final nesta célula.')
extract_root = Path('/content/pt_variant_final')
extract_root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(uploaded[names[0]])) as archive:
    for item in archive.infolist():
        path = (extract_root / item.filename).resolve()
        if not path.is_relative_to(extract_root.resolve()):
            raise ValueError('Caminho inválido no ZIP.')
    archive.extractall(extract_root)
PACKAGE_ROOT = extract_root / 'track_b_final'
manifest = json.loads((PACKAGE_ROOT / 'package_manifest.json').read_text())
for name, expected in manifest['sha256'].items():
    if hashlib.sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Arquivo alterado: {name}')
selection = json.loads((PACKAGE_ROOT / 'selection.json').read_text())
print('Corpus:', manifest['rows'], 'textos; passos finais:', selection['final_protocol']['max_steps'])
"""),
        notebook_cell("markdown", """## Preparar GPU e salvar no Drive
Se a sessão anterior já está aberta, pode usar a mesma conta e GPU.
O treino salva em **MyDrive/pt-variant-track-b/final_colab_v1**.
As pastas dos experimentos anteriores continuam disponíveis.
"""),
        notebook_cell("code", """subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                '-r', str(PACKAGE_ROOT / 'requirements-colab.txt')], check=True)
subprocess.run([sys.executable, '-c',
    "import torch; assert torch.cuda.is_available(), 'Selecione GPU no Colab'; "
    "print('GPU:', torch.cuda.get_device_name(0))"], check=True)
from google.colab import drive
drive.mount('/content/drive')
OUTPUT_ROOT = Path('/content/drive/MyDrive/pt-variant-track-b/final_colab_v1')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
"""),
        notebook_cell("markdown", """## Treinar uma vez e validar a exportação
A duração foi escolhida apenas a partir dos checkpoints de desenvolvimento.
O script recomeça do encoder original com cabeça nova e treina todos os textos.
Se houver interrupção, repita esta célula: o último checkpoint será retomado.
Se já terminou, a célula apenas verifica o artefato salvo.
"""),
        notebook_cell("code", """command = [sys.executable, str(PACKAGE_ROOT / 'train_track_b_final.py'),
           '--package-root', str(PACKAGE_ROOT), '--output-dir', str(OUTPUT_ROOT)]
subprocess.run(command, check=True)
subprocess.run(command + ['--verify-only'], check=True)
final_manifest = json.loads((OUTPUT_ROOT / 'artifact/manifest.json').read_text())
print('Treino final concluído:', final_manifest['rows_fit'], 'textos')
print('Passos:', final_manifest['global_steps'])
print('Tempo de treino:', round(final_manifest['training_seconds_this_session'], 1), 'segundos')
print('Inferência CPU após recarregar: verificada')
"""),
        notebook_cell("markdown", """## Exportar modelo completo e recibo pequeno
O ZIP grande contém apenas o artefato final: pesos, tokenizer, seleção e manifesto.
Ele não inclui dados de treino nem otimizadores. O recibo pequeno ajuda a verificar
a conclusão antes de transferir o modelo. Baixe os dois arquivos para Downloads.
"""),
        notebook_cell("code", """artifact = OUTPUT_ROOT / 'artifact'
archive_path = Path('/content/track_b_final_model_v1.zip')
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(artifact.rglob('*')):
        if path.is_file():
            archive.write(path, arcname='track_b_final_v1/' + path.relative_to(artifact).as_posix())
receipt_path = Path('/content/track_b_final_receipt_v1.json')
receipt = {'manifest': final_manifest,
           'artifact_zip_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
           'artifact_zip_bytes': archive_path.stat().st_size}
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
print('Modelo completo:', round(archive_path.stat().st_size / 1024**2, 1), 'MiB')
files.download(str(receipt_path))
files.download(str(archive_path))
"""),
    ]
    return {"cells": cells, "metadata": {"colab": {"name": "Track_B_Final_Colab.ipynb"},
            "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Pacote final já existe: {OUTPUT}")
    source_manifest = json.loads((SOURCE / "package_manifest.json").read_text())
    for name, expected in source_manifest["sha256"].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pacote original alterado: {name}")
    selection = selection_from_results()
    package = OUTPUT / "track_b_final"
    package.mkdir(parents=True)
    for name in source_manifest["sha256"]:
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((SOURCE / name).read_bytes())
    (package / "train_track_b_final.py").write_bytes(Path("colab/train_track_b_final.py").read_bytes())
    write_json(package / "selection.json", selection)
    files = sorted(path for path in package.rglob("*") if path.is_file())
    package_manifest = {"schema_version": "1.0", "rows": 1877, "seeds": [7, 42, 2026],
                        "scope": "final fit on public development only",
                        "original_data_sha256": source_manifest["original_data_sha256"],
                        "sha256": {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    write_json(package / "package_manifest.json", package_manifest)
    zip_path = OUTPUT / "track_b_final_package_v1.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(OUTPUT).as_posix())
    notebook_path = Path("colab/Track_B_Final_Colab.ipynb")
    write_json(notebook_path, notebook())
    print(json.dumps({"notebook": str(notebook_path), "package": str(zip_path),
                      "bytes": zip_path.stat().st_size, "selection": selection}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
