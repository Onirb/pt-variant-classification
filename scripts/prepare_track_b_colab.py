"""Constrói notebook e pacote Colab por uma lista explícita de arquivos."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from src.track_b import TRACK_B_PREPARED_PATH, TRACK_B_PREPARATION_MANIFEST_PATH
from src.track_b_experiments import TRACK_B_SEEDS, document_split


OUTPUT = Path("runs/track_b/colab_b1_4_bundle_v1")


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def notebook_cell(kind: str, text: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(text)
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebook() -> dict:
    cells = [
        notebook_cell("markdown", """# Track B — B1.4 no Google Colab

Abra **Ambiente de execução → Alterar tipo de ambiente de execução → GPU** (T4, se disponível).
Execute as células em ordem. Quando solicitado, envie **track_b_colab_package_v1.zip**.
O notebook faz um teste de 10 passos; o treino completo começa somente na célula própria.
As sementes 7, 42 e 2026 usam exatamente as divisões congeladas dos baselines.
O teste final DSL-TL não faz parte deste pacote.

O Drive é opcional, mas recomendado para sobreviver ao encerramento da sessão.
Se usado, o Colab solicitará sua autorização. Reserve espaço para pesos e checkpoints
(aproximadamente 10 GB para as três sementes, dependendo dos artefatos).
Os escores do teste curto não serão usados para selecionar o modelo.
"""),
        notebook_cell("code", """from pathlib import Path
import hashlib, io, json, subprocess, sys, zipfile
from google.colab import files

uploaded = files.upload()
zip_names = [filename for filename in uploaded if filename.lower().endswith('.zip')]
if len(zip_names) != 1:
    raise ValueError('Envie apenas o ZIP do pacote nesta célula.')
name = zip_names[0]
EXTRACT_ROOT = Path('/content/pt_variant_b1_4')
EXTRACT_ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(uploaded[name])) as archive:
    for item in archive.infolist():
        destination = (EXTRACT_ROOT / item.filename).resolve()
        if not destination.is_relative_to(EXTRACT_ROOT.resolve()):
            raise ValueError('Caminho inválido dentro do pacote.')
    archive.extractall(EXTRACT_ROOT)
PACKAGE_ROOT = EXTRACT_ROOT / 'track_b_colab'
manifest = json.loads((PACKAGE_ROOT / 'package_manifest.json').read_text())
for filename, expected in manifest['sha256'].items():
    actual = hashlib.sha256((PACKAGE_ROOT / filename).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Hash divergente: {filename}')
print('Pacote validado:', manifest['rows'], 'textos; sementes', manifest['seeds'])
"""),
        notebook_cell("markdown", """## Dependências
O ambiente manterá o PyTorch/CUDA que o Colab fornece. O treino roda em um processo
Python novo, para usar as versões instaladas aqui. O BERTimbau público será baixado
na primeira execução; nenhuma chave de API é necessária.
"""),
        notebook_cell("code", """subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    '-r', str(PACKAGE_ROOT / 'requirements-colab.txt')
], check=True)
subprocess.run([sys.executable, '-c',
    "import torch; assert torch.cuda.is_available(), 'GPU ausente: selecione GPU e reconecte'; "
    "print('GPU:', torch.cuda.get_device_name(0)); "
    "print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"
], check=True)
"""),
        notebook_cell("markdown", """## Persistência
Mantenha USE_DRIVE=True para salvar checkpoints no Drive. A pasta é exclusiva deste
experimento. Se uma sessão cair, reabra este notebook, selecione GPU, execute as
células anteriores e repita a célula de treino: as sementes concluídas serão
reconhecidas e a semente incompleta retomará o último checkpoint.
Com USE_DRIVE=False, os arquivos em /content podem desaparecer quando a sessão terminar.
"""),
        notebook_cell("code", """USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_ROOT = Path('/content/drive/MyDrive/pt-variant-track-b/b1_4_colab_v1')
else:
    OUTPUT_ROOT = Path('/content/b1_4_colab_v1')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print('Resultados e checkpoints:', OUTPUT_ROOT)
"""),
        notebook_cell("markdown", """## Teste curto: tempo e memória
Execute primeiro. Ele testa download do modelo, treinamento FP16, validação e gravação
de checkpoint. É separado do experimento completo. A estimativa usa a taxa observada,
mas escrita no Drive e interrupções podem alterar o tempo final.
"""),
        notebook_cell("code", """def run_training(mode, seed=None):
    command = [sys.executable, str(PACKAGE_ROOT / 'train_b1_4.py'),
               '--package-root', str(PACKAGE_ROOT), '--output-dir', str(OUTPUT_ROOT),
               '--mode', mode]
    if seed is not None:
        command += ['--seed', str(seed)]
    subprocess.run(command, check=True)

run_training('smoke')
smoke = json.loads((OUTPUT_ROOT / 'smoke/seed_42/report.json').read_text())
print('GPU:', smoke['gpu'])
print('Pico de memória GPU:', round(smoke['peak_allocated_gpu_gib'], 2), 'GiB')
seconds_per_step = smoke['training_metrics']['train_runtime'] / smoke['global_steps']
rough_minutes = seconds_per_step * smoke['planned_full_steps_per_seed'] * 3 / 60
print('Estimativa inicial das três sementes:', round(rough_minutes, 1), 'minutos')
print('Teste curto concluído. A célula abaixo inicia o experimento completo.')
"""),
        notebook_cell("markdown", """## Treinamento completo
Esta célula executa as três sementes em sequência. São três épocas por semente,
learning rate 2e-5, batch 8 com acumulação de 2, máximo de 256 tokens.
São avaliados e salvos checkpoints a cada 50 passos; o melhor F1 macro de validação
define o checkpoint escolhido dentro de cada execução. Isto não é treino final
para publicação: o resultado será comparado aos outros baselines locais.
Para rodar apenas uma semente, use por exemplo run_training('full', seed=7).
"""),
        notebook_cell("code", """run_training('full')
aggregate_path = OUTPUT_ROOT / 'full/aggregate.json'
if aggregate_path.exists():
    aggregate = json.loads(aggregate_path.read_text())
    print('F1 macro médio:', round(aggregate['macro_f1_mean'], 6))
    print('Desvio entre sementes:', round(aggregate['macro_f1_std'], 6))
"""),
        notebook_cell("markdown", """## Baixar resultados para análise aqui
A exportação pequena contém relatórios, predições e configurações; não inclui
checkpoints pesados. Pesos e checkpoints continuam na pasta de resultados.
Você pode baixar esses modelos pelo Drive depois de compararmos os resultados.
"""),
        notebook_cell("code", """results_zip = Path('/content/track_b_b1_4_results_v1.zip')
with zipfile.ZipFile(results_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT_ROOT.rglob('*')):
        if path.is_file() and path.name in {'report.json', 'predictions.parquet', 'run_config.json', 'aggregate.json'}:
            archive.write(path, arcname=path.relative_to(OUTPUT_ROOT))
files.download(str(results_zip))
print('Guarde o ZIP e copie-o depois para a pasta runs/track_b do projeto local.')
"""),
    ]
    return {
        "cells": cells,
        "metadata": {"colab": {"name": "Track_B_B1_4_Colab.ipynb"}, "accelerator": "GPU",
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Pacote já existe: {OUTPUT}; preserve-o e versione uma nova preparação.")
    original = json.loads(TRACK_B_PREPARATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if hashlib.sha256(TRACK_B_PREPARED_PATH.read_bytes()).hexdigest() != original["data_sha256"]:
        raise ValueError("Dados preparados divergiram do manifesto congelado.")
    frame = pd.read_parquet(TRACK_B_PREPARED_PATH)
    # Lista explícita de campos: não incluir source_en ou original_target na nuvem.
    frame = frame[["text", "label", "domain", "document_id", "segment_id", "lp"]].reset_index(drop=True)
    splits = {}
    for seed in TRACK_B_SEEDS:
        split = document_split(frame, seed=seed)
        validation_docs = set(split.validation.document_id)
        validation_mask = frame.document_id.isin(validation_docs)
        baseline = pd.read_parquet("runs/track_b/b1_1_char_word_svm_v2/predictions.parquet")
        baseline = baseline.loc[baseline.seed.eq(seed)]
        expected = set(zip(baseline.document_id, baseline.segment_id, baseline.label))
        actual = set(zip(frame.loc[validation_mask, "document_id"], frame.loc[validation_mask, "segment_id"], frame.loc[validation_mask, "label"]))
        if expected != actual:
            raise AssertionError(f"Divisão seed {seed} diverge das predições B1.1.")
        splits[str(seed)] = {
            "train_indices": frame.index[~validation_mask].tolist(),
            "validation_indices": frame.index[validation_mask].tolist(),
            "validation_documents": sorted(validation_docs),
        }
    package = OUTPUT / "track_b_colab"
    (package / "data").mkdir(parents=True)
    frame.to_parquet(package / "data/development.parquet", index=False)
    write_json(package / "data/splits.json", splits)
    write_json(package / "data/preparation_manifest.json", original)
    for source, name in [(Path("colab/train_b1_4.py"), "train_b1_4.py"),
                         (Path("colab/requirements-colab.txt"), "requirements-colab.txt")]:
        (package / name).write_bytes(source.read_bytes())
    files = sorted(path for path in package.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "1.0", "scope": "public WMT24++ development only; no external final test",
        "rows": len(frame), "seeds": list(TRACK_B_SEEDS),
        "original_data_sha256": original["data_sha256"],
        "split_source": "exact memberships verified against saved B1.1 v2 predictions",
        "sha256": {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    write_json(package / "package_manifest.json", manifest)
    archive_path = OUTPUT / "track_b_colab_package_v1.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(OUTPUT).as_posix())
    notebook_path = Path("colab/Track_B_B1_4_Colab.ipynb")
    write_json(notebook_path, build_notebook())
    print(json.dumps({"notebook": str(notebook_path), "package": str(archive_path),
                      "package_bytes": archive_path.stat().st_size, "rows": len(frame),
                      "seeds": manifest["seeds"]}, indent=2))


if __name__ == "__main__":
    main()
