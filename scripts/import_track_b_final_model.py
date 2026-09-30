"""Confere recibo e pesos finais antes de registrar o modelo no projeto local."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from src.fine_tuned_inference import DEFAULT_ARTIFACT, FineTunedVariantClassifier, sha256_file


SELECTION = Path("runs/track_b/colab_final_bundle_v1/track_b_final/selection.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("receipt_path", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt_path.read_text(encoding="utf-8"))
    if sha256_file(args.zip_path) != receipt["artifact_zip_sha256"]:
        raise ValueError("Hash do ZIP não corresponde ao recibo.")
    if args.zip_path.stat().st_size != receipt["artifact_zip_bytes"]:
        raise ValueError("Tamanho do ZIP não corresponde ao recibo.")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if DEFAULT_ARTIFACT.exists():
        classifier = FineTunedVariantClassifier(DEFAULT_ARTIFACT)
        if classifier.manifest != receipt["manifest"]:
            raise FileExistsError("Modelo final existente difere do recibo; não será sobrescrito.")
        print(json.dumps(classifier.verify_probes(), indent=2))
        return
    staging = DEFAULT_ARTIFACT.with_name(DEFAULT_ARTIFACT.name + ".pending")
    if staging.exists():
        raise FileExistsError(f"Importação pendente já existe: {staging}. Revisar antes de repetir.")
    with ZipFile(args.zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP corrompido.")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contém entradas duplicadas.")
        manifest = json.loads(archive.read("track_b_final_v1/manifest.json"))
        if manifest != receipt["manifest"]:
            raise ValueError("Manifesto do ZIP difere do recibo.")
        if manifest["final_protocol"] != selection["final_protocol"]:
            raise ValueError("Protocolo diferente da seleção final congelada.")
        if manifest["model"] != selection["model"] or manifest["revision"] != selection["revision"]:
            raise ValueError("Encoder ou revisão incompatível com a seleção final.")
        if manifest["rows_fit"] != 1877 or manifest["global_steps"] != 318:
            raise ValueError("Ajuste final incompleto ou corpus incompatível.")
        expected_data_hash = json.loads((SELECTION.parent / "package_manifest.json").read_text())["sha256"]["data/development.parquet"]
        if manifest["development_data_sha256"] != expected_data_hash:
            raise ValueError("Hash do corpus de desenvolvimento difere do pacote final.")
        staged_selection = json.loads(archive.read("track_b_final_v1/selection.json"))
        if staged_selection != selection:
            raise ValueError("Regra de seleção foi alterada após congelamento.")
        allowed = set(manifest["sha256"]) | {"manifest.json"}
        entries = [(name, name.removeprefix("track_b_final_v1/")) for name in names]
        if any(not name.startswith("track_b_final_v1/") or relative not in allowed for name, relative in entries):
            raise ValueError("ZIP contém arquivo não declarado no manifesto.")
        for name, relative in entries:
            path = (staging / relative).resolve()
            if not path.is_relative_to(staging.resolve()):
                raise ValueError("Caminho inseguro no artefato.")
        staging.mkdir(parents=True)
        for name, relative in entries:
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, path.open("wb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
    classifier = FineTunedVariantClassifier(staging)
    verification = classifier.verify_probes()
    del classifier
    staging.rename(DEFAULT_ARTIFACT)
    print(json.dumps({"artifact": str(DEFAULT_ARTIFACT), **verification}, indent=2))


if __name__ == "__main__":
    main()
