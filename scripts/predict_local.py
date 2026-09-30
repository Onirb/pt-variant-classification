"""CLI local para o classificador final pt-BR vs pt-PT."""

from __future__ import annotations

import argparse
import json

from src.inference import LocalVariantClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifica um texto como pt-BR ou pt-PT localmente.")
    parser.add_argument("text", help="Texto a classificar. Use aspas se houver espaços.")
    args = parser.parse_args()

    result = LocalVariantClassifier().predict(args.text)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print("Observação: margin é uma margem de decisão, não uma probabilidade calibrada.")


if __name__ == "__main__":
    main()
