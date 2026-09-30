"""Recuperação do teste oficial e auditoria lexical, sem inferência ou ajuste."""

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


REVISION = "44a083029be0c2fa7f304323908e808215c8eee1"
DATA_DIR = Path("data/external/dsl_tl_test_v1")
TEST_PATH = DATA_DIR / "pt_test_recovered.parquet"
AUDIT_DIR = Path("runs/track_b/b4_pre_inference_audit_v1")
FINAL_DIR = Path("runs/track_b/b4_dsl_tl_test_v1")
LABELS = {0: "PT-PT", 1: "PT-BR"}
LEXICAL_RULES = {
    "normalization": "Unicode NFKC + casefold + whitespace collapse; preserve accents",
    "exact": "normalized equality",
    "character": "binary char 5-gram cosine >= 0.90; both texts >= 80 normalized characters",
    "word": "binary word 5-gram Jaccard >= 0.80; both have >= 6 unique shingles",
    "caveat": "lexical heuristics, not semantic/paraphrase detection or proof of no pretraining contamination",
}


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def fingerprint(frame):
    payload = [{"id": str(row.id), "text": str(row.text)} for row in frame.itertuples()]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def parse_tsv(path, fields):
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
    if any(len(row) != fields for row in rows):
        raise ValueError(f"Formato TSV inválido: {path}")
    return rows


def recover_test(root, *, expected_portuguese_rows=495):
    features = parse_tsv(root / "PT_withFeatures.tsv", 4)
    if features[0] != ["Id", "Instance", "Markers", "New.Gold.Label"]:
        raise ValueError("Cabeçalho das anotações incompatível.")
    gold = {}
    for identifier, text, markers, label in features[1:]:
        if identifier in gold or label not in {"PT", "PT-PT", "PT-BR"}:
            raise ValueError("ID repetido ou rótulo inválido nas anotações.")
        gold[identifier] = (text, label)
    known = parse_tsv(root / "PT_train.tsv", 3) + parse_tsv(root / "PT_dev.tsv", 3)
    known_ids = set()
    for identifier, text, label in known:
        if identifier in known_ids or gold.get(identifier) != (text, label):
            raise ValueError("Treino/dev não corresponde aos IDs, textos e rótulos oficiais.")
        known_ids.add(identifier)
    remaining = {identifier: row for identifier, row in gold.items() if identifier not in known_ids}
    if len(remaining) != expected_portuguese_rows:
        raise ValueError("Quantidade recuperada difere do protocolo.")
    by_text = defaultdict(list)
    for identifier, (text, label) in remaining.items():
        by_text[text].append((identifier, label))
    tests = (root / "DSL-TL-test.tsv").read_text(encoding="utf-8").splitlines()
    if len(tests) != len(set(tests)):
        raise ValueError("Teste oficial contém textos duplicados; revisar mapeamento.")
    rows = []
    for line_number, text in enumerate(tests, 1):
        candidates = by_text.get(text, [])
        if len(candidates) > 1:
            raise ValueError("Mais de um ID de teste para o mesmo texto.")
        if candidates:
            identifier, raw_label = candidates[0]
            rows.append({"id": identifier, "official_test_line": line_number,
                         "text": text, "raw_label": raw_label, "domain": "journalistic",
                         "label_nature": "human annotated; New.Gold.Label"})
    if len(rows) != len(remaining) or set(row["id"] for row in rows) != set(remaining):
        raise ValueError("Teste não cobre exatamente todas as anotações portuguesas restantes.")
    frame = pd.DataFrame(rows)
    frame["label"] = frame.raw_label.map({"PT-PT": 0, "PT-BR": 1}).astype("Int64")
    return frame, {"portuguese_rows": len(frame), "known_train_dev_ids": len(known_ids),
                   "exact_id_text_label_consistency": True, "test_ids_disjoint_from_train_dev": True,
                   "label_counts": {str(key): int(value) for key, value in frame.raw_label.value_counts().items()},
                   "recovery": "IDs absent from official train/dev; exact raw sentence match to multilingual official test"}


def lexical_overlap(query, references):
    """Todos os pares são examinados por matrizes esparsas, em blocos de 32."""
    qtexts = [normalize(value) for value in query.text]
    rtexts = [normalize(value) for value in references.text]
    if not qtexts or not rtexts or any(not value for value in qtexts + rtexts):
        raise ValueError("Auditoria exige textos não vazios.")
    qids = query.id.astype(str).tolist()
    records = references.to_dict("records")
    exact_index = defaultdict(list)
    for index, text in enumerate(rtexts):
        exact_index[text].append(index)
    hits = []
    for qi, text in enumerate(qtexts):
        for ri in exact_index.get(text, []):
            hits.append({"test_id": qids[qi], "reference_source": records[ri]["source"],
                         "reference_id": str(records[ri]["id"]), "method": "exact_normalized", "similarity": 1.0})
    texts = qtexts + rtexts
    modes = (
        ("char5_cosine", TfidfVectorizer(analyzer="char", ngram_range=(5, 5),
            lowercase=False, use_idf=False, binary=True, dtype=np.float32)),
        ("word5_jaccard", CountVectorizer(analyzer="word", ngram_range=(5, 5),
            token_pattern=r"(?u)\b\w+\b", lowercase=False, binary=True, dtype=np.int32)),
    )
    for method, vectorizer in modes:
        try:
            matrix = vectorizer.fit_transform(texts).tocsr()
        except ValueError as error:
            if "empty vocabulary" in str(error):
                continue
            raise
        qmatrix, rmatrix = matrix[:len(qtexts)], matrix[len(qtexts):]
        qcounts = np.asarray(qmatrix.sum(axis=1)).ravel()
        rcounts = np.asarray(rmatrix.sum(axis=1)).ravel()
        for start in range(0, len(qtexts), 32):
            product = (qmatrix[start:start + 32] @ rmatrix.T).tocoo()
            qis = product.row + start
            ris = product.col
            if method == "word5_jaccard":
                similarities = product.data / (qcounts[qis] + rcounts[ris] - product.data)
                eligible = (similarities >= 0.80) & (qcounts[qis] >= 6) & (rcounts[ris] >= 6)
            else:
                similarities = product.data
                eligible = (similarities >= 0.90) & (np.array([len(qtexts[qi]) for qi in qis]) >= 80) & \
                           (np.array([len(rtexts[ri]) for ri in ris]) >= 80)
            for qi, ri, similarity in zip(qis[eligible], ris[eligible], similarities[eligible]):
                # Igualdade já foi registrada; não duplica o mesmo par em três métodos.
                if qtexts[qi] == rtexts[ri]:
                    continue
                hits.append({"test_id": qids[qi], "reference_source": records[ri]["source"],
                             "reference_id": str(records[ri]["id"]), "method": method,
                             "similarity": float(similarity)})
        del matrix, qmatrix, rmatrix
    return hits


def metrics(actual, predicted):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    if not len(actual):
        return {"rows": 0, "status": "empty cohort"}
    per_class = classification_report(actual, predicted, labels=[0, 1],
                                     target_names=[LABELS[0], LABELS[1]], output_dict=True, zero_division=0)
    return {"rows": len(actual), "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "per_class": {label: per_class[label] for label in LABELS.values()},
            "recall_gap_br_minus_pt": per_class["PT-BR"]["recall"] - per_class["PT-PT"]["recall"],
            "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(actual, predicted, labels=[0, 1]).tolist()}
