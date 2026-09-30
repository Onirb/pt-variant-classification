"""Read-only publication check; reports file/line/category, never secret values."""

import json
import re
import subprocess
from pathlib import Path


PATTERNS = {
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".ipynb", ".tsv", ".csv", ".yaml", ".yml", ".toml"}


def secret_locations(text):
    findings = []
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"kind": kind, "line": text.count("\n", 0, match.start()) + 1})
    return findings


def forbidden_new_path(name):
    path = Path(name)
    if name.startswith(("runs/", "data/external/", "data/track_b/", ".venv/", ".hf_cache/", ".aws/", ".secrets/")):
        return True
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return True
    if path.name.lower().startswith("credentials") and path.suffix == ".json":
        return True
    return path.suffix.lower() in {".pem", ".key", ".pth", ".pt", ".safetensors", ".onnx", ".joblib", ".zip", ".parquet"}


def git_paths(*args):
    data = subprocess.check_output(["git", "ls-files", "-z", *args])
    return {value.decode("utf-8") for value in data.split(b"\0") if value}


def main():
    tracked = git_paths("--cached")
    untracked = git_paths("--others", "--exclude-standard")
    added = set(subprocess.check_output(["git", "diff", "--cached", "--name-only", "--diff-filter=A"]).decode().splitlines())
    findings = []
    scanned = 0
    for name in sorted(tracked | untracked):
        path = Path(name)
        if (name in untracked or name in added) and forbidden_new_path(name):
            findings.append({"file": name, "kind": "forbidden_new_artifact_or_local_config"})
        if not path.is_file() or (path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore"):
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        findings.extend({"file": name, **finding} for finding in secret_locations(text))
    result = {"files_scanned": scanned, "findings": findings,
              "scope": "tracked + nonignored new files; pattern-based check, not a guarantee against every secret"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
