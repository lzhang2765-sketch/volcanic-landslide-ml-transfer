from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".tsv", ".csv", ".cff"}


def public_text() -> str:
    chunks: list[str] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def test_no_local_user_paths_or_cloud_roots() -> None:
    text = public_text()
    assert "/" + "Users" + "/" not in text
    assert "Library" + "/CloudStorage" not in text
    assert "/content/drive/" + "MyDrive" not in text


def test_no_obvious_private_keys_or_tokens() -> None:
    text = public_text()
    patterns = [
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"ghp_[A-Za-z0-9]{30,}",
        r"github_pat_[A-Za-z0-9_]{30,}",
        r"AKIA[0-9A-Z]{16}",
        r"(?i)api[_-]?key\s*[:=]\s*['\"][^'\"]+",
        r"(?i)password\s*[:=]\s*['\"][^'\"]+",
    ]
    for pattern in patterns:
        assert re.search(pattern, text) is None, pattern
