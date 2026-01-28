# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
from typing import Dict, Any


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index_signature(index_dir: Path) -> Dict[str, Any]:
    index_dir = Path(index_dir)
    files = ["index.faiss", "index.pkl", "kb_meta.pkl"]
    out = {"dir": str(index_dir), "files": {}}
    for fn in files:
        p = index_dir / fn
        if p.exists():
            out["files"][fn] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
        else:
            out["files"][fn] = {"missing": True}
    return out
