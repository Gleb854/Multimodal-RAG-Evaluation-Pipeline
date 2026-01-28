# src/eval/dataset_generator.py
# -*- coding: utf-8 -*-
import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document as LCDocument

log = logging.getLogger("eval")

# -------------------- helpers --------------------

def _make_http_client(proxy_url: Optional[str]):
    if not proxy_url:
        return None
    try:
        return httpx.Client(proxy=proxy_url, timeout=httpx.Timeout(60.0))
    except TypeError:
        return httpx.Client(proxies=proxy_url, timeout=httpx.Timeout(60.0))


def _parse_mix(mix_str: str) -> List[Tuple[str, float]]:
    parts = []
    for seg in (mix_str or "").split(","):
        seg = seg.strip()
        if not seg:
            continue
        k, v = seg.split("=", 1)
        parts.append((k.strip(), float(v.strip())))
    s = sum(v for _, v in parts) or 1.0
    return [(k, v / s) for k, v in parts]


def _choice_weighted(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    x = rng.random()
    cum = 0.0
    for k, w in items:
        cum += w
        if x <= cum:
            return k
    return items[-1][0]


def _doc_to_evidence(doc: LCDocument) -> Dict[str, Any]:
    md = doc.metadata or {}
    return {
        "chunk_id": md.get("chunk_id"),
        "source": md.get("source"),
        "page": md.get("page"),
        "doc_type": md.get("doc_type", md.get("element_type")),
        "figure_number": md.get("figure_number"),
        "table_number": md.get("table_number"),
        "image_index": md.get("image_index"),
    }


def _compact_context(docs: List[LCDocument], max_chars: int = 3500) -> str:
    parts = []
    total = 0
    for d in docs:
        md = d.metadata or {}
        header = f"SOURCE={md.get('source','?')} PAGE={md.get('page','?')} TYPE={md.get('doc_type', md.get('element_type','text'))}"
        if md.get("figure_number"):
            header += f" FIG={md.get('figure_number')}"
        if md.get("table_number"):
            header += f" TABLE={md.get('table_number')}"
        if md.get("image_index") is not None:
            header += f" IMG_IDX={md.get('image_index')}"
        txt = (d.page_content or "").strip()
        snippet = txt[:1200]
        block = header + "\n" + snippet
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def _config_hash(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_progress(path: Path, progress: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_progress(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# -------------------- question quality gates --------------------

_BAD_MENTIONS_RE = re.compile(
    r"(?i)\b(in paper|according to (the )?paper|paper\s+\S+|arxiv|\.pdf\b|v\d+\b|SOURCE=)\b"
)

_TOO_GENERIC_RE = re.compile(
    r"(?i)\b(what information is provided|describe (the )?(figure|image|diagram)|summarize|overview of)\b"
)

def _is_human_like_question(q: str) -> bool:
    if not q:
        return False
    q = q.strip()
    if len(q) < 12:
        return False
    if _BAD_MENTIONS_RE.search(q):
        return False
    if _TOO_GENERIC_RE.search(q):
        return False
    # people usually ask a question with "?" (allow rare missing but penalize)
    if "?" not in q and len(q) < 60:
        return False
    return True


# -------------------- prompts --------------------

_SYSTEM = (
    "You generate evaluation questions for a PDF QA system.\n"
    "You will be given EVIDENCE (one or more chunks) extracted from a document.\n"
    "Create ONE human-like question that can be answered using ONLY this evidence.\n"
    "\n"
    "Critical rules:\n"
    "- The question MUST be grounded in the evidence. No hallucinations.\n"
    "- DO NOT mention filenames, arXiv IDs, 'the paper', 'in paper ...', or SOURCE_HINT in the question.\n"
    "- Avoid generic prompts like 'What information is provided...'. Ask for a concrete fact.\n"
    "\n"
    "If evidence is IMAGE:\n"
    "- Ask about a specific element: axis label, trend, which component connects to which, what a legend indicates.\n"
    "\n"
    "If evidence is TABLE:\n"
    "- Ask about a specific cell/value/row/column.\n"
    "\n"
    "Output STRICT JSON ONLY with keys: question, question_type.\n"
    "question_type must be one of: text, image, table, mixed.\n"
)

_USER_TMPL = (
    "EVIDENCE:\n"
    "{context}\n\n"
    "SOURCE_HINT (for grounding only; DO NOT mention it in the question): {source_hint}\n\n"
    "Return JSON only."
)


# -------------------- main API --------------------

def generate_dataset(
    kb,
    index_name: str,
    out_dir: Path,
    n: int = 200,
    seed: int = 42,
    mix_str: str = "text=0.6,image=0.25,table=0.15",
    k_evidence: int = 1,
    generator_model: str = "gpt-4",
    openai_api_key: Optional[str] = None,
    proxy_url: Optional[str] = None,
    include_source_hint: bool = True,
    max_examples_per_source: int = 50,
    resume: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "dataset.jsonl"
    errors_path = out_dir / "errors.jsonl"
    progress_path = out_dir / "progress.json"

    cfg = {
        "index_name": index_name,
        "n": n,
        "seed": seed,
        "mix": mix_str,
        "k_evidence": k_evidence,
        "generator_model": generator_model,
        "include_source_hint": include_source_hint,
        "max_examples_per_source": max_examples_per_source,
    }
    cfg_hash = _config_hash(cfg)

    n_done = 0
    per_source_counter: Dict[str, int] = {}

    if resume:
        existing_progress = _read_progress(progress_path)
        if existing_progress and existing_progress.get("config_hash") == cfg_hash:
            n_done = existing_progress.get("n_done", 0)
            per_source_counter = existing_progress.get("per_source_counter", {})
            log.info("Resuming from %d/%d examples", n_done, n)

            actual_lines = _count_jsonl_lines(dataset_path)
            if actual_lines != n_done:
                log.warning("Line count mismatch: progress=%d file=%d; using file count", n_done, actual_lines)
                n_done = actual_lines
        elif existing_progress:
            log.warning("Config hash mismatch. Starting fresh.")
            n_done = 0
            dataset_path.unlink(missing_ok=True)
            errors_path.unlink(missing_ok=True)

    t_all = time.perf_counter()
    started_at = datetime.now().isoformat()

    dataset_path.touch(exist_ok=True)
    errors_path.touch(exist_ok=True)

    _write_progress(progress_path, {
        "n_target": n,
        "n_done": n_done,
        "n_failed": 0,
        "n_skipped_no_chunk_id": 0,
        "n_skipped_source_limit": 0,
        "n_skipped_no_candidates": 0,
        "n_skipped_bad_question": 0,
        "tries": 0,
        "started_at": started_at,
        "last_updated": datetime.now().isoformat(),
        "seed": seed,
        "config_hash": cfg_hash,
        "per_source_counter": per_source_counter,
        "elapsed_sec": 0.0,
        "status": "running",
    })

    rng = random.Random(seed)
    mix = _parse_mix(mix_str)
    for _ in range(n_done * 10):
        rng.random()

    http_client = _make_http_client(proxy_url)
    llm = ChatOpenAI(
        model=generator_model,
        openai_api_key=openai_api_key,
        temperature=0.0,
        max_tokens=400,
        http_client=http_client,
    )

    n_ok = n_done
    n_failed = 0
    n_skipped_no_chunk_id = 0
    n_skipped_source_limit = 0
    n_skipped_no_candidates = 0
    n_skipped_bad_question = 0
    tries = 0
    max_tries = n * 25  # чуть больше, т.к. добавили фильтрации

    log.info("Dataset generation: n=%d seed=%d mix=%s k_evidence=%d", n, seed, mix_str, k_evidence)
    log.info("Generator model: %s | source_hint=%s | max_per_source=%d", generator_model, include_source_hint, max_examples_per_source)
    log.info("Output dir: %s", out_dir)

    while n_ok < n and tries < max_tries:
        tries += 1
        target_type = _choice_weighted(rng, mix)

        candidates = kb.sample_documents(
            n=1,
            filter_dict={"doc_type": target_type},
            seed=rng.randint(1, 10**9),
        )
        if not candidates:
            n_skipped_no_candidates += 1
            continue

        base_doc = candidates[0]
        src = (base_doc.metadata or {}).get("source") or "Unknown"

        if per_source_counter.get(src, 0) >= max_examples_per_source:
            n_skipped_source_limit += 1
            continue

        evidence_docs = [base_doc]
        if k_evidence > 1:
            same_source_docs = kb.get_documents_by_source(src) or []
            same_type = [d for d in same_source_docs if (d.metadata or {}).get("doc_type") == target_type]
            fallback_text = [d for d in same_source_docs if (d.metadata or {}).get("doc_type") == "text"]
            pool = same_type if len(same_type) >= k_evidence else (same_type + fallback_text)
            rng.shuffle(pool)
            for d in pool:
                if len(evidence_docs) >= k_evidence:
                    break
                cid = (d.metadata or {}).get("chunk_id")
                if cid and any((x.metadata or {}).get("chunk_id") == cid for x in evidence_docs):
                    continue
                evidence_docs.append(d)

        doc_types = {(d.metadata or {}).get("doc_type") for d in evidence_docs}
        if len(doc_types) == 1:
            qtype = list(doc_types)[0] if list(doc_types)[0] in {"text", "image", "table"} else "text"
        else:
            qtype = "mixed"

        context = _compact_context(evidence_docs)
        source_hint = src if include_source_hint else ""

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=_USER_TMPL.format(context=context, source_hint=source_hint)),
        ]

        try:
            resp = llm.invoke(messages)
            raw = (resp.content or "").strip()
            data = json.loads(raw)

            question = (data.get("question") or "").strip()
            question_type = (data.get("question_type") or qtype).strip().lower()

            if question_type not in {"text", "image", "table", "mixed"}:
                question_type = qtype

            if not _is_human_like_question(question):
                n_skipped_bad_question += 1
                continue

        except Exception as e:
            n_failed += 1
            _append_jsonl(errors_path, {
                "try": tries,
                "target_type": target_type,
                "source": src,
                "error": f"{type(e).__name__}: {e}",
                "timestamp": datetime.now().isoformat(),
            })
            continue

        row = {
            "id": str(uuid.uuid4()),
            "index_name": index_name,
            "question": question,
            "question_type": question_type,
            "source_scope": (src if include_source_hint and src != "Unknown" else None),
            "gold_evidence": [_doc_to_evidence(d) for d in evidence_docs],
        }

        if any(not ev.get("chunk_id") for ev in row["gold_evidence"]):
            n_skipped_no_chunk_id += 1
            continue

        _append_jsonl(dataset_path, row)
        n_ok += 1
        per_source_counter[src] = per_source_counter.get(src, 0) + 1

        _write_progress(progress_path, {
            "n_target": n,
            "n_done": n_ok,
            "n_failed": n_failed,
            "n_skipped_no_chunk_id": n_skipped_no_chunk_id,
            "n_skipped_source_limit": n_skipped_source_limit,
            "n_skipped_no_candidates": n_skipped_no_candidates,
            "n_skipped_bad_question": n_skipped_bad_question,
            "tries": tries,
            "started_at": started_at,
            "last_updated": datetime.now().isoformat(),
            "seed": seed,
            "config_hash": cfg_hash,
            "per_source_counter": per_source_counter,
            "elapsed_sec": round(time.perf_counter() - t_all, 2),
            "status": "running",
        })

        if n_ok % 10 == 0 or n_ok == 1 or n_ok == n:
            log.info(
                "Generated %d/%d (tries=%d, failed=%d, bad_q=%d, skip_chunk=%d)",
                n_ok, n, tries, n_failed, n_skipped_bad_question, n_skipped_no_chunk_id
            )

    elapsed = time.perf_counter() - t_all
    log.info(
        "Dataset generation done: %d examples in %.2fs (failed=%d, bad_q=%d)",
        n_ok, elapsed, n_failed, n_skipped_bad_question
    )

    _write_progress(progress_path, {
        "n_target": n,
        "n_done": n_ok,
        "n_failed": n_failed,
        "n_skipped_no_chunk_id": n_skipped_no_chunk_id,
        "n_skipped_source_limit": n_skipped_source_limit,
        "n_skipped_no_candidates": n_skipped_no_candidates,
        "n_skipped_bad_question": n_skipped_bad_question,
        "tries": tries,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "seed": seed,
        "config_hash": cfg_hash,
        "per_source_counter": per_source_counter,
        "elapsed_sec": round(elapsed, 2),
        "status": "completed" if n_ok >= n else "partial",
    })

    return dataset_path
