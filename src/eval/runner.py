# -*- coding: utf-8 -*-
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from .io import read_jsonl, write_jsonl
from .metrics import compute_retrieval_metrics
from .index_signature import build_index_signature
from .judge import LLMJudge

log = logging.getLogger("eval")


def _extract_retrieved_chunk_ids(context_docs) -> List[str]:
    ids = []
    for d in context_docs or []:
        md = d.metadata or {}
        cid = md.get("chunk_id")
        if cid:
            ids.append(cid)
    return ids


def _extract_gold_chunk_ids(example: Dict[str, Any]) -> List[str]:
    gold = []
    for ev in example.get("gold_evidence") or []:
        cid = ev.get("chunk_id")
        if cid:
            gold.append(cid)
    return gold


def _light_doc_view(doc) -> Dict[str, Any]:
    md = doc.metadata or {}
    txt = (doc.page_content or "")
    return {
        "chunk_id": md.get("chunk_id"),
        "source": md.get("source"),
        "page": md.get("page"),
        "doc_type": md.get("doc_type", md.get("element_type")),
        "figure_number": md.get("figure_number"),
        "table_number": md.get("table_number"),
        "image_index": md.get("image_index"),
        "preview": (txt[:200] + "...") if len(txt) > 200 else txt,
    }


def _filter_docs_by_scope(docs, scope: str):
    """Filter documents to only those matching the given source scope."""
    if not scope:
        return docs
    scoped = []
    for d in docs:
        md = getattr(d, "metadata", None) or {}
        if md.get("source") == scope:
            scoped.append(d)
    return scoped


def run_eval(
    kb,
    retriever,
    qa,
    dataset_path: Path,
    out_dir: Path,
    index_name: str,
    smart_routing: bool = True,
    k: int = 5,
    do_judge: bool = True,
    judge_model: Optional[str] = None,
    judge_temperature: float = 0.0,
    openai_api_key: Optional[str] = None,
    proxy_url: Optional[str] = None,
    max_examples: Optional[int] = None,
    use_source_scope: bool = False,
    strict_scope: bool = False,
) -> Dict[str, Any]:
    t_all = time.perf_counter()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = read_jsonl(dataset_path)
    if max_examples is not None:
        dataset = dataset[: max_examples]

    log.info("Loaded dataset: %d examples", len(dataset))
    log.info("Writing to: %s", str(out_dir))

    # index signature (determinism)
    idx_dir = Path(kb.index_dir) / index_name
    log.info("Building index signature: %s", str(idx_dir))
    signature = build_index_signature(idx_dir)
    (out_dir / "index_signature.json").write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")

    judge = None
    if do_judge:
        judge = LLMJudge(
            model=judge_model,
            openai_api_key=openai_api_key,
            proxy_url=proxy_url,
            temperature=judge_temperature,
        )

    pred_rows = []
    judge_rows = []

    for ex in dataset:
        i = len(pred_rows) + 1
        t0 = time.perf_counter()

        q = ex["question"]

        # Scope clamp setup
        scope = ""
        if use_source_scope:
            scope = (ex.get("source_scope") or "").strip()

        # Wrapper that applies scope filtering after retrieval
        class _ScopedRetriever:
            def __init__(self, base, scope_val: str, strict: bool):
                self.base = base
                self.scope_val = scope_val
                self.strict = strict

            def _apply_scope(self, docs):
                if not self.scope_val:
                    return docs
                docs_before = len(docs)
                scoped = _filter_docs_by_scope(docs, self.scope_val)
                if scoped:
                    docs = scoped
                elif self.strict:
                    docs = []
                # else: fallback to original docs
                if self.scope_val:
                    log.debug("Scope clamp: scope='%s' docs=%d -> %d (strict=%s)",
                              self.scope_val, docs_before, len(docs), self.strict)
                return docs

            def smart_retrieve(self, query, k=5):
                docs = self.base.smart_retrieve(query, k=k)
                return self._apply_scope(docs)

            def retrieve(self, query, k=5, filter_by_type=None):
                docs = self.base.retrieve(query, k=k, filter_by_type=filter_by_type)
                return self._apply_scope(docs)

            def format_context(self, docs, include_metadata=True):
                return self.base.format_context(docs, include_metadata=include_metadata)

            def infer_source_scope(self, query):
                return self.base.infer_source_scope(query)

        # choose routing
        if smart_routing:
            scoped_retriever = _ScopedRetriever(retriever, scope, strict_scope)
            result = qa.process_with_retrieval(q, scoped_retriever, k=k)
        else:
            # shim retriever: force plain text retrieve + scope filtering
            class _Plain:
                def __init__(self, base, scope_val: str, strict: bool):
                    self.base = base
                    self.scope_val = scope_val
                    self.strict = strict

                def _apply_scope(self, docs):
                    if not self.scope_val:
                        return docs
                    docs_before = len(docs)
                    scoped = _filter_docs_by_scope(docs, self.scope_val)
                    if scoped:
                        docs = scoped
                    elif self.strict:
                        docs = []
                    if self.scope_val:
                        log.debug("Scope clamp (plain): scope='%s' docs=%d -> %d (strict=%s)",
                                  self.scope_val, docs_before, len(docs), self.strict)
                    return docs

                def smart_retrieve(self, query, k=5):
                    docs = self.base.retrieve(query, k=k, filter_by_type="text")
                    return self._apply_scope(docs)

                def format_context(self, docs, include_metadata=True):
                    return self.base.format_context(docs, include_metadata=include_metadata)

                def infer_source_scope(self, query):
                    return self.base.infer_source_scope(query)

            result = qa.process_with_retrieval(q, _Plain(retriever, scope, strict_scope), k=k)

        t_retr = time.perf_counter()

        gold_ids = _extract_gold_chunk_ids(ex)
        retrieved_ids = _extract_retrieved_chunk_ids(result.context_docs)
        m = compute_retrieval_metrics(retrieved_ids, gold_ids)

        log.debug("Example %d/%d | retrieved=%d | gold=%d", i, len(dataset), len(retrieved_ids), len(gold_ids))

        pred = {
            "id": ex["id"],
            "index_name": index_name,
            "question": q,
            "question_type": ex.get("question_type"),
            "answer": result.answer,
            "sources": result.sources,
            "retrieval_metrics": m,
            "gold_evidence": ex.get("gold_evidence"),
            "retrieved": [_light_doc_view(d) for d in (result.context_docs or [])],
            "context_used": result.context_used,
        }
        pred_rows.append(pred)

        t_j = time.perf_counter()
        if judge is not None:
            j = judge.score(question=q, answer=result.answer, context=result.context_used)
            judge_rows.append({"id": ex["id"], **j})
            t_j = time.perf_counter()

        t1 = time.perf_counter()

        # Progress every 10 examples
        if i % 10 == 0 or i == 1 or i == len(dataset):
            dt = t1 - t0
            dt_retr = t_retr - t0
            dt_ans = (t_j - t_retr) if do_judge else (t1 - t_retr)
            dt_j = (t1 - t_j) if do_judge else 0.0

            log.info(
                "Progress %d/%d | total=%.2fs (retr=%.2f, ans=%.2f, judge=%.2f)",
                i, len(dataset), dt, dt_retr, dt_ans, dt_j
            )

    # persist
    log.info("Saving predictions.jsonl (%d rows)", len(pred_rows))
    write_jsonl(out_dir / "predictions.jsonl", pred_rows)
    if judge_rows:
        log.info("Saving judge.jsonl (%d rows)", len(judge_rows))
        write_jsonl(out_dir / "judge.jsonl", judge_rows)

    log.info("Done. Total time: %.2fs", time.perf_counter() - t_all)

    return {
        "dataset_path": str(dataset_path),
        "predictions_path": str(out_dir / "predictions.jsonl"),
        "judge_path": str(out_dir / "judge.jsonl") if judge_rows else None,
        "index_signature_path": str(out_dir / "index_signature.json"),
        "n": len(pred_rows),
    }
