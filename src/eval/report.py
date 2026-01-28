# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .io import read_jsonl


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _group_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = r.get(key) or "unknown"
        out.setdefault(k, []).append(r)
    return out


def build_reports(out_dir: Path, artifacts: Dict[str, Any]) -> None:
    out_dir = Path(out_dir)

    preds = read_jsonl(out_dir / "predictions.jsonl")
    judge_path = out_dir / "judge.jsonl"
    judges = read_jsonl(judge_path) if judge_path.exists() else []
    judge_by_id = {j["id"]: j for j in judges}

    # aggregate retrieval metrics
    recalls = [p["retrieval_metrics"]["recall"] for p in preds]
    mrrs = [p["retrieval_metrics"]["mrr"] for p in preds]
    ndcgs = [p["retrieval_metrics"]["ndcg"] for p in preds]

    # judge aggregates
    faith = []
    rel = []
    cite = []
    refc = []
    overall = []
    for p in preds:
        j = judge_by_id.get(p["id"])
        if not j:
            continue
        faith.append(j.get("faithfulness", 0.0))
        rel.append(j.get("answer_relevance", 0.0))
        cite.append(j.get("citation_support", 0.0))
        refc.append(j.get("refusal_correctness", 0.0))
        overall.append(j.get("overall", 0.0))

    report = {
        "n": len(preds),
        "retrieval": {
            "recall_mean": _mean(recalls),
            "mrr_mean": _mean(mrrs),
            "ndcg_mean": _mean(ndcgs),
        },
        "judge": {
            "enabled": bool(judges),
            "faithfulness_mean": _mean(faith),
            "answer_relevance_mean": _mean(rel),
            "citation_support_mean": _mean(cite),
            "refusal_correctness_mean": _mean(refc),
            "overall_mean": _mean(overall),
        },
        "by_type": {},
    }

    # breakdown by question_type
    by_type = _group_by(preds, "question_type")
    for t, rows in by_type.items():
        r_rec = [x["retrieval_metrics"]["recall"] for x in rows]
        r_mrr = [x["retrieval_metrics"]["mrr"] for x in rows]
        r_ndcg = [x["retrieval_metrics"]["ndcg"] for x in rows]

        j_faith = []
        j_rel = []
        j_cite = []
        j_over = []
        for x in rows:
            j = judge_by_id.get(x["id"])
            if not j:
                continue
            j_faith.append(j.get("faithfulness", 0.0))
            j_rel.append(j.get("answer_relevance", 0.0))
            j_cite.append(j.get("citation_support", 0.0))
            j_over.append(j.get("overall", 0.0))

        report["by_type"][t] = {
            "n": len(rows),
            "retrieval": {"recall_mean": _mean(r_rec), "mrr_mean": _mean(r_mrr), "ndcg_mean": _mean(r_ndcg)},
            "judge": {
                "faithfulness_mean": _mean(j_faith),
                "answer_relevance_mean": _mean(j_rel),
                "citation_support_mean": _mean(j_cite),
                "overall_mean": _mean(j_over),
            },
        }

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown summary
    md = []
    md.append(f"# Evaluation Report\n")
    md.append(f"- N: {report['n']}\n")
    md.append("## Retrieval\n")
    md.append(f"- Recall@K (mean): {report['retrieval']['recall_mean']:.4f}\n")
    md.append(f"- MRR@K (mean): {report['retrieval']['mrr_mean']:.4f}\n")
    md.append(f"- nDCG@K (mean): {report['retrieval']['ndcg_mean']:.4f}\n")

    if report["judge"]["enabled"]:
        md.append("\n## LLM-as-a-Judge\n")
        md.append(f"- Faithfulness (mean): {report['judge']['faithfulness_mean']:.4f}\n")
        md.append(f"- Answer relevance (mean): {report['judge']['answer_relevance_mean']:.4f}\n")
        md.append(f"- Citation support (mean): {report['judge']['citation_support_mean']:.4f}\n")
        md.append(f"- Refusal correctness (mean): {report['judge']['refusal_correctness_mean']:.4f}\n")
        md.append(f"- Overall (mean): {report['judge']['overall_mean']:.4f}\n")

    md.append("\n## Breakdown by question_type\n")
    for t, block in report["by_type"].items():
        md.append(f"\n### {t}\n")
        md.append(f"- N: {block['n']}\n")
        md.append(f"- Recall@K: {block['retrieval']['recall_mean']:.4f}\n")
        md.append(f"- MRR@K: {block['retrieval']['mrr_mean']:.4f}\n")
        md.append(f"- nDCG@K: {block['retrieval']['ndcg_mean']:.4f}\n")
        if report["judge"]["enabled"]:
            md.append(f"- Faithfulness: {block['judge']['faithfulness_mean']:.4f}\n")
            md.append(f"- Relevance: {block['judge']['answer_relevance_mean']:.4f}\n")
            md.append(f"- Citations: {block['judge']['citation_support_mean']:.4f}\n")
            md.append(f"- Overall: {block['judge']['overall_mean']:.4f}\n")

    (out_dir / "report.md").write_text("".join(md), encoding="utf-8")
