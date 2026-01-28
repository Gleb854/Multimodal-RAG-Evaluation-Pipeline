# -*- coding: utf-8 -*-
import argparse
import json
import random
from pathlib import Path

from src.config import get_settings
from src.eval.io import read_jsonl
from src.eval.judge import LLMJudge


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", choices=["normal", "empty_context", "shuffled_context"], default="normal")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--proxy-enabled", action="store_true")
    ap.add_argument("--proxy-url", default=None)
    args = ap.parse_args()

    settings = get_settings()
    proxy_url = args.proxy_url or (settings.proxy_url_http if args.proxy_enabled else None)

    rows = read_jsonl(Path(args.predictions))
    rows = rows[: args.n]

    judge = LLMJudge(
        model=args.judge_model or settings.llm_model,
        openai_api_key=settings.openai_api_key,
        proxy_url=proxy_url,
        temperature=args.temp,
    )

    # prepare contexts
    contexts = [r.get("context_used") or "" for r in rows]
    if args.mode == "empty_context":
        contexts = ["" for _ in contexts]
    elif args.mode == "shuffled_context":
        rnd = random.Random(42)
        rnd.shuffle(contexts)

    scores = {"faithfulness": [], "answer_relevance": [], "citation_support": [], "refusal_correctness": [], "overall": []}

    for r, ctx in zip(rows, contexts):
        q = r["question"]
        a = r["answer"]
        j = judge.score(question=q, answer=a, context=ctx)
        for k in scores:
            if k in j and isinstance(j[k], (int, float)):
                scores[k].append(float(j[k]))

    out = {k + "_mean": mean(v) for k, v in scores.items() if v}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
