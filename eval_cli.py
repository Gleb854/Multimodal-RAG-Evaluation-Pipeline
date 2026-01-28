#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime

from src.config import get_settings, INDEX_DIR, EVAL_DIR
EVAL_DIR.mkdir(parents=True, exist_ok=True)


from src.rag import KnowledgeBase, Retriever, QAProcessor
from src.eval.dataset_generator import generate_dataset
from src.eval.runner import run_eval
from src.eval.report import build_reports


def setup_logger(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("eval")


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def cmd_generate(args):
    log = logging.getLogger("eval")
    settings = get_settings()

    # Determine proxy URL from args
    proxy_url = None
    if getattr(args, "proxy_url", None):
        proxy_url = args.proxy_url
    elif getattr(args, "proxy_enabled", False):
        proxy_url = settings.proxy_url_http

    log.info("Loading index: %s", args.index_name)
    log.info("Proxy: %s", proxy_url if proxy_url else "OFF")
    log.info("Embedding model: %s", settings.embedding_model)

    kb = KnowledgeBase(
        embedding_model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        index_dir=INDEX_DIR,
        proxy_url=proxy_url,
    )

    if not kb.load(args.index_name):
        raise SystemExit(f"Index '{args.index_name}' not found in {INDEX_DIR}")

    log.info("Index loaded. Documents: %d", kb.document_count)

    out_dir = EVAL_DIR / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    generator_model = args.generator_model or settings.llm_model
    log.info("Generating dataset: n=%d mix=%s k_evidence=%d", args.n, args.mix, args.k_evidence)
    log.info("Generator model: %s", generator_model)
    if args.resume:
        log.info("Resume mode: ON")

    cfg = {
        "index_name": args.index_name,
        "n": args.n,
        "seed": args.seed,
        "mix": args.mix,
        "k_evidence": args.k_evidence,
        "generator_model": generator_model,
        "include_source_hint": (not args.no_source_hint),
        "max_per_source": args.max_per_source,
        "proxy_enabled": bool(proxy_url),
    }
    (out_dir / "dataset_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_path = generate_dataset(
        kb=kb,
        index_name=args.index_name,
        out_dir=out_dir,
        n=args.n,
        seed=args.seed,
        mix_str=args.mix,
        k_evidence=args.k_evidence,
        generator_model=generator_model,
        openai_api_key=settings.openai_api_key,
        proxy_url=proxy_url,
        include_source_hint=(not args.no_source_hint),
        max_examples_per_source=args.max_per_source,
        resume=args.resume,
    )

    log.info("[OK] Dataset saved: %s", dataset_path)
    log.info(" - progress: %s", out_dir / "progress.json")
    log.info(" - errors: %s", out_dir / "errors.jsonl")


def cmd_run(args):
    log = logging.getLogger("eval")
    settings = get_settings()

    # Determine proxy URL from args
    proxy_url = None
    if getattr(args, "proxy_url", None):
        proxy_url = args.proxy_url
    elif getattr(args, "proxy_enabled", False):
        proxy_url = settings.proxy_url_http

    log.info("Loading index: %s", args.index_name)
    log.info("Proxy: %s", proxy_url if proxy_url else "OFF")
    log.info("Embedding model: %s", settings.embedding_model)

    kb = KnowledgeBase(
        embedding_model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        index_dir=INDEX_DIR,
        proxy_url=proxy_url,
    )

    if not kb.load(args.index_name):
        raise SystemExit(f"Index '{args.index_name}' not found in {INDEX_DIR}")

    log.info("Index loaded. Documents: %d", kb.document_count)

    retriever = Retriever(kb, top_k=args.k)

    llm_model = args.llm_model or settings.llm_model
    qa = QAProcessor(
        model=llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=args.llm_temperature,
        proxy_url=proxy_url,
    )

    log.info("Answer model: %s (temp=%.2f)", llm_model, args.llm_temperature)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    run_name = args.run_name or f"run_{args.index_name}_{_now_tag()}"
    out_dir = EVAL_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_model = args.judge_model or llm_model
    if not args.no_judge:
        log.info("Judge: %s (temp=%.2f)", judge_model, args.judge_temperature)
    else:
        log.info("Judge: DISABLED")

    config = {
        "index_name": args.index_name,
        "dataset_path": str(dataset_path),
        "run_name": run_name,
        "k": args.k,
        "smart_routing": (not args.no_smart_routing),
        "llm_model": llm_model,
        "llm_temperature": args.llm_temperature,
        "judge": (not args.no_judge),
        "judge_model": judge_model,
        "judge_temperature": args.judge_temperature,
        "max_examples": args.max_examples,
        "proxy_enabled": bool(proxy_url),
        "use_source_scope": args.use_source_scope,
        "strict_scope": args.strict_scope,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.use_source_scope:
        log.info("Source scope clamp: ON (strict=%s)", args.strict_scope)

    artifacts = run_eval(
        kb=kb,
        retriever=retriever,
        qa=qa,
        dataset_path=dataset_path,
        out_dir=out_dir,
        index_name=args.index_name,
        smart_routing=(not args.no_smart_routing),
        k=args.k,
        do_judge=(not args.no_judge),
        judge_model=judge_model,
        judge_temperature=args.judge_temperature,
        openai_api_key=settings.openai_api_key,
        proxy_url=proxy_url,
        max_examples=args.max_examples,
        use_source_scope=args.use_source_scope,
        strict_scope=args.strict_scope,
    )

    build_reports(out_dir=out_dir, artifacts=artifacts)

    log.info("[OK] Eval run saved: %s", out_dir)
    log.info(" - predictions: %s", out_dir / "predictions.jsonl")
    log.info(" - report: %s", out_dir / "report.md")


def main():
    parser = argparse.ArgumentParser(description="PDF Q&A System - Evaluation CLI")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    sub = parser.add_subparsers(dest="command")

    # ---------------- generate ----------------
    p_gen = sub.add_parser("generate", help="Generate eval dataset (jsonl) with gold_evidence")
    p_gen.add_argument("--index-name", required=True, help="Index name to sample evidence from")
    p_gen.add_argument("--dataset-name", default=None, help="Output folder name under data/eval/")
    p_gen.add_argument("--n", type=int, default=200, help="Number of examples")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.add_argument("--mix", default="text=0.6,image=0.25,table=0.15", help="Type mix ratios")
    p_gen.add_argument("--k-evidence", type=int, default=1, help="How many evidence chunks per question (1-3 recommended)")
    p_gen.add_argument("--generator-model", default=None, help="LLM model for dataset generation (default: settings.llm_model)")
    p_gen.add_argument("--no-source-hint", action="store_true", help="Do NOT include explicit source hint in questions")
    p_gen.add_argument("--max-per-source", type=int, default=50, help="Cap examples per source PDF (anti-bias)")
    p_gen.add_argument("--resume", action="store_true", help="Resume from last checkpoint if dataset exists")
    p_gen.add_argument("--proxy-enabled", action="store_true", help="Enable proxy for OpenAI requests")
    p_gen.add_argument("--proxy-url", default=None, help="Override proxy URL (e.g. http://user:pass@host:port)")
    p_gen.set_defaults(func=lambda a: cmd_generate(_fix_dataset_name(a)))

    # ---------------- run ----------------
    p_run = sub.add_parser("run", help="Run eval on a dataset and compute metrics + judge")
    p_run.add_argument("--index-name", required=True, help="Index name to evaluate")
    p_run.add_argument("--dataset-path", required=True, help="Path to dataset.jsonl")
    p_run.add_argument("--run-name", default=None, help="Output folder name under data/eval/")
    p_run.add_argument("--k", type=int, default=5, help="Retriever top_k")
    p_run.add_argument("--no-smart-routing", action="store_true", help="Disable smart_retrieve routing (use retrieve only)")
    p_run.add_argument("--llm-model", default=None, help="Answering model (default: settings.llm_model)")
    p_run.add_argument("--llm-temperature", type=float, default=0.1, help="Answering temperature")
    p_run.add_argument("--no-judge", action="store_true", help="Disable LLM-as-a-Judge scoring")
    p_run.add_argument("--judge-model", default=None, help="Judge model (default: llm-model)")
    p_run.add_argument("--judge-temperature", type=float, default=0.0, help="Judge temperature")
    p_run.add_argument("--max-examples", type=int, default=None, help="Limit number of examples from dataset")
    p_run.add_argument("--proxy-enabled", action="store_true", help="Enable proxy for OpenAI requests")
    p_run.add_argument("--proxy-url", default=None, help="Override proxy URL (e.g. http://user:pass@host:port)")
    p_run.add_argument("--use-source-scope", action="store_true",
                       help="Clamp retrieved docs to example['source_scope'] (diagnostic mode)")
    p_run.add_argument("--strict-scope", action="store_true",
                       help="If scope filtering yields 0 docs, do NOT fallback to unclamped docs")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    log = setup_logger(getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        return

    args.func(args)


def _fix_dataset_name(args):
    # default dataset name if not provided
    if args.dataset_name is None:
        args.dataset_name = f"ds_{args.index_name}_{_now_tag()}"
    return args


if __name__ == "__main__":
    main()
