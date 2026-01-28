# -*- coding: utf-8 -*-
import json
from typing import Dict, Any, Optional

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def _make_http_client(proxy_url: Optional[str]):
    if not proxy_url:
        return None
    try:
        return httpx.Client(proxy=proxy_url, timeout=httpx.Timeout(60.0))
    except TypeError:
        return httpx.Client(proxies=proxy_url, timeout=httpx.Timeout(60.0))


_JUDGE_SYSTEM = (
    "You are a strict evaluator for a Retrieval-Augmented Generation (RAG) QA system.\n"
    "You will receive: QUESTION, ANSWER, CONTEXT (with [Source N] blocks).\n"
    "Evaluate whether the answer is grounded ONLY in the context and properly cited.\n"
    "Return STRICT JSON only with keys:\n"
    "faithfulness, answer_relevance, citation_support, refusal_correctness, overall, notes.\n"
    "Scores are floats in [0,1].\n"
    "refusal_correctness: if the answer correctly says info is not found when context lacks it.\n"
    "overall: your holistic score.\n"
    "notes: short reason.\n"
)

_JUDGE_USER_TMPL = (
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "CONTEXT:\n{context}\n\n"
    "Return JSON only."
)


class LLMJudge:
    def __init__(
        self,
        model: str,
        openai_api_key: str,
        proxy_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ):
        http_client = _make_http_client(proxy_url)
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            http_client=http_client,
        )

    def score(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        messages = [
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=_JUDGE_USER_TMPL.format(question=question, answer=answer, context=context)),
        ]
        try:
            resp = self.llm.invoke(messages)
            raw = (resp.content or "").strip()
            data = json.loads(raw)
        except Exception as e:
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "citation_support": 0.0,
                "refusal_correctness": 0.0,
                "overall": 0.0,
                "notes": f"judge_parse_error: {type(e).__name__}",
            }

        # sanitize
        out = {}
        for k in ["faithfulness", "answer_relevance", "citation_support", "refusal_correctness", "overall"]:
            v = data.get(k, 0.0)
            try:
                v = float(v)
            except Exception:
                v = 0.0
            out[k] = max(0.0, min(1.0, v))

        out["notes"] = (data.get("notes") or "").strip()[:500]
        return out
