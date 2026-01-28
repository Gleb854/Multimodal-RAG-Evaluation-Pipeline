# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple


def _dcg(rels: List[int]) -> float:
    # rels in ranked order, binary relevance
    s = 0.0
    for i, r in enumerate(rels, 1):
        if r:
            # log2(i+1)
            import math
            s += 1.0 / math.log2(i + 1)
    return s


def compute_retrieval_metrics(
    retrieved_chunk_ids: List[str],
    gold_chunk_ids: List[str],
) -> Dict[str, Any]:
    gold_set = set(gold_chunk_ids)
    k = len(retrieved_chunk_ids) if retrieved_chunk_ids is not None else 0

    hits = 0
    first_rank = None
    rels = []

    for i, cid in enumerate(retrieved_chunk_ids, 1):
        is_rel = 1 if cid in gold_set else 0
        rels.append(is_rel)
        if is_rel:
            hits += 1
            if first_rank is None:
                first_rank = i

    recall = hits / len(gold_set) if gold_set else 0.0
    mrr = (1.0 / first_rank) if first_rank else 0.0

    dcg = _dcg(rels)
    ideal_rels = [1] * min(len(gold_set), k) + [0] * max(0, k - min(len(gold_set), k))
    idcg = _dcg(ideal_rels)
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "k": k,
        "hits": hits,
        "gold": len(gold_set),
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "first_hit_rank": first_rank,
    }
