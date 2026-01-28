# src/rag/retriever.py
from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Any
import re

from langchain_core.documents import Document as LCDocument

from .knowledge_base import KnowledgeBase


class Retriever:
    """
    Retriever that supports:
    - normal semantic retrieval
    - "smart" routing for Table/Figure-like queries
    - robust retrieval for rare doc types (table/image) via increased fetch_k and/or fallbacks
    - dedupe + stable context formatting
    """

    _RE_TABLE = re.compile(r"\b(table)\s*([0-9]+|[IVXLCDM]+)\b", re.IGNORECASE)
    _RE_FIG = re.compile(r"\b(fig\.|figure)\s*([0-9]+|[IVXLCDM]+)\b", re.IGNORECASE)

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        # How much we over-fetch from FAISS when we later filter by metadata.
        # Needed because FAISS itself doesn't filter by metadata.
        fetch_k_multiplier: int = 50,
        min_fetch_k: int = 200,
        # When query is table/figure-like, we try to mix types.
        table_mix_ratio: float = 0.6,
        fig_mix_ratio: float = 0.6,
    ):
        self.kb = knowledge_base
        self.top_k = top_k
        self.score_threshold = score_threshold

        self.fetch_k_multiplier = fetch_k_multiplier
        self.min_fetch_k = min_fetch_k

        self.table_mix_ratio = table_mix_ratio
        self.fig_mix_ratio = fig_mix_ratio

    # -----------------------------
    # Public API
    # -----------------------------
    def infer_source_scope(self, query: str) -> Optional[str]:
        """
        Infer which specific source (PDF filename) the query is about.
        Used by QAProcessor to scope context to a single document.
        
        Returns exact source filename if confident, None otherwise.
        """
        hint = self._extract_source_hint(query)
        if not hint:
            return None
        
        # Find matching source in index
        sources = self.kb.get_unique_sources()
        for src in sources:
            if hint in src.lower():
                return src
        
        return None

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_by_type: Optional[str] = None,
    ) -> List[LCDocument]:
        k = k or self.top_k

        filter_dict = {"doc_type": filter_by_type} if filter_by_type else None

        # Important: when filter is used, we must fetch much more from FAISS
        # and then filter in Python. Otherwise rare types ("table") will never appear in top-k.
        fetch_k = None
        if filter_dict:
            fetch_k = max(self.min_fetch_k, k * self.fetch_k_multiplier)

        return self.kb.similarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict,
            score_threshold=self.score_threshold,
            fetch_k=fetch_k,
        )

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter_by_type: Optional[str] = None,
    ) -> List[Tuple[LCDocument, float]]:
        """
        Returns list of (doc, score). Score is whatever FAISS returns (distance),
        lower is typically "more similar" in LangChain FAISS store.
        """
        k = k or self.top_k

        filter_dict = {"doc_type": filter_by_type} if filter_by_type else None
        fetch_k = None
        if filter_dict:
            fetch_k = max(self.min_fetch_k, k * self.fetch_k_multiplier)

        return self.kb.similarity_search_with_scores(
            query=query,
            k=k,
            filter_dict=filter_dict,
            score_threshold=self.score_threshold,
            fetch_k=fetch_k,
        )

    def retrieve_by_type(
        self,
        query: str,
        doc_types: Optional[List[str]] = None,
        k_per_type: Optional[int] = None,
    ) -> Dict[str, List[LCDocument]]:
        doc_types = doc_types or ["text", "table", "image"]
        k_per_type = k_per_type or self.top_k

        out: Dict[str, List[LCDocument]] = {}
        for dt in doc_types:
            out[dt] = self.retrieve(query, k=k_per_type, filter_by_type=dt)
        return out

    def smart_retrieve(self, query: str, k: Optional[int] = None) -> List[LCDocument]:
        """
        Heuristic routing:
        - If query mentions Table -> prioritize tables + some text
        - If query mentions Fig/Figure/Image -> prioritize images + some text
        - Else -> normal text-heavy retrieval
        """
        k = k or self.top_k
        q = (query or "").strip()
        q_lower = q.lower()

        is_table_like = self._looks_like_table_query(q_lower)
        is_fig_like = self._looks_like_figure_query(q_lower)

        if is_table_like:
            kt = max(1, int(round(k * self.table_mix_ratio)))
            kx = max(0, k - kt)

            # Extract table number and source hint for strict matching
            tbl_num = self._extract_table_number(q)
            src_hint = self._extract_source_hint(q)

            # Build filter for strict table retrieval
            filter_dict = {"doc_type": "table"}
            if src_hint:
                filter_dict["source__contains"] = src_hint

            # Over-fetch tables with source filtering
            fetch_k = max(self.min_fetch_k, kt * self.fetch_k_multiplier)
            raw_tables = self.kb.similarity_search(
                query=q,
                k=max(kt * 10, 30),
                filter_dict=filter_dict,
                fetch_k=fetch_k,
            )

            # If table number specified, filter strictly by table_number first
            if tbl_num:
                exact_matches = [
                    d for d in raw_tables
                    if str(d.metadata.get("table_number", "")).strip().upper() == tbl_num
                ]
                if exact_matches:
                    raw_tables = exact_matches + [d for d in raw_tables if d not in exact_matches]

            table_docs = raw_tables[:kt]

            # Fallback 1: if source filter was too strict, try without it
            if not table_docs and src_hint:
                raw_tables_no_src = self.retrieve(q, k=max(kt * 5, 20), filter_by_type="table")
                if tbl_num:
                    exact_matches = [
                        d for d in raw_tables_no_src
                        if str(d.metadata.get("table_number", "")).strip().upper() == tbl_num
                    ]
                    if exact_matches:
                        raw_tables_no_src = exact_matches + [d for d in raw_tables_no_src if d not in exact_matches]
                table_docs = raw_tables_no_src[:kt]

            text_docs = self.retrieve(q, k=kx, filter_by_type="text") if kx else []

            merged = self._merge_and_dedupe(table_docs + text_docs, k)

            # Fallback 2: if tables not found, use more text
            if not any(d.metadata.get("doc_type") == "table" for d in merged):
                merged = self._merge_and_dedupe(
                    table_docs + self.retrieve(q, k=min(20, k * 4), filter_by_type="text"),
                    k,
                )

            # Boost exact table matches if present
            merged = self._boost_exact_table_matches(q, merged)
            return merged[:k]

        if is_fig_like:
            ki = max(1, int(round(k * self.fig_mix_ratio)))
            kx = max(0, k - ki)

            # Extract figure number and source hint for strict matching
            fig_num = self._extract_figure_number(q)
            src_hint = self._extract_source_hint(q)

            # Build filter for strict image retrieval
            filter_dict = {"doc_type": "image"}
            if src_hint:
                filter_dict["source__contains"] = src_hint

            # Over-fetch images with source filtering
            fetch_k = max(self.min_fetch_k, ki * self.fetch_k_multiplier)
            raw_images = self.kb.similarity_search(
                query=q,
                k=max(ki * 10, 50),
                filter_dict=filter_dict,
                fetch_k=fetch_k,
            )

            # If figure number specified, filter strictly by figure_number first
            if fig_num:
                exact_matches = [
                    d for d in raw_images
                    if str(d.metadata.get("figure_number", "")).strip().upper() == fig_num
                ]
                if exact_matches:
                    raw_images = exact_matches + [d for d in raw_images if d not in exact_matches]

            # Apply quality filter
            image_docs = self._filter_image_docs(raw_images, query=q)[:ki]

            # Fallback 1: if source filter was too strict, try without it
            if not image_docs and src_hint:
                raw_images_no_src = self.retrieve(q, k=max(ki * 5, 20), filter_by_type="image")
                if fig_num:
                    exact_matches = [
                        d for d in raw_images_no_src
                        if str(d.metadata.get("figure_number", "")).strip().upper() == fig_num
                    ]
                    if exact_matches:
                        raw_images_no_src = exact_matches + [d for d in raw_images_no_src if d not in exact_matches]
                image_docs = self._filter_image_docs(raw_images_no_src, query=q)[:ki]

            text_docs = self.retrieve(q, k=kx, filter_by_type="text") if kx else []

            merged = self._merge_and_dedupe(image_docs + text_docs, k)

            # Fallback 2: if after filtering we got no images, relax filter
            if not any(d.metadata.get("doc_type") == "image" for d in merged):
                image_docs_relaxed = self.retrieve(q, k=max(ki * 5, 20), filter_by_type="image")[:ki]
                merged = self._merge_and_dedupe(image_docs_relaxed + text_docs, k)

            # Fallback 3: if still no images, use more text
            if not any(d.metadata.get("doc_type") == "image" for d in merged):
                merged = self._merge_and_dedupe(
                    self.retrieve(q, k=min(20, k * 4), filter_by_type="text"),
                    k,
                )

            # Boost exact figure matches if present
            merged = self._boost_exact_figure_matches(q, merged)
            return merged[:k]

        # Default: keep it simple (text first)
        # You can still get tables/images here depending on semantics, but text dominates.
        return self.retrieve(q, k=k)

    def format_context(self, documents: List[LCDocument], include_metadata: bool = True) -> str:
        """
        Stable formatting: match your QAProcessor prompt citations [Source N].
        """
        parts: List[str] = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            doc_type = doc.metadata.get("doc_type", "text")
            section = doc.metadata.get("section", "")

            header = f"[Source {i}: {source}"
            if page not in (None, "?", ""):
                header += f", Page {page}"
            if section:
                header += f", Section: {section}"
            header += f", Type: {doc_type}]"

            if include_metadata:
                parts.append(f"{header}\n{doc.page_content}")
            else:
                parts.append(doc.page_content)

        return "\n\n---\n\n".join(parts)

    # -----------------------------
    # Internals
    # -----------------------------
    _BAD_VISION_DECISIONS = {"error"}
    _BAD_SKIP_REASONS = {"too_small", "probable_cover_first_page", "low_value_caption"}

    def _truthy(self, v: Any, default: bool = True) -> bool:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        return default

    def _norm_source(self, s: Any) -> str:
        return str(s or "").strip().lower()

    def _extract_source_hint(self, q: str) -> Optional[str]:
        """
        Try to infer which PDF the question is about.
        Supports arXiv-like ids present in filenames: 1112.1158, 1202.0452, etc.
        Also supports keywords like "wireless", "game-theoretic", "smart grid paper".
        """
        # arXiv id
        m = re.search(r"\b(\d{4}\.\d{4,5})\b", q)
        if m:
            return m.group(1).lower()
        
        # Keyword hints for common papers
        q_lower = q.lower()
        if any(w in q_lower for w in ["wireless", "wireless communication", "wireless smart grid"]):
            return "1112.1158"
        if any(w in q_lower for w in ["game theoretic", "game-theoretic", "game theory"]):
            return "1202.0452"
        
        return None

    def _extract_figure_number(self, q: str) -> Optional[str]:
        """Extract figure number from query like 'Fig. 1', 'Figure 2'."""
        m = self._RE_FIG.search(q)
        if m:
            return (m.group(2) or "").strip().upper()
        return None

    def _extract_table_number(self, q: str) -> Optional[str]:
        """Extract table number from query like 'Table I', 'Table 2'."""
        m = self._RE_TABLE.search(q)
        if m:
            return (m.group(2) or "").strip().upper()
        return None

    def _filter_image_docs(self, docs: List[LCDocument], query: str) -> List[LCDocument]:
        """
        Hard filter for figure-like quality:
        - keep figure_candidate=True when present
        - drop vision_decision=error
        - drop known skip reasons (covers/logos/tiny)
        - optionally narrow by source hint (arxiv id) if present in query
        """
        src_hint = self._extract_source_hint(query)
        out: List[LCDocument] = []

        for d in docs:
            md = d.metadata or {}
            if md.get("doc_type") != "image":
                continue

            # Source narrowing (if we can infer it)
            if src_hint:
                src = self._norm_source(md.get("source") or md.get("source_pdf"))
                if src_hint not in src:
                    continue

            # Figure candidate gating (default True for backward compatibility)
            if not self._truthy(md.get("figure_candidate"), default=True):
                continue

            # Vision decision gating
            vd = str(md.get("vision_decision") or "").strip().lower()
            if vd in self._BAD_VISION_DECISIONS:
                continue

            # Skip reasons gating
            sr = str(md.get("vision_skip_reason") or "").strip().lower()
            if any(bad in sr for bad in self._BAD_SKIP_REASONS):
                continue

            # Also drop known "too small" notes if they were only stored in content
            content_low = (d.page_content or "").lower()
            if "skipped: too small" in content_low:
                continue

            out.append(d)

        return out

    def _looks_like_table_query(self, q_lower: str) -> bool:
        if self._RE_TABLE.search(q_lower):
            return True
        table_words = ["table", "tables", "column", "columns", "row", "rows", "values", "dataset"]
        return any(w in q_lower for w in table_words)

    def _looks_like_figure_query(self, q_lower: str) -> bool:
        if self._RE_FIG.search(q_lower):
            return True
        fig_words = ["fig", "fig.", "figure", "image", "chart", "graph", "diagram", "picture", "caption"]
        return any(w in q_lower for w in fig_words)

    def _merge_and_dedupe(self, documents: List[LCDocument], k: int) -> List[LCDocument]:
        """
        Dedupes by stable content signature. Avoid python built-in hash() because
        it is salted per process -> non-deterministic between runs.
        """
        seen = set()
        out: List[LCDocument] = []
        for d in documents:
            sig = self._doc_signature(d)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(d)
            if len(out) >= k:
                break
        return out

    def _doc_signature(self, doc: LCDocument) -> str:
        """
        Build deterministic signature using metadata + first chars.
        """
        source = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", ""))
        doc_type = str(doc.metadata.get("doc_type", ""))
        head = (doc.page_content or "")[:200]
        return f"{source}|{page}|{doc_type}|{head}"

    def _boost_exact_table_matches(self, query: str, docs: List[LCDocument]) -> List[LCDocument]:
        m = self._RE_TABLE.search(query)
        if not m:
            return docs

        target_no = (m.group(2) or "").strip()
        if not target_no:
            return docs

        def score(d: LCDocument) -> int:
            # Higher is better
            dt = d.metadata.get("doc_type")
            if dt != "table":
                return 0
            # If you store table_number in metadata (you do), boost exact match
            if str(d.metadata.get("table_number", "")).strip().lower() == target_no.lower():
                return 2
            # Otherwise fallback: check content line prefix
            if (d.page_content or "").lower().startswith(f"table {target_no.lower()}"):
                return 1
            return 0

        return sorted(docs, key=score, reverse=True)

    def _boost_exact_figure_matches(self, query: str, docs: List[LCDocument]) -> List[LCDocument]:
        m = self._RE_FIG.search(query)
        if not m:
            return docs

        target_no = (m.group(2) or "").strip()
        if not target_no:
            return docs

        def score(d: LCDocument) -> int:
            dt = d.metadata.get("doc_type")
            if dt != "image":
                return 0
            if str(d.metadata.get("figure_number", "")).strip().lower() == target_no.lower():
                return 2
            if (d.page_content or "").lower().startswith(f"figure: fig. {target_no.lower()}"):
                return 1
            return 0

        return sorted(docs, key=score, reverse=True)
