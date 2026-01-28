# src/rag/knowledge_base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import pickle
import re

import httpx
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import FAISS

try:
    # Newer langchain-openai package
    from langchain_openai import OpenAIEmbeddings
except Exception:
    # Fallback for older installs (if you used it)
    from langchain.embeddings.openai import OpenAIEmbeddings


@dataclass
class KnowledgeBase:
    """
    Thin wrapper over LangChain FAISS vectorstore + persistence.

    Key requirement for your pipeline:
    - FAISS cannot filter by metadata at search time.
      We implement: over-fetch -> filter in python -> cut to k.
    """

    embedding_model: str
    openai_api_key: Optional[str]
    index_dir: Union[str, Path]
    proxy_url: Optional[str] = None

    def __post_init__(self) -> None:
        self.index_dir = Path(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        self._embeddings = self._build_embeddings()
        self._vs: Optional[FAISS] = None  # vector store (FAISS)
        self._loaded_index_name: Optional[str] = None

    # ----------------------------
    # Embeddings
    # ----------------------------
    def _build_embeddings(self):
        if self.proxy_url:
            http_client = httpx.Client(proxy=self.proxy_url)
            return OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=self.openai_api_key,
                http_client=http_client,
            )
        return OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=self.openai_api_key,
        )

    # ----------------------------
    # Document ingestion
    # ----------------------------
    def add_documents(self, documents: List[Any]) -> None:
        """
        Accepts:
        - your ingestion Document objects (have .to_langchain_document())
        - or already-built langchain_core.documents.Document
        """
        lc_docs: List[LCDocument] = []
        for d in documents:
            if isinstance(d, LCDocument):
                lc_docs.append(d)
            else:
                # your ingestion Document wrapper
                if hasattr(d, "to_langchain_document"):
                    lc_docs.append(d.to_langchain_document())
                else:
                    raise TypeError(
                        f"Unsupported document type: {type(d)}. "
                        "Expected langchain Document or object with to_langchain_document()."
                    )

        if not lc_docs:
            return

        if self._vs is None:
            self._vs = FAISS.from_documents(lc_docs, self._embeddings)
        else:
            self._vs.add_documents(lc_docs)

    @property
    def document_count(self) -> int:
        if self._vs is None:
            return 0
        if hasattr(self._vs, "docstore") and hasattr(self._vs.docstore, "_dict"):
            return len(self._vs.docstore._dict)
        return 0

    # ----------------------------
    # Document access (for eval/dataset generation)
    # ----------------------------
    def get_all_documents(self) -> List[LCDocument]:
        """Return all documents from the index."""
        if self._vs is None:
            return []
        if hasattr(self._vs, "docstore") and hasattr(self._vs.docstore, "_dict"):
            return list(self._vs.docstore._dict.values())
        return []

    def iter_documents(self):
        """Generator yielding all documents from the index."""
        if self._vs is None:
            return
        if hasattr(self._vs, "docstore") and hasattr(self._vs.docstore, "_dict"):
            yield from self._vs.docstore._dict.values()

    def sample_documents(
        self,
        n: int,
        filter_dict: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> List[LCDocument]:
        """
        Sample n documents from the index, optionally filtered.
        Useful for dataset generation.
        """
        import random
        
        all_docs = self.get_all_documents()
        
        if filter_dict:
            # Reuse _apply_filter logic (need to wrap docs as (doc, 0.0) tuples)
            docs_with_scores = [(d, 0.0) for d in all_docs]
            filtered = self._apply_filter(docs_with_scores, filter_dict)
            all_docs = [d for d, _ in filtered]
        
        if seed is not None:
            random.seed(seed)
        
        if len(all_docs) <= n:
            return all_docs
        
        return random.sample(all_docs, n)

    def get_documents_by_source(self, source_substr: str) -> List[LCDocument]:
        """Get all documents whose source contains the given substring."""
        return [
            d for d in self.get_all_documents()
            if source_substr.lower() in str(d.metadata.get("source", "")).lower()
        ]

    def get_unique_sources(self) -> List[str]:
        """Get list of unique source filenames in the index."""
        sources = set()
        for doc in self.iter_documents():
            src = doc.metadata.get("source")
            if src:
                sources.add(src)
        return sorted(sources)

    # ----------------------------
    # Search
    # ----------------------------
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
    ) -> List[LCDocument]:
        """
        Returns only documents (no scores).

        - If filter_dict is provided, we over-fetch and filter locally.
        - score_threshold uses distance from FAISS (smaller = more similar).
          We treat threshold as "max distance allowed".
        """
        docs_scores = self.similarity_search_with_scores(
            query=query,
            k=k,
            filter_dict=filter_dict,
            score_threshold=score_threshold,
            fetch_k=fetch_k,
        )
        return [d for d, _ in docs_scores]

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
    ) -> List[Tuple[LCDocument, float]]:
        """
        Returns list of (doc, score). For FAISS similarity_search_with_score:
        score is typically distance (lower is better).
        """
        if self._vs is None:
            return []

        if k <= 0:
            return []

        # How many we ask from FAISS:
        # - if no filter: fetch exactly k
        # - if filter: fetch larger (fetch_k or fallback heuristic)
        n = k
        if filter_dict:
            if fetch_k is not None:
                n = max(fetch_k, k)
            else:
                n = max(k * 8, 40)

        raw = self._vs.similarity_search_with_score(query, k=n)

        # Apply score threshold first (if provided)
        if score_threshold is not None:
            raw = [(d, s) for (d, s) in raw if s <= score_threshold]

        # Apply metadata filter
        if filter_dict:
            raw = self._apply_filter(raw, filter_dict)

        return raw[:k]

    def _apply_filter(
        self,
        docs_scores: List[Tuple[LCDocument, float]],
        filter_dict: Dict[str, Any],
    ) -> List[Tuple[LCDocument, float]]:
        """
        Supported filter ops:
          - key=value (exact)
          - key__in=[...]
          - key__ne=value
          - key__contains="substr" (case-insensitive)
          - key__regex=r"..."
          - key__callable=fn(meta)->bool  (expected is callable)

        Backward compat:
          - doc_type also checks element_type
        """
        out: List[Tuple[LCDocument, float]] = []

        def get_meta_value(meta: Dict[str, Any], key: str):
            if key == "doc_type":
                return meta.get("doc_type"), meta.get("element_type")
            return meta.get(key), None

        for doc, score in docs_scores:
            meta = doc.metadata or {}
            ok = True

            for raw_key, expected in filter_dict.items():
                # parse operator suffixes
                if "__" in raw_key:
                    key, op = raw_key.split("__", 1)
                else:
                    key, op = raw_key, "eq"

                v1, v2 = get_meta_value(meta, key)

                # for doc_type fallback: accept either match
                def any_match_eq(val) -> bool:
                    if key == "doc_type":
                        return (v1 == val) or (v2 == val)
                    return v1 == val

                def any_match_contains(substr: str) -> bool:
                    substr = (substr or "").lower()
                    if key == "doc_type":
                        a = str(v1 or "").lower()
                        b = str(v2 or "").lower()
                        return substr in a or substr in b
                    return substr in str(v1 or "").lower()

                if op == "eq":
                    if callable(expected):
                        if not bool(expected(meta)):
                            ok = False
                            break
                    else:
                        if not any_match_eq(expected):
                            ok = False
                            break

                elif op == "ne":
                    if callable(expected):
                        if bool(expected(meta)):
                            ok = False
                            break
                    else:
                        if any_match_eq(expected):
                            ok = False
                            break

                elif op == "in":
                    if not isinstance(expected, (list, tuple, set)):
                        ok = False
                        break
                    if key == "doc_type":
                        if (v1 not in expected) and (v2 not in expected):
                            ok = False
                            break
                    else:
                        if v1 not in expected:
                            ok = False
                            break

                elif op == "contains":
                    if not any_match_contains(str(expected)):
                        ok = False
                        break

                elif op == "regex":
                    try:
                        pattern = re.compile(str(expected), re.IGNORECASE)
                        text = str(v1 or "")
                        text2 = str(v2 or "")
                        if key == "doc_type":
                            if not (pattern.search(text) or pattern.search(text2)):
                                ok = False
                                break
                        else:
                            if not pattern.search(text):
                                ok = False
                                break
                    except Exception:
                        ok = False
                        break

                elif op == "callable":
                    if not callable(expected) or not bool(expected(meta)):
                        ok = False
                        break

                else:
                    ok = False
                    break

            if ok:
                out.append((doc, score))

        return out

    # ----------------------------
    # Persistence
    # ----------------------------
    def save(self, index_name: str) -> str:
        """
        Persist FAISS index + docstore to index_dir/index_name.
        """
        if self._vs is None:
            raise RuntimeError("Nothing to save: vector store is empty")

        save_path = self.index_dir / index_name
        save_path.mkdir(parents=True, exist_ok=True)

        # LangChain FAISS native persistence: saves index.faiss + index.pkl
        self._vs.save_local(str(save_path))

        # Optional small meta
        meta = {
            "embedding_model": self.embedding_model,
            "index_name": index_name,
        }
        with open(save_path / "kb_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        self._loaded_index_name = index_name
        return str(save_path)

    def load(self, index_name: str) -> bool:
        """
        Load FAISS index from index_dir/index_name.
        """
        load_path = self.index_dir / index_name
        if not load_path.exists():
            return False

        try:
            self._vs = FAISS.load_local(
                str(load_path),
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
            )
            self._loaded_index_name = index_name
            return True
        except Exception:
            return False
