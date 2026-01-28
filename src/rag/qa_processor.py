from typing import List, Dict, Any
from dataclasses import dataclass, field
import os

from langchain_core.documents import Document as LCDocument
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import httpx


@dataclass
class QAResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    query: str = ""
    context_used: str = ""
    context_docs: List[LCDocument] = field(default_factory=list)


class QAProcessor:
    SYSTEM_PROMPT = """You are an AI assistant answering questions based on technical documents.
Your task is to provide accurate, detailed answers using ONLY the information from the provided context.

Rules:
1) Base your answer ONLY on the provided context.
2) Cite sources using format [Source N] where N is the source number.
3) If the context contains tables, you may reference specific data from them.
4) If the context contains image descriptions/captions, use that information when relevant.
5) If the answer is not in the context, say: "I cannot find this information in the provided documents".
6) Be concise but thorough.
7) Use markdown formatting for readability.
8) If the question specifies a particular paper/document, use ONLY sources from that paper and ignore others, even if present in context.
"""

    def __init__(
        self,
        model: str = "gpt-4",
        openai_api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        proxy_url: str = None
    ):
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")

        if proxy_url:
            http_client = httpx.Client(proxy=proxy_url)
            self.llm = ChatOpenAI(
                model=model,
                openai_api_key=openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                http_client=http_client
            )
        else:
            self.llm = ChatOpenAI(
                model=model,
                openai_api_key=openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )

    def _build_prompt(self, query: str, context: str) -> List:
        user_content = f"""Context from documents:
{context}

---

Question: {query}

Please provide a detailed answer based on the context above. Include source citations like [Source 1]."""

        return [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_content)
        ]

    def _extract_sources(self, documents: List[LCDocument]) -> List[Dict[str, Any]]:
        sources = []
        for i, doc in enumerate(documents, 1):
            sources.append({
                "index": i,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page"),
                "section": doc.metadata.get("section"),
                "doc_type": doc.metadata.get("doc_type", doc.metadata.get("element_type", "text")),
                "preview": (doc.page_content[:200] + "...") if (doc.page_content and len(doc.page_content) > 200) else (doc.page_content or "")
            })
        return sources

    def process(self, query: str, context: str, documents: List[LCDocument] = None) -> QAResult:
        messages = self._build_prompt(query, context)
        response = self.llm.invoke(messages)

        docs = documents if documents else []
        return QAResult(
            answer=response.content,
            sources=self._extract_sources(docs) if docs else [],
            query=query,
            context_used=context,
            context_docs=docs
        )

    def process_with_retrieval(self, query: str, retriever, k: int = 5) -> QAResult:
        documents = retriever.smart_retrieve(query, k=k)

        if not documents:
            return QAResult(
                answer="I couldn't find any relevant information in the documents.",
                sources=[],
                query=query,
                context_used="",
                context_docs=[]
            )

        # Hard scope clamp (prevents mixing papers in the final context)
        scope_source = None
        if hasattr(retriever, "infer_source_scope"):
            scope_source = retriever.infer_source_scope(query)

        if scope_source:
            scoped_docs = [d for d in documents if d.metadata.get("source") == scope_source]
            # If scoping would empty the context, keep original (failsafe)
            if scoped_docs:
                documents = scoped_docs

        context = retriever.format_context(documents)
        return self.process(query, context, documents)

    def batch_process(self, queries: List[str], retriever, k: int = 5) -> List[QAResult]:
        return [self.process_with_retrieval(q, retriever, k=k) for q in queries]
