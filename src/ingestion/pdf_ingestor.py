from typing import List, Union
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re

from .text_processor import TextProcessor
from .table_processor import TableProcessor
from .image_processor import ImageProcessor


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_type: str = "text"

    def _compute_chunk_id(self) -> str:
        """
        Deterministic chunk_id based on content + key metadata.
        Stable across runs for reproducibility.
        Must match run.py:_compute_chunk_id_for_doc() for migration consistency.
        """
        source = str(self.metadata.get("source", "")).strip()
        page = str(self.metadata.get("page", "")).strip()
        doc_type = (self.doc_type or "text").strip()
        # Include figure/table/image_index for better uniqueness
        fig = str(self.metadata.get("figure_number", "")).strip().upper()
        tbl = str(self.metadata.get("table_number", "")).strip().upper()
        img_idx = str(self.metadata.get("image_index", "")).strip()
        content_head = (self.content or "")[:500]
        
        signature = f"{source}|{page}|{doc_type}|{fig}|{tbl}|{img_idx}|{content_head}"
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]

    @property
    def chunk_id(self) -> str:
        """Return cached or compute chunk_id."""
        if "chunk_id" not in self.metadata:
            self.metadata["chunk_id"] = self._compute_chunk_id()
        return self.metadata["chunk_id"]

    def to_langchain_document(self):
        from langchain_core.documents import Document as LCDocument
        meta = dict(self.metadata or {})
        meta["doc_type"] = self.doc_type
        meta.setdefault("element_type", self.doc_type)
        # Ensure chunk_id is in metadata
        meta["chunk_id"] = self.chunk_id
        return LCDocument(page_content=self.content, metadata=meta)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "doc_type": self.doc_type,
            "chunk_id": self.chunk_id,
        }


class PDFIngestor:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        output_dir: Path = None,
        openai_api_key: str = None,
        vision_model: str = "gpt-4-vision-preview",
        proxy_url: str = None,
    ):
        # robust roman/numeric patterns
        self._re_fig = re.compile(r"\b(?:Fig\.|Figure)\s*([0-9]+|[IVXLCDM]+)\b", re.IGNORECASE)
        self._re_table = re.compile(r"\b(?:Table)\s*([0-9]+|[IVXLCDM]+)\b", re.IGNORECASE)

        self.text_processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.table_processor = TableProcessor()

        # IMPORTANT: images extracted by unstructured will be written here
        self.output_dir = output_dir or Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store images under output_dir/images to keep everything together & deterministic
        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        self.image_processor = ImageProcessor(
            output_dir=images_dir,
            openai_api_key=openai_api_key,
            vision_model=vision_model,
            proxy_url=proxy_url,
        )

        print(
            "[ImageProcessor] "
            f"proxy={'ON' if proxy_url else 'OFF'}, "
            f"vision_model={vision_model}, "
            f"openai_key_set={bool(openai_api_key)}; "
            f"images_dir={images_dir}"
)

    def _generate_doc_id(self, pdf_path: str) -> str:
        return hashlib.md5(str(pdf_path).encode()).hexdigest()[:12]

    def _parse_with_unstructured(self, pdf_path: str, strategy: str = "auto") -> List[dict]:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Figure"],
            extract_image_block_output_dir=str(self.image_processor.output_dir),
        )

        parsed = []
        current_title = None

        for element in elements:
            elem_type = type(element).__name__
            elem_dict = {"type": elem_type, "text": str(element), "metadata": {}}

            if hasattr(element, "metadata"):
                meta = element.metadata
                if hasattr(meta, "page_number"):
                    elem_dict["metadata"]["page"] = meta.page_number
                if hasattr(meta, "text_as_html"):
                    elem_dict["html"] = meta.text_as_html
                if hasattr(meta, "image_path"):
                    elem_dict["image_path"] = meta.image_path

            if elem_type == "Title":
                current_title = str(element)
                elem_dict["is_title"] = True
            else:
                elem_dict["parent_title"] = current_title

            parsed.append(elem_dict)

        return parsed

    def ingest(
        self,
        pdf_path: Union[str, Path],
        strategy: str = "auto",
        generate_image_descriptions: bool = True,
        use_fast_parser: bool = False,
    ) -> List[Document]:
        if use_fast_parser:
            raise ValueError("Fast parser is disabled. Use unstructured only.")

        pdf_path = str(pdf_path)
        doc_id = self._generate_doc_id(pdf_path)
        filename = Path(pdf_path).name

        base_metadata = {"doc_id": doc_id, "source": filename, "source_path": pdf_path}

        try:
            elements = self._parse_with_unstructured(pdf_path, strategy)
        except Exception as e:
            raise RuntimeError(f"Unstructured parsing failed for {pdf_path}: {e}") from e

        documents: List[Document] = []
        current_section = None
        text_buffer: List[str] = []
        text_metadata = {}

        # Track numbering + dedup
        last_table_number_by_page = {}
        image_counter_by_page = {}
        seen_image_paths = set()

        for elem in elements:
            elem_type = elem.get("type", "")
            text = elem.get("text", "") or ""
            meta = {**base_metadata, **elem.get("metadata", {})}
            page_num = meta.get("page")

            # detect latest "Table X" seen on this page (helps when unstructured table meta misses number)
            mt = self._re_table.search(text)
            if mt and page_num:
                last_table_number_by_page[page_num] = mt.group(1)

            if elem_type == "Title":
                if text_buffer:
                    documents.extend(self._flush_text_buffer(text_buffer, current_section, text_metadata))
                    text_buffer = []

                current_section = text
                text_metadata = meta.copy()
                continue

            if elem_type == "Table":
                if text_buffer:
                    documents.extend(self._flush_text_buffer(text_buffer, current_section, text_metadata))
                    text_buffer = []

                html = elem.get("html", "") or ""
                if page_num and page_num in last_table_number_by_page:
                    meta["table_number"] = last_table_number_by_page[page_num]

                if html.strip():
                    table_chunk = self.table_processor.process(html, meta)
                    documents.append(Document(content=table_chunk.content, metadata=table_chunk.metadata, doc_type="table"))
                else:
                    # fallback
                    table_no = meta.get("table_number") or (last_table_number_by_page.get(page_num) if page_num else None)
                    prefix = f"Table {table_no}:" if table_no else "Table:"
                    documents.append(
                        Document(
                            content=f"{prefix}\n{text}".strip(),
                            metadata={**meta, "element_type": "table"},
                            doc_type="table",
                        )
                    )
                continue

            if elem_type in ("Image", "Figure"):
                if text_buffer:
                    documents.extend(self._flush_text_buffer(text_buffer, current_section, text_metadata))
                    text_buffer = []

                image_path = elem.get("image_path")
                caption = (elem.get("text", "") or "").strip()

                # dedupe by image_path if present
                if image_path:
                    if image_path in seen_image_paths:
                        continue
                    seen_image_paths.add(image_path)

                # stable per-page image_index
                if page_num is not None:
                    idx = image_counter_by_page.get(page_num, 0)
                    image_counter_by_page[page_num] = idx + 1
                else:
                    idx = image_counter_by_page.get("_nopage_", 0)
                    image_counter_by_page["_nopage_"] = idx + 1

                meta["image_index"] = idx

                # Extract figure number from caption if possible
                mf = self._re_fig.search(caption) if caption else None
                if mf:
                    meta["figure_number"] = mf.group(1)

                image_chunk = self.image_processor.process(
                    image_path=image_path,
                    caption=caption,
                    metadata=meta,
                    generate_description=generate_image_descriptions,
                )

                content = self._inject_figure_anchor(image_chunk.content, meta)
                documents.append(Document(content=content, metadata=image_chunk.metadata, doc_type="image"))
                continue

            if elem_type in ("NarrativeText", "ListItem", "Text"):
                if text.strip():
                    text_buffer.append(text)
                    if not text_metadata:
                        text_metadata = meta.copy()
                continue

            # ignore everything else

        if text_buffer:
            documents.extend(self._flush_text_buffer(text_buffer, current_section, text_metadata))

        return documents

    def _inject_figure_anchor(self, content: str, meta: dict) -> str:
        fig_no = meta.get("figure_number")
        if not fig_no:
            return content
        anchor = f"FIGURE: Fig. {fig_no}"
        if content.startswith(anchor):
            return content
        return f"{anchor}\n{content}"

    def _flush_text_buffer(self, buffer: List[str], section_title: str, metadata: dict) -> List[Document]:
        if not buffer:
            return []

        combined_text = "\n\n".join(buffer).strip()
        if not combined_text:
            return []

        if section_title:
            metadata = {**(metadata or {}), "section": section_title}

        chunks = self.text_processor.merge_title_with_content(section_title, combined_text, metadata)

        return [
            Document(content=chunk.content, metadata={**chunk.metadata, "element_type": "text"}, doc_type="text")
            for chunk in chunks
        ]

    def ingest_multiple(self, pdf_paths: List[Union[str, Path]], **kwargs) -> List[Document]:
        all_documents = []
        for pdf_path in pdf_paths:
            all_documents.extend(self.ingest(pdf_path, **kwargs))
        return all_documents

    def save_documents(self, documents: List[Document], output_path: Union[str, Path] = None) -> str:
        output_path = Path(output_path) if output_path else (self.output_dir / "documents.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
        return str(output_path)

    def load_documents(self, input_path: Union[str, Path]) -> List[Document]:
        docs = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                docs.append(Document(content=data["content"], metadata=data["metadata"], doc_type=data.get("doc_type", "text")))
        return docs
