# src/ui/streamlit_app.py
import streamlit as st
from pathlib import Path
import os
from datetime import datetime
import uuid

from src.config import get_settings, UPLOADS_DIR, INDEX_DIR, PROCESSED_DIR
from src.ingestion import PDFIngestor
from src.rag import KnowledgeBase, Retriever, QAProcessor


def init_session_state():
    st.session_state.setdefault("knowledge_base", None)
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("qa_processor", None)
    st.session_state.setdefault("documents_loaded", False)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("use_proxy", False)  # default OFF
    st.session_state.setdefault("current_index_name", None)


def get_available_indexes():
    indexes = []
    if INDEX_DIR.exists():
        for item in INDEX_DIR.iterdir():
            if item.is_dir() and (item / "index.faiss").exists():
                indexes.append(item.name)
    return sorted(indexes)


def _proxy_url(settings, use_proxy: bool):
    return settings.proxy_url_http if use_proxy else None


def load_index(settings, index_name, use_proxy):
    proxy_url = _proxy_url(settings, use_proxy)

    kb = KnowledgeBase(
        embedding_model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        index_dir=INDEX_DIR,
        proxy_url=proxy_url,
    )

    if kb.load(index_name):
        st.session_state.knowledge_base = kb
        st.session_state.retriever = Retriever(kb, top_k=settings.retrieval_top_k)
        st.session_state.qa_processor = QAProcessor(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            proxy_url=proxy_url,
        )
        st.session_state.documents_loaded = True
        st.session_state.current_index_name = index_name
        return True
    return False


def _make_upload_run_dir() -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    p = UPLOADS_DIR / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _make_index_name_from_files(files) -> str:
    # короткое имя индекса, но уникальное
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "upload"
    if files:
        name0 = Path(files[0].name).stem[:20].replace(" ", "_")
        base = name0 if name0 else base
    return f"{base}_{ts}"


def process_uploaded_files(files, settings, use_proxy, generate_vision: bool):
    proxy_url = _proxy_url(settings, use_proxy)

    # ВАЖНО: output_dir = PROCESSED_DIR (единый канон для processed/images cache)
    ingestor = PDFIngestor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        output_dir=PROCESSED_DIR,
        openai_api_key=settings.openai_api_key,
        vision_model=settings.openai_vision_model,
        proxy_url=proxy_url,
    )

    run_dir = _make_upload_run_dir()
    index_name = _make_index_name_from_files(files)

    all_documents = []

    for file in files:
        pdf_path = run_dir / file.name
        with open(pdf_path, "wb") as f:
            f.write(file.getvalue())

        with st.spinner(f"Ingesting {file.name} (vision={'ON' if generate_vision else 'OFF'}) ..."):
            docs = ingestor.ingest(
                str(pdf_path),
                generate_image_descriptions=generate_vision,
                use_fast_parser=False,
            )
            all_documents.extend(docs)

    if not all_documents:
        st.error("No documents were extracted")
        return False

    kb = KnowledgeBase(
        embedding_model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        index_dir=INDEX_DIR,
        proxy_url=proxy_url,
    )

    with st.spinner(f"Building FAISS index '{index_name}' ..."):
        kb.add_documents(all_documents)
        kb.save(index_name)

    # сразу грузим в сессию
    st.session_state.knowledge_base = kb
    st.session_state.retriever = Retriever(kb, top_k=settings.retrieval_top_k)
    st.session_state.qa_processor = QAProcessor(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        proxy_url=proxy_url,
    )
    st.session_state.documents_loaded = True
    st.session_state.current_index_name = index_name
    return True


def display_sources(sources):
    if not sources:
        return

    with st.expander("View Sources", expanded=False):
        for source in sources:
            st.markdown(
                f"""
**Source {source['index']}**: {source['source']}
- Page: {source.get('page', 'N/A')}
- Type: {source['doc_type']}

> {source['preview'][:300]}...
                """.strip()
            )
            st.divider()


def main():
    st.set_page_config(page_title="PDF Q&A System", page_icon="📄", layout="wide")
    st.title("PDF Question Answering System")

    init_session_state()
    settings = get_settings()

    with st.sidebar:
        st.header("Settings")

        api_key = st.text_input("OpenAI API Key", value=settings.openai_api_key, type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            settings.openai_api_key = api_key

        st.session_state.use_proxy = st.checkbox(
            "Use Proxy",
            value=st.session_state.use_proxy,
            help="Enable proxy for OpenAI API requests",
        )
        if st.session_state.use_proxy:
            st.caption(f"Proxy: {settings.proxy_host}:{settings.proxy_port_http}")

        st.divider()

        st.header("Load Index")
        available_indexes = get_available_indexes()
        if available_indexes:
            selected_index = st.selectbox("Select index", options=available_indexes, index=0)

            if st.button("Load Selected Index"):
                if not api_key:
                    st.error("Please enter OpenAI API key first")
                else:
                    with st.spinner(f"Loading index '{selected_index}' ..."):
                        if load_index(settings, selected_index, st.session_state.use_proxy):
                            st.success(f"Index '{selected_index}' loaded!")
                        else:
                            st.error(f"Failed to load index '{selected_index}'")
        else:
            st.warning("No indexes found. Upload PDFs to create one.")

        st.divider()

        st.header("Upload PDFs → Build Index")
        generate_vision = st.checkbox("Generate image descriptions (vision)", value=True)

        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            if st.button("Process & Index Documents"):
                if not api_key:
                    st.error("Please enter OpenAI API key first")
                elif process_uploaded_files(uploaded_files, settings, st.session_state.use_proxy, generate_vision):
                    st.success(f"Indexed {len(uploaded_files)} PDF(s). Current index: {st.session_state.current_index_name}")

        st.divider()

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.markdown(
            f"""
**Status:**
- Index loaded: {'Yes' if st.session_state.documents_loaded else 'No'}
- Current index: {st.session_state.current_index_name or '-'}
- Proxy: {'On' if st.session_state.use_proxy else 'Off'}
            """.strip()
        )

    # render history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])

    # input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.documents_loaded:
            st.warning("Please load an index or upload PDFs first")
            return

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_processor.process_with_retrieval(
                    prompt,
                    st.session_state.retriever,
                    k=settings.retrieval_top_k,
                )

            st.markdown(result.answer)
            display_sources(result.sources)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": result.answer, "sources": result.sources}
            )


if __name__ == "__main__":
    main()
