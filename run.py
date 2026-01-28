# run.py
#!/usr/bin/env python3
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings, INDEX_DIR, PROCESSED_DIR
from src.ingestion import PDFIngestor
from src.rag import KnowledgeBase, Retriever, QAProcessor


def _compute_chunk_id_for_doc(doc) -> str:
    """
    Compute deterministic chunk_id for existing documents (migration).
    Must match Document._compute_chunk_id() logic for consistency.
    """
    md = doc.metadata or {}
    source = str(md.get("source", "")).strip()
    page = str(md.get("page", "")).strip()
    doc_type = str(md.get("doc_type", md.get("element_type", "text"))).strip()
    # Include figure/table/image_index for better uniqueness
    fig = str(md.get("figure_number", "")).strip().upper()
    tbl = str(md.get("table_number", "")).strip().upper()
    img_idx = str(md.get("image_index", "")).strip()
    content_head = (doc.page_content or "")[:500]
    
    signature = f"{source}|{page}|{doc_type}|{fig}|{tbl}|{img_idx}|{content_head}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def ingest_command(args):
    settings = get_settings()
    proxy_url = settings.proxy_url_http if args.proxy else None

    ingestor = PDFIngestor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        output_dir=PROCESSED_DIR,
        openai_api_key=settings.openai_api_key,
        vision_model=settings.openai_vision_model,
        proxy_url=proxy_url,
    )

    pdf_path = Path(args.pdf)
    pdf_files = list(pdf_path.glob("*.pdf")) if pdf_path.is_dir() else [pdf_path]
    if not pdf_files:
        print("No PDF files found")
        return

    print(f"Processing {len(pdf_files)} PDF file(s) with unstructured...")
    all_documents = []
    for pdf_file in pdf_files:
        print(f"  Processing: {pdf_file.name}")
        docs = ingestor.ingest(
            str(pdf_file),
            generate_image_descriptions=(not args.no_vision),
            use_fast_parser=False,
        )
        all_documents.extend(docs)
        print(f"    Extracted {len(docs)} chunks")

    kb = KnowledgeBase(
        embedding_model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        index_dir=INDEX_DIR,
        proxy_url=proxy_url,
    )

    print("Building index...")
    kb.add_documents(all_documents)
    kb.save(args.index_name)
    print(f"Index saved as '{args.index_name}'")


def query_command(args):
    settings = get_settings()
    proxy_url = settings.proxy_url_http if args.proxy else None

    kb = KnowledgeBase(
        embedding_model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        index_dir=INDEX_DIR,
        proxy_url=proxy_url,
    )

    if not kb.load(args.index_name):
        print(f"Index '{args.index_name}' not found")
        return

    retriever = Retriever(kb, top_k=settings.retrieval_top_k)
    qa = QAProcessor(model=settings.llm_model, openai_api_key=settings.openai_api_key, proxy_url=proxy_url)

    result = qa.process_with_retrieval(args.question, retriever)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result.answer)

    if result.sources:
        print("\n" + "-" * 60)
        print("SOURCES:")
        print("-" * 60)
        for source in result.sources:
            print(f"  [{source['index']}] {source['source']} (Page {source.get('page', '?')})")


def ui_command(args):
    import subprocess
    app_path = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
    subprocess.run(["streamlit", "run", str(app_path), "--server.port", str(args.port), "--server.address", "0.0.0.0"])


def migrate_index_command(args):
    """
    Migrate old index to add chunk_id to all documents.
    This is a lightweight operation - only modifies docstore metadata,
    no re-embedding needed.
    """
    settings = get_settings()
    
    print(f"Loading index '{args.index}'...")
    kb = KnowledgeBase(
        embedding_model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        index_dir=INDEX_DIR,
    )
    
    if not kb.load(args.index):
        print(f"Error: Index '{args.index}' not found in {INDEX_DIR}")
        return
    
    # Access docstore directly
    if kb._vs is None or not hasattr(kb._vs, "docstore"):
        print("Error: Cannot access docstore")
        return
    
    docstore = kb._vs.docstore._dict
    total = len(docstore)
    patched = 0
    already_ok = 0
    
    print(f"Migrating {total} documents...")
    
    for doc_id, doc in docstore.items():
        md = doc.metadata or {}
        if md.get("chunk_id"):
            already_ok += 1
            continue
        
        # Compute and add chunk_id
        md["chunk_id"] = _compute_chunk_id_for_doc(doc)
        doc.metadata = md
        patched += 1
    
    print(f"  Already had chunk_id: {already_ok}")
    print(f"  Patched: {patched}")
    
    out_name = args.out or f"{args.index}_migrated"
    kb.save(out_name)
    print(f"Saved migrated index as '{out_name}'")


def main():
    parser = argparse.ArgumentParser(description="PDF Q&A System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF documents (unstructured)")
    ingest_parser.add_argument("pdf", help="Path to PDF file or directory")
    ingest_parser.add_argument("--index-name", default="default", help="Name for the index")
    ingest_parser.add_argument("--no-vision", action="store_true", help="Disable vision descriptions for images")
    ingest_parser.add_argument("--proxy", action="store_true", help="Use proxy for OpenAI")

    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--index-name", default="default", help="Index to query")
    query_parser.add_argument("--proxy", action="store_true", help="Use proxy for OpenAI")

    ui_parser = subparsers.add_parser("ui", help="Start the web UI")
    ui_parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit")

    migrate_parser = subparsers.add_parser("migrate-index", help="Add chunk_id to old index (for eval)")
    migrate_parser.add_argument("--index", required=True, help="Source index name to migrate")
    migrate_parser.add_argument("--out", default=None, help="Output index name (default: <index>_migrated)")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "query":
        query_command(args)
    elif args.command == "ui":
        ui_command(args)
    elif args.command == "migrate-index":
        migrate_index_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
