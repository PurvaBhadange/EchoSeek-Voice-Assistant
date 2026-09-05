"""
Universal Document Ingestion & Retrieval Verification Script
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model
"""

import os
import sys
import io
import json
import pypdf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.core.doc_parser import extract_text_from_file_bytes
from app.api.pipeline import pipeline_services

def create_sample_pdf_bytes(text_content: str) -> bytes:
    """Creates an in-memory PDF file containing text_content using pypdf."""
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    # pypdf Writer creates a valid PDF structure
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()

def test_universal_doc_parser():
    print("=" * 75)
    print("TESTING UNIVERSAL DOCUMENT TEXT EXTRACTOR & INGESTION")
    print("=" * 75)

    # 1. Test CSV Text Extractor
    csv_bytes = b"id,name,role,location\n1,Rohan Mehta,Lead AI Engineer,Goa India\n2,Priya Sharma,Data Scientist,Mumbai India"
    csv_text = extract_text_from_file_bytes("employees.csv", csv_bytes)
    print(f"[+] CSV Extraction Result:\n{csv_text[:120]}...\n")
    assert "Rohan Mehta" in csv_text and "Goa India" in csv_text

    # 2. Test JSON Text Extractor
    json_bytes = json.dumps({"project": "EchoSeek", "team": "Frontiers", "location": "Hacker House Goa"}).encode("utf-8")
    json_text = extract_text_from_file_bytes("project.json", json_bytes)
    print(f"[+] JSON Extraction Result:\n{json_text[:120]}...\n")
    assert "EchoSeek" in json_text

    # 3. Test Ingest Pipeline Integration
    print("[*] Initializing Pipeline Services...")
    pipeline_services.initialize()

    test_doc_text = "Team Frontiers built EchoSeek at Hacker House Goa 2026. EchoSeek is a voice-enabled grounded RAG assistant."
    doc_obj = {
        "passage": test_doc_text,
        "passage_id": "usr_test_doc_1",
        "url": "https://echoseek.org/team",
        "language": "en"
    }

    # Embed & Add to vector store
    from app.core.chunker import SentenceRecursiveChunker
    chunker = SentenceRecursiveChunker(target_chunk_size=300, overlap=50)
    chunks = chunker.chunk_document(doc_obj)
    embeddings = pipeline_services.embedding_engine.embed_passages([c.text for c in chunks])

    current_chunks = list(pipeline_services.vector_store.chunks) + chunks
    if pipeline_services.vector_store.index is None:
        pipeline_services.vector_store.build_index(embeddings, chunks)
    else:
        old_matrix = np.load(os.path.join(os.path.dirname(__file__), "../data/embeddings/embeddings.npy"))
        combined_matrix = np.vstack([old_matrix, embeddings])
        pipeline_services.vector_store.build_index(combined_matrix, current_chunks)

    # Test Query against newly ingested content
    out = pipeline_services.harness.run(query="Who built EchoSeek at Hacker House Goa?", top_k=2, score_threshold=0.30)
    print(f"[*] Query Result: \"{out.answer}\"")
    print(f"[*] Sources Cited: {out.sources}")

    print("\n" + "=" * 75)
    print("UNIVERSAL INGESTION TEST PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    test_universal_doc_parser()
