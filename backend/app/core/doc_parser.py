"""
Universal Document Text Extractor
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Extracts clean text content from ANY uploaded document file format:
- PDF (.pdf) via pypdf
- Word (.docx, .doc) via python-docx
- CSV / TSV (.csv, .tsv) via pandas
- Excel (.xlsx, .xls) via pandas / openpyxl
- JSON (.json) structured dumps
- HTML / XML (.html, .htm, .xml) via regex / tag stripping
- Plain Text / Markdown / Code (.txt, .md, .log, .py, .js, etc.) via multi-encoding fallback
- Universal binary fallback with non-printable character cleaning
"""

import os
import io
import re
import json
from typing import Optional

def extract_text_from_file_bytes(filename: str, raw_bytes: bytes) -> str:
    """
    Extracts text content from raw file bytes based on file extension and format heuristics.
    Returns cleaned string passage.
    """
    fname_lower = (filename or "").lower().strip()

    # 1. PDF Documents
    if fname_lower.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            text_pages = []
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_pages.append(f"[Page {page_idx}]\n{page_text.strip()}")
            if text_pages:
                return "\n\n".join(text_pages)
        except Exception as e:
            print(f"[!] pypdf extraction warning for '{filename}': {e}")

    # 2. Word DOCX Documents
    if fname_lower.endswith(".docx") or fname_lower.endswith(".doc"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(raw_bytes))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            if paragraphs:
                return "\n\n".join(paragraphs)
        except Exception as e:
            print(f"[!] docx extraction warning for '{filename}': {e}")

    # 3. CSV / TSV Files
    if fname_lower.endswith(".csv") or fname_lower.endswith(".tsv"):
        try:
            import pandas as pd
            sep = "\t" if fname_lower.endswith(".tsv") else ","
            df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep)
            rows = []
            for idx, row in df.iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                rows.append(f"Record {idx+1}: {row_str}")
            if rows:
                return "\n".join(rows)
        except Exception as e:
            print(f"[!] CSV extraction warning for '{filename}': {e}")

    # 4. Excel Spreadsheets
    if fname_lower.endswith(".xlsx") or fname_lower.endswith(".xls"):
        try:
            import pandas as pd
            excel_file = pd.ExcelFile(io.BytesIO(raw_bytes))
            sheet_texts = []
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                rows = [f"Sheet: {sheet_name}"]
                for idx, row in df.iterrows():
                    row_str = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    rows.append(f"Row {idx+1}: {row_str}")
                sheet_texts.append("\n".join(rows))
            if sheet_texts:
                return "\n\n".join(sheet_texts)
        except Exception as e:
            print(f"[!] Excel extraction warning for '{filename}': {e}")

    # 5. JSON Files
    if fname_lower.endswith(".json"):
        try:
            data = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # 6. HTML / XML Files
    if fname_lower.endswith(".html") or fname_lower.endswith(".htm") or fname_lower.endswith(".xml"):
        try:
            raw_str = raw_bytes.decode("utf-8", errors="ignore")
            clean_str = re.sub(r"<[^>]+>", " ", raw_str)
            clean_str = re.sub(r"\s+", " ", clean_str).strip()
            if clean_str:
                return clean_str
        except Exception:
            pass

    # 7. Plain Text / Markdown / Code Files (UTF-8, Latin-1, CP1252)
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            decoded = raw_bytes.decode(encoding)
            if decoded and len(decoded.strip()) > 0:
                return decoded
        except Exception:
            continue

    # 8. Universal Binary Fallback: Decode errors='ignore' and clean unprintable control chars
    decoded_fallback = raw_bytes.decode("utf-8", errors="ignore")
    printable_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", decoded_fallback)
    return printable_text.strip()
