"""
Google Document AI — OCR parser for intelli-credit.

Replaces the PyMuPDF raw-text extraction path for PDFs that are:
  - Scanned / image-only (PyMuPDF returns empty text)
  - Multi-column layouts (PyMuPDF reads columns in wrong order)
  - Complex financial tables (PyMuPDF loses cell boundaries)
  - Rotated or watermarked pages

This module is a drop-in replacement.  It exposes the same
`analyze_pdf_risks` / `analyze_pdf_risks_with_schema` interface so
app.py doesn't need to change at all.

FALLBACK CHAIN (automatic, no user action needed):
  Google Document AI  →  PyMuPDF
  Falls back whenever:
    • SDK not installed
    • Credentials file missing / invalid
    • DOCAI_PROCESSOR_ID not configured
    • Any API / network error at runtime
    • Document AI returns empty text

Setup:
  1. Set GOOGLE_APPLICATION_CREDENTIALS in .env → path to your JSON key
  2. Set GOOGLE_CLOUD_PROJECT in .env → your GCP project ID
  3. Set DOCAI_PROCESSOR_ID in .env → processor ID from Document AI console
  4. Set DOCAI_LOCATION in .env → region (us / eu / asia)
  5. `pip install google-cloud-documentai`
"""

from __future__ import annotations

import os
import re
import io
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

# ── Lazy import so the rest of the app still works if SDK is not installed ──
try:
    from google.cloud import documentai
    _DOCAI_AVAILABLE = True
except ImportError:
    _DOCAI_AVAILABLE = False


# ── Config (read from environment) ─────────────────────────────────────────
_PROJECT_ID   = os.getenv("GOOGLE_CLOUD_PROJECT", "")
_LOCATION     = os.getenv("DOCAI_LOCATION", "us")
_PROCESSOR_ID = os.getenv("DOCAI_PROCESSOR_ID", "")


def _check_config() -> str | None:
    """Returns an error string if configuration is incomplete, else None."""
    if not _DOCAI_AVAILABLE:
        return (
            "⚠️ `google-cloud-documentai` is not installed. "
            "Run: `pip install google-cloud-documentai`"
        )
    if not _PROJECT_ID:
        return "⚠️ GOOGLE_CLOUD_PROJECT is not set in .env"
    if not _PROCESSOR_ID or _PROCESSOR_ID == "YOUR_PROCESSOR_ID_HERE":
        return (
            "⚠️ DOCAI_PROCESSOR_ID is not set in .env. "
            "Get it from the Document AI console."
        )
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_path and not os.path.exists(creds_path):
        return (
            f"⚠️ Credentials file not found at: {creds_path}. "
            "Download the JSON key from GCP → IAM → Service Accounts."
        )
    return None


# ── PDF chunking helper ─────────────────────────────────────────────────────
# Document AI free tier limit = 30 pages per request.
# We split large PDFs into 30-page chunks, process each, then merge results.

_DOCAI_PAGE_LIMIT = 15  # stay safely under the 30-page hard limit

def _split_pdf_bytes(pdf_bytes: bytes, chunk_size: int) -> list[bytes]:
    """
    Splits a PDF into chunks of `chunk_size` pages each.
    Returns a list of PDF bytes, one per chunk.
    Requires PyMuPDF (fitz) which is already a project dependency.
    """
    import fitz
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(src)
    chunks = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        dst = fitz.open()
        dst.insert_pdf(src, from_page=start, to_page=end - 1)
        chunks.append(dst.tobytes())
        dst.close()
    src.close()
    print(f"   [DOCAI] Split {total}-page PDF into {len(chunks)} chunks of ≤{chunk_size} pages")
    return chunks


def _docai_process_chunk(client, processor_name: str, pdf_bytes: bytes):
    """Send one chunk to Document AI and return the document object."""
    raw_document = documentai.RawDocument(
        content=pdf_bytes,
        mime_type="application/pdf",
    )
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document,
    )
    result = client.process_document(request=request)
    return result.document


# ── Core OCR function ───────────────────────────────────────────────────────

def extract_text_with_docai(pdf_bytes: bytes) -> tuple[str, str, int]:
    """
    Sends PDF bytes to Google Document AI and returns the same tuple as
    PyMuPDF's _extract_text_from_pdf:
        (full_document_text, extracted_risk_text, hit_count)

    Automatically splits PDFs that exceed the 30-page limit into chunks,
    processes each chunk, and merges the results transparently.
    """
    err = _check_config()
    if err:
        raise RuntimeError(err)

    processor_name = (
        f"projects/{_PROJECT_ID}/locations/{_LOCATION}"
        f"/processors/{_PROCESSOR_ID}"
    )
    client_options = {"api_endpoint": f"{_LOCATION}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(
        client_options=client_options
    )

    from src.data_ingestor.unstructured_parser import keyword_processor

    # Split into chunks so we never exceed the page limit
    chunks = _split_pdf_bytes(pdf_bytes, _DOCAI_PAGE_LIMIT)

    full_document_text = ""
    extracted_risk_text = ""
    hit_count = 0
    page_offset = 0  # track real page numbers across chunks

    for chunk_idx, chunk_bytes in enumerate(chunks, 1):
        print(f"   [DOCAI] Processing chunk {chunk_idx}/{len(chunks)}...")
        document = _docai_process_chunk(client, processor_name, chunk_bytes)

        full_document_text += document.text

        for page in document.pages:
            real_page_num = page_offset + int(page.page_number)
            page_text_parts = []
            for block in page.blocks:
                for segment in block.layout.text_anchor.text_segments:
                    start = int(segment.start_index) if segment.start_index else 0
                    end   = int(segment.end_index)   if segment.end_index   else 0
                    page_text_parts.append(document.text[start:end])
            page_text = " ".join(page_text_parts).strip()

            matches = keyword_processor.extract_keywords(page_text)
            if matches:
                hit_count += len(matches)
                unique_matches = list(set(m.lower() for m in matches))
                extracted_risk_text += (
                    f"\n--- [PAGE {real_page_num}] "
                    f"Keywords: {', '.join(unique_matches)} ---\n"
                    + page_text[:1500] + "...\n"
                )

        page_offset += len(document.pages)

    return full_document_text, extracted_risk_text, hit_count


# ── Table extraction helper ─────────────────────────────────────────────────

def extract_tables_with_docai(pdf_bytes: bytes) -> str:
    """
    Returns a Markdown representation of every table found by Document AI.
    Automatically chunks large PDFs to stay under the 30-page API limit.
    """
    err = _check_config()
    if err:
        return ""

    processor_name = (
        f"projects/{_PROJECT_ID}/locations/{_LOCATION}"
        f"/processors/{_PROCESSOR_ID}"
    )
    client_options = {"api_endpoint": f"{_LOCATION}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(
        client_options=client_options
    )

    chunks = _split_pdf_bytes(pdf_bytes, _DOCAI_PAGE_LIMIT)
    md_tables = []
    page_offset = 0

    for chunk_bytes in chunks:
        document = _docai_process_chunk(client, processor_name, chunk_bytes)

        for page in document.pages:
            real_page_num = page_offset + int(page.page_number)
            for table_idx, table in enumerate(page.tables, 1):
                rows_md = []
                header_done = False

                for row in table.header_rows:
                    cells = []
                    for cell in row.cells:
                        segs = cell.layout.text_anchor.text_segments
                        text = " ".join(
                            document.text[
                                int(s.start_index or 0): int(s.end_index or 0)
                            ].strip()
                            for s in segs
                        ).replace("\n", " ")
                        cells.append(text)
                    rows_md.append("| " + " | ".join(cells) + " |")
                    if not header_done:
                        rows_md.append("|" + " --- |" * len(cells))
                        header_done = True

                for row in table.body_rows:
                    cells = []
                    for cell in row.cells:
                        segs = cell.layout.text_anchor.text_segments
                        text = " ".join(
                            document.text[
                                int(s.start_index or 0): int(s.end_index or 0)
                            ].strip()
                            for s in segs
                        ).replace("\n", " ")
                        cells.append(text)
                    rows_md.append("| " + " | ".join(cells) + " |")

                if rows_md:
                    md_tables.append(
                        f"\n#### [Page {real_page_num} · Table {table_idx}]\n"
                        + "\n".join(rows_md)
                    )

        page_offset += len(document.pages)

    return "\n".join(md_tables)


# ── Fallback helpers ────────────────────────────────────────────────────────

def _smart_extract(full_text: str, schema: list[str] | None, max_chars: int = 28000) -> str:
    """
    Instead of naively taking the first N characters (which misses Balance
    Sheet / EPS / Dividends buried deep in the annual report), this function:

    1. Splits the full OCR text into page-sized blocks.
    2. Scores each block by how many schema fields + financial keywords it contains.
    3. Takes the top-scoring blocks up to max_chars, ensuring the LLM sees the
       most financially dense pages regardless of where they appear in the PDF.
    4. Fills remaining budget with front-matter (first 3000 chars) for context.
    """
    # Financial terms we always want to find — covers Balance Sheet, P&L, Notes
    FINANCE_SIGNALS = [
        "net worth", "shareholders equity", "total equity", "book value",
        "total assets", "fixed assets", "current assets", "non-current assets",
        "earnings per share", "eps", "basic eps", "diluted eps",
        "dividend", "dividend per share", "interim dividend", "final dividend",
        "profit after tax", "pat", "profit before tax", "pbt",
        "ebitda", "operating profit", "gross profit", "net profit",
        "revenue from operations", "total revenue", "turnover",
        "return on equity", "roe", "return on assets", "roa", "roce",
        "debt to equity", "debt-to-equity", "net debt", "gross debt",
        "cash and cash equivalents", "cash equivalents",
        "net npa", "gross npa", "capital adequacy", "crar", "tier 1",
        "interest income", "net interest income", "nim",
        "standalone", "consolidated", "balance sheet", "statement of profit",
        "notes to financial", "auditor", "independent auditor",
    ]

    # Add schema field names as extra signals
    schema_signals = [f.lower() for f in (schema or [])]
    all_signals = FINANCE_SIGNALS + schema_signals

    # Split into page blocks (Document AI separates pages with \n\n typically)
    # Use a page-size window of ~1500 chars to score
    window = 1500
    blocks = []
    for i in range(0, len(full_text), window):
        block = full_text[i: i + window]
        block_lower = block.lower()
        score = sum(1 for sig in all_signals if sig in block_lower)
        blocks.append((score, i, block))

    # Sort by score descending, keep top blocks up to max_chars * 0.7
    blocks_sorted = sorted(blocks, key=lambda x: x[0], reverse=True)
    budget = int(max_chars * 0.75)
    selected_positions = set()
    selected_text_parts = []
    used = 0

    for score, pos, block in blocks_sorted:
        if score == 0:
            break
        if used + len(block) > budget:
            continue
        selected_positions.add(pos)
        selected_text_parts.append((pos, f"\n[...page ~{pos//1500 + 1}...]\n" + block))
        used += len(block)

    # Fill remaining budget with the document front matter (executive summary etc.)
    front = full_text[:max_chars - used]
    selected_text_parts.append((0, front))

    # Re-sort by original position so the text reads naturally
    selected_text_parts.sort(key=lambda x: x[0])
    result = "\n".join(t for _, t in selected_text_parts)

    print(f"   [DOCAI] Smart extract: {len(full_text):,} → {len(result):,} chars "
          f"({len(selected_positions)} high-value blocks selected)")
    return result[:max_chars]


def _pymupdf_fallback(uploaded_file, llm_function: Callable, doc_type, schema) -> str:
    """Re-runs analysis using the original PyMuPDF path."""
    from src.data_ingestor.unstructured_parser import analyze_pdf_risks_with_schema as _pymupdf_run
    uploaded_file.seek(0)
    return _pymupdf_run(
        uploaded_file=uploaded_file,
        llm_function=llm_function,
        doc_type=doc_type,
        schema=schema,
    )


def _docai_or_fallback(
    uploaded_file,
    llm_function: Callable,
    doc_type: str | None,
    schema: list[str] | None,
) -> tuple[str, str, str | None]:
    """
    Attempts Document AI OCR.  Returns:
        (full_document_text, table_md, error_reason | None)

    error_reason is a human-readable string when Document AI failed,
    or None when it succeeded.
    """
    # ── Config gate ───────────────────────────────────────────────────────
    config_err = _check_config()
    if config_err:
        return "", "", config_err

    try:
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()

        full_document_text, extracted_risk_text, hit_count = extract_text_with_docai(
            pdf_bytes
        )

        # If Document AI returned nothing, treat as failure → fallback
        if not full_document_text.strip():
            return "", "", "Document AI returned empty text"

        table_md = extract_tables_with_docai(pdf_bytes)
        return full_document_text, table_md, None  # ← success

    except Exception as exc:
        return "", "", str(exc)


# ── Public API (drop-in for unstructured_parser.py) ────────────────────────

def analyze_pdf_risks(uploaded_file, llm_function: Callable) -> str:
    """Drop-in for unstructured_parser.analyze_pdf_risks"""
    return analyze_pdf_risks_with_schema(
        uploaded_file=uploaded_file,
        llm_function=llm_function,
        doc_type=None,
        schema=None,
    )


def analyze_pdf_risks_with_schema(
    uploaded_file,
    llm_function: Callable,
    doc_type: str | None,
    schema: list[str] | None,
) -> str:
    """
    Tries Google Document AI first.  If anything fails at any point
    (bad config, network error, quota exceeded, empty result, etc.)
    it automatically falls back to the original PyMuPDF pipeline.
    The user sees a small notice but always gets a result.
    """
    full_document_text, table_md, failure_reason = _docai_or_fallback(
        uploaded_file, llm_function, doc_type, schema
    )

    # ── FALLBACK: Document AI failed → silently switch to PyMuPDF ────────
    if failure_reason is not None:
        print(f"⚠️  [DOCAI] Failed ({failure_reason}). Falling back to PyMuPDF...")
        fallback_result = _pymupdf_fallback(
            uploaded_file, llm_function, doc_type, schema
        )
        # Prepend a small notice so the user knows which engine ran
        notice = (
            f"\n> ⚠️ **Google Document AI unavailable** "
            f"(`{failure_reason}`). "
            f"Analysis completed using **PyMuPDF** fallback engine.\n\n"
        )
        return notice + fallback_result

    # ── SUCCESS: Document AI returned text — run normal scenario logic ────
    # Append structured table markdown so the LLM sees clean tables
    if table_md:
        full_document_text += (
            "\n\n---\n### TABLES EXTRACTED BY DOCUMENT AI:\n" + table_md
        )

    # Smart-extract the most financially relevant sections of the document
    smart_text = _smart_extract(full_document_text, schema)

    # Re-run keyword scan on the full OCR'd text to determine scenario
    from src.data_ingestor.unstructured_parser import keyword_processor
    matches_all = keyword_processor.extract_keywords(full_document_text)
    hit_count = len(matches_all)
    unique_kw = list(set(m.lower() for m in matches_all))

    # ── Schema injection block ────────────────────────────────────────────
    schema_block = ""
    if schema:
        fields_list = "\n".join(f"  - {f}" for f in schema)
        schema_block = f"""
### DYNAMIC EXTRACTION SCHEMA (User-Defined — MANDATORY):
You MUST extract the following fields from the document. Present them as a
Markdown table with columns: | Field | Extracted Value | Source Page |
If a field is not found, write "Not Available" in the value column.

{fields_list}

After the schema table, proceed with the risk/forensic summary below.
"""

    doc_type_line = (
        f"Document Type (User-Confirmed): **{doc_type}**"
        if doc_type
        else "Document Type: Unknown"
    )

    # ── SCENARIO 2: 0 risk keywords ──────────────────────────────────────
    if hit_count == 0:
        print(
            f"📄 [DOCAI] 0 keywords in '{uploaded_file.name}'. "
            "Sending smart-extracted OCR text to AI..."
        )
        prompt = f"""
Act as a senior credit analyst at a Tier-1 bank.
{doc_type_line}
{schema_block}
I am providing key sections of a corporate document extracted using
Google Document AI OCR. The text has been intelligently selected to
prioritise pages containing financial statements, balance sheet data,
and schema-relevant information.

### DOCUMENT TEXT (Smart-Extracted via Document AI OCR):
{smart_text}

### INSTRUCTIONS:
1. Extract ALL fields from the DYNAMIC EXTRACTION SCHEMA above (if provided).
   Search carefully — data may appear in tables, notes, or appendices.
   If a value truly cannot be found anywhere, only then write "Not Available".
2. Summarize the document's key financial performance data.
3. Confirm whether the document appears clean or flags any hidden risks.

### FORMATTING RULES:
- Use strict Markdown.
- Use bullet points for every observation.
- **Bold** key metrics and risk terms.
- Keep it professional and concise.
"""
        return llm_function(prompt)

    # ── SCENARIO 3: risk keywords found ──────────────────────────────────
    print(
        f"📄 [DOCAI] {hit_count} keywords in '{uploaded_file.name}'. "
        "Sending to AI..."
    )
    prompt = f"""
Act as a senior credit analyst at a Tier-1 bank.
{doc_type_line}
{schema_block}
I have pre-filtered the document using a 5,000-word forensic dictionary
and extracted {hit_count} keyword hits across multiple pages.
The text was extracted using Google Document AI OCR and intelligently
selected to include the most financially relevant pages.

### DOCUMENT EXTRACT (Smart-Selected · Document AI OCR):
{smart_text}

### INSTRUCTIONS:
1. Extract ALL fields from the DYNAMIC EXTRACTION SCHEMA above (if provided).
   Search carefully — Net Worth, Total Assets, EPS, Dividends and similar
   metrics are often in the Balance Sheet or Notes sections. Do NOT mark
   a field as "Not Available" unless it is truly absent everywhere.
2. Extract specific quantitative statistics (Revenue, Net Profit, EBITDA,
   Margins, Growth %, Net Worth, NPA %, EPS, Dividend, Total Assets, etc.).
3. Summarize actual risks. If a risk keyword appears in a benign context
   (e.g., "We have NO litigation"), state that explicitly.

### FORMATTING RULES:
- Use strict Markdown.
- Use bullet points for every observation.
- **Bold** specific metrics, keywords, and risk categories.
- Do NOT use raw HTML. Keep it clean and readable.
"""
    return llm_function(prompt)
