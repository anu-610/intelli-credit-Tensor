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

# ── In-process DocAI document cache ────────────────────────────────────────
# Keyed by sha256(pdf_bytes) so the same PDF is never sent to DocAI twice
# within one Streamlit session / server process.
import hashlib as _hashlib
_DOCAI_DOC_CACHE: dict[str, list] = {}   # hash → list of documentai.Document objects


def _cache_key(pdf_bytes: bytes) -> str:
    return _hashlib.sha256(pdf_bytes).hexdigest()


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
    cached_docs = []  # store document objects for the cache

    for chunk_idx, chunk_bytes in enumerate(chunks, 1):
        print(f"   [DOCAI] Processing chunk {chunk_idx}/{len(chunks)}...")
        document = _docai_process_chunk(client, processor_name, chunk_bytes)
        cached_docs.append((page_offset, document))  # ← save for evidence reuse

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

    # ── Populate cache so extract_schema_with_evidence can reuse these docs ──
    _DOCAI_DOC_CACHE[_cache_key(pdf_bytes)] = cached_docs
    print(f"   [DOCAI] Cached {len(cached_docs)} document chunks for evidence reuse.")

    return full_document_text, extracted_risk_text, hit_count


# ── Table extraction helper ─────────────────────────────────────────────────

def extract_tables_with_docai(pdf_bytes: bytes) -> str:
    """
    Returns a Markdown representation of every table found by Document AI.
    Reuses cached documents from extract_text_with_docai — zero extra API calls.
    """
    err = _check_config()
    if err:
        return ""

    ck = _cache_key(pdf_bytes)

    # ── Use cached docs if available (they were populated by extract_text_with_docai) ──
    if ck in _DOCAI_DOC_CACHE:
        print("   [DOCAI] Tables: reusing cached documents — no extra API call.")
        cached_docs = _DOCAI_DOC_CACHE[ck]
    else:
        # Fallback: process now and cache
        print("   [DOCAI] Tables: cache miss — processing PDF...")
        processor_name = (
            f"projects/{_PROJECT_ID}/locations/{_LOCATION}"
            f"/processors/{_PROCESSOR_ID}"
        )
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{_LOCATION}-documentai.googleapis.com"}
        )
        chunks = _split_pdf_bytes(pdf_bytes, _DOCAI_PAGE_LIMIT)
        cached_docs = []
        page_offset = 0
        for chunk_bytes in chunks:
            document = _docai_process_chunk(client, processor_name, chunk_bytes)
            cached_docs.append((page_offset, document))
            page_offset += len(document.pages)
        _DOCAI_DOC_CACHE[ck] = cached_docs

    md_tables = []

    for page_offset, document in cached_docs:
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

    return "\n".join(md_tables)


# ── Schema extraction with bounding boxes + confidence ─────────────────────

def extract_schema_with_evidence(
    pdf_bytes: bytes,
    schema_fields: list[str],
    llm_function: Callable,
) -> list[dict]:
    """
    For each field in schema_fields, finds its value in the PDF and returns:
        {
          "field":       str,   # e.g. "Revenue (INR)"
          "value":       str,   # e.g. "₹9,01,468 Crore"
          "confidence":  int,   # 0-100
          "page_number": int,   # 1-based page where found
          "bbox":        dict,  # {x0, y0, x1, y1} as 0.0-1.0 fractions of page
          "context":     str,   # surrounding sentence for hover tooltip
        }

    IMPORTANT: Reuses the DocAI documents already cached by extract_text_with_docai
    (called earlier in the pipeline) so the PDF is NEVER sent to DocAI a second time.
    """
    err = _check_config()
    if err:
        return [
            {"field": f, "value": "N/A", "confidence": 0,
             "page_number": 0, "bbox": {}, "context": err}
            for f in schema_fields
        ]

    import json as _json

    ck = _cache_key(pdf_bytes)

    # ── Try to reuse cached DocAI documents ──────────────────────────────
    if ck in _DOCAI_DOC_CACHE:
        print("   [EVIDENCE] ✅ Reusing cached DocAI documents — skipping re-processing.")
        cached_docs = _DOCAI_DOC_CACHE[ck]   # list of (page_offset, document)
    else:
        # Cache miss: run DocAI now (only happens if called before extract_text_with_docai)
        print("   [EVIDENCE] Cache miss — running DocAI for evidence extraction...")
        processor_name = (
            f"projects/{_PROJECT_ID}/locations/{_LOCATION}"
            f"/processors/{_PROCESSOR_ID}"
        )
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{_LOCATION}-documentai.googleapis.com"}
        )
        chunks = _split_pdf_bytes(pdf_bytes, _DOCAI_PAGE_LIMIT)
        cached_docs = []
        page_offset = 0
        for chunk_idx, chunk_bytes in enumerate(chunks, 1):
            print(f"   [EVIDENCE] Processing chunk {chunk_idx}/{len(chunks)}...")
            document = _docai_process_chunk(client, processor_name, chunk_bytes)
            cached_docs.append((page_offset, document))
            page_offset += len(document.pages)
        _DOCAI_DOC_CACHE[ck] = cached_docs

    # ── Build token index from (cached) document objects ─────────────────
    token_index: list[dict] = []
    full_text_parts: list[str] = []

    for page_offset, document in cached_docs:
        full_text_parts.append(document.text)
        for page in document.pages:
            real_page = page_offset + int(page.page_number)
            for token in (page.tokens or []):
                segs = token.layout.text_anchor.text_segments
                for seg in segs:
                    s = int(seg.start_index or 0)
                    e = int(seg.end_index or 0)
                    tok_text = document.text[s:e]
                    verts = token.layout.bounding_poly.normalized_vertices
                    if len(verts) >= 4:
                        xs = [v.x for v in verts]
                        ys = [v.y for v in verts]
                        bbox = {
                            "x0": min(xs), "y0": min(ys),
                            "x1": max(xs), "y1": max(ys),
                        }
                    else:
                        bbox = {}
                    token_index.append({
                        "text": tok_text,
                        "page": real_page,
                        "bbox": bbox,
                    })

    full_text = "".join(full_text_parts)

    # ── Build a LINE index from the token index ───────────────────────────
    # Group tokens into lines by page + vertical band (y-center within 1% tolerance).
    # Each line entry: {page, text (joined tokens), bbox (union of token bboxes)}
    # This is what we search against — single tokens never contain full values.
    line_index: list[dict] = []
    if token_index:
        # Sort tokens by page, then by y-center, then x
        sorted_tokens = sorted(
            token_index,
            key=lambda t: (t["page"], round((t["bbox"].get("y0", 0) + t["bbox"].get("y1", 0)) / 2 * 200), t["bbox"].get("x0", 0))
            if t["bbox"] else (t["page"], 0, 0)
        )
        # Group into lines: same page + y-center within 0.8% of page height
        current_line_tokens: list[dict] = []
        current_page = None
        current_y_center = None

        def _flush_line(line_tokens: list[dict]) -> None:
            if not line_tokens:
                return
            texts  = [t["text"] for t in line_tokens if t["text"].strip()]
            bboxes = [t["bbox"] for t in line_tokens if t["bbox"]]
            if not texts or not bboxes:
                return
            x0 = min(b["x0"] for b in bboxes)
            y0 = min(b["y0"] for b in bboxes)
            x1 = max(b["x1"] for b in bboxes)
            y1 = max(b["y1"] for b in bboxes)
            line_index.append({
                "page": line_tokens[0]["page"],
                "text": " ".join(texts),
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            })

        for tok in sorted_tokens:
            if not tok["bbox"]:
                continue
            pg = tok["page"]
            yc = (tok["bbox"]["y0"] + tok["bbox"]["y1"]) / 2
            if current_page != pg or current_y_center is None or abs(yc - current_y_center) > 0.008:
                _flush_line(current_line_tokens)
                current_line_tokens = [tok]
                current_page = pg
                current_y_center = yc
            else:
                current_line_tokens.append(tok)
        _flush_line(current_line_tokens)

    # ── Ask LLM to extract each field + confidence ────────────────────────
    fields_list = "\n".join(f"  - {f}" for f in schema_fields)
    smart = _smart_extract(full_text, schema_fields)

    llm_prompt = f"""
You are a precise financial data extraction engine.
From the document text below, extract EACH of the following fields.

For EACH field return a JSON object with:
  "value"      : the extracted value (string), or "Not Found" if absent
  "confidence" : integer 0-100 (how certain you are this is correct)
  "page_hint"  : approximate page number where you found it (integer, 0 if unknown)
  "context"    : the exact 6-12 word phrase from the document that contains this value
                 (must be copy-pasted verbatim from the document text, not paraphrased)

Respond with ONLY a JSON object mapping field name → extraction object.
Example:
{{
  "Revenue (INR)": {{"value": "₹9,01,468 Crore", "confidence": 95, "page_hint": 12, "context": "Revenue from Operations 9,01,468 8,97,122"}},
  "Net Worth (INR)": {{"value": "Not Found", "confidence": 0, "page_hint": 0, "context": ""}}
}}

### FIELDS TO EXTRACT:
{fields_list}

### DOCUMENT TEXT:
{smart}
"""
    raw = llm_function(llm_prompt)

    # ── Parse LLM JSON ────────────────────────────────────────────────────
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        llm_extractions: dict = _json.loads(clean)
    except Exception:
        llm_extractions = {}
        for field in schema_fields:
            m = re.search(rf'"{re.escape(field)}"\s*:\s*\{{[^}}]+\}}', raw)
            if m:
                try:
                    llm_extractions[field] = _json.loads("{" + m.group(0).split("{", 1)[1])
                except Exception:
                    pass

    # ── Match each field to a line in the line_index → get real bbox ─────
    def _best_line_match(search_phrase: str, page_hint: int) -> dict | None:
        """
        Find the line in line_index whose text best overlaps with search_phrase.
        Strategy:
          1. Normalize both strings (strip ₹ , . spaces, lowercase)
          2. Score = number of search_phrase words (≥2 chars) found in line text
          3. Among ties, prefer the line closest to page_hint
        Returns the best matching line dict or None.
        """
        if not search_phrase or not line_index:
            return None

        def _norm(s: str) -> str:
            return re.sub(r"[₹,.\s]+", " ", s).lower().strip()

        phrase_norm = _norm(search_phrase)
        phrase_words = [w for w in phrase_norm.split() if len(w) >= 2]
        if not phrase_words:
            return None

        best_score = 0
        best_dist  = 9999
        best_line  = None

        for line in line_index:
            line_norm = _norm(line["text"])
            score = sum(1 for w in phrase_words if w in line_norm)
            if score == 0:
                continue
            dist = abs(line["page"] - page_hint) if page_hint else 0
            # Prefer higher score; break ties by proximity to page_hint
            if score > best_score or (score == best_score and dist < best_dist):
                best_score = score
                best_dist  = dist
                best_line  = line

        # Require at least 2 words to match to avoid false positives on short phrases
        return best_line if best_score >= 2 else None

    evidence: list[dict] = []

    for field in schema_fields:
        entry = llm_extractions.get(field, {})
        value      = entry.get("value", "Not Found")
        confidence = int(entry.get("confidence", 0))
        page_hint  = int(entry.get("page_hint", 0))
        context    = entry.get("context", "")

        bbox       = {}
        found_page = page_hint

        if value and value not in ("Not Found", "N/A", ""):
            # Try context phrase first (most reliable — verbatim from document)
            match = _best_line_match(context, page_hint)

            # Fallback: try matching the value itself if context didn't work
            if not match:
                match = _best_line_match(value, page_hint)

            if match:
                found_page = match["page"]
                # Expand bbox horizontally to full page width so the
                # highlight covers the entire table row, not just the matched words
                b = match["bbox"]
                bbox = {
                    "x0": 0.0,            # start at left margin
                    "y0": max(0.0, b["y0"] - 0.005),   # tiny padding above
                    "x1": 1.0,            # end at right margin
                    "y1": min(1.0, b["y1"] + 0.005),   # tiny padding below
                }

        evidence.append({
            "field":       field,
            "value":       value,
            "confidence":  confidence,
            "page_number": found_page,
            "bbox":        bbox,
            "context":     context,
        })
        hit = "✅" if bbox else "❌ no bbox"
        print(f"   [EVIDENCE] {field}: '{value}' | conf={confidence}% | page={found_page} | bbox={hit}")

    return evidence


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
