import pandas as pd
from src.research_agent.llm_router import analyze_text_with_fallback

def analyze_structured_data(uploaded_files) -> str:
    """
    Original entry-point (no schema).  Kept for backward compatibility.
    """
    return analyze_structured_data_with_schema(
        uploaded_files=uploaded_files,
        schemas=None,
        classifications=None,
    )


def analyze_structured_data_with_schema(
    uploaded_files,
    schemas: dict[str, list[str]] | None,
    classifications: dict[str, str] | None,
) -> str:
    """
    Schema-aware structured data parser.

    Parameters
    ----------
    uploaded_files : list[UploadedFile]
        CSV / XLSX files uploaded in Stage 2.
    schemas : dict[str, list[str]] | None
        Maps document type → list of field names the user wants extracted.
        e.g. {"Borrowing Profile": ["Total Debt (INR)", "Debt-to-Equity Ratio", ...]}
        When None, falls back to generic forensic analysis.
    classifications : dict[str, str] | None
        Maps filename → document type (from Stage 3 HITL classification).
        e.g. {"borrowing_2024.csv": "Borrowing Profile"}
        Used to look up the correct schema for each file and to label the
        AI context with the user-confirmed document type.
    """
    if not uploaded_files:
        return "No structured files provided."

    report = "### Structured Data Synthesis\n"

    total_gst_revenue = None
    total_bank_deposits = None
    data_samples = ""

    for file in uploaded_files:
        try:
            df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            df.columns = df.columns.str.strip().str.replace("\ufeff", "").str.lower()

            # ── Resolve document type & schema for this file ──────────────────
            doc_type = (classifications or {}).get(file.name, "Unknown Document")
            file_schema = (schemas or {}).get(doc_type, [])

            file_label = f"File: {file.name}  |  Type: {doc_type}"

            # ── Legacy deterministic checks (preserved) ───────────────────────
            if "declared_revenue_inr" in df.columns:
                file_label += "  (GST/Revenue Filings)"
                total_gst_revenue = int(df["declared_revenue_inr"].sum())
            elif "total_inward_deposits_inr" in df.columns:
                file_label += "  (Bank Statements)"
                total_bank_deposits = int(df["total_inward_deposits_inr"].sum())
            elif "gross_receipts_inr" in df.columns or "net_profit_inr" in df.columns:
                file_label += "  (ITR / Tax Returns)"
            elif "top_buyer_name" in df.columns or "counterparty_name" in df.columns:
                file_label += "  (Counterparty Ledger)"

            # ── Schema column spotlight ───────────────────────────────────────
            # If the user defined a schema, try to find exact or fuzzy-matching
            # columns in the dataframe and surface them prominently in the
            # prompt so the LLM focuses on them first.
            schema_preview = ""
            if file_schema:
                matched_rows = []
                df_cols_lower = {c.lower(): c for c in df.columns}
                for field in file_schema:
                    # Exact column match (case-insensitive)
                    col_key = field.lower().replace(" ", "_")
                    matched_col = df_cols_lower.get(col_key) or df_cols_lower.get(field.lower())
                    if matched_col:
                        value = df[matched_col].dropna().iloc[0] if not df[matched_col].dropna().empty else "N/A"
                        matched_rows.append(f"| {field} | {value} |")
                    else:
                        matched_rows.append(f"| {field} | *(not found as column — extract from text)* |")

                schema_preview = (
                    "\n**Schema Fields Requested by Analyst:**\n"
                    "| Field | Detected Value |\n"
                    "|-------|---------------|\n"
                    + "\n".join(matched_rows)
                    + "\n"
                )

            sample_csv = df.head(50).to_csv(index=False)
            data_samples += (
                f"\n--- {file_label} ---\n"
                f"{schema_preview}"
                f"\nFull Data Sample (up to 50 rows):\n{sample_csv}\n"
            )

        except Exception as e:
            report += f"⚠️ Error parsing {file.name}: {e}\n"
            continue

    # ── Deterministic GST vs Bank cross-check (preserved) ─────────────────────
    if total_gst_revenue is not None and total_bank_deposits is not None:
        discrepancy = total_gst_revenue - total_bank_deposits
        report += f"- **Declared GST Revenue:** ₹{total_gst_revenue:,}\n"
        report += f"- **Actual Bank Deposits:** ₹{total_bank_deposits:,}\n"

        if discrepancy > 1_000_000:
            report += (
                f"\n**⚠️ EARLY WARNING SIGNAL:** Major discrepancy detected! "
                f"GST revenue is inflated by ₹{discrepancy:,} compared to actual bank "
                f"deposits. Potential revenue inflation or circular trading identified.\n"
            )
        elif discrepancy < -1_000_000:
            report += (
                f"\n**⚠️ ANOMALY:** Bank deposits significantly exceed declared GST "
                f"revenue by ₹{abs(discrepancy):,}. Possible unrecorded sales or "
                f"non-GST income.\n"
            )
        else:
            report += (
                "\n✅ GST flows match bank statements within an acceptable margin. "
                "Excellent financial hygiene.\n"
            )
    else:
        report += (
            "\n*Note: Insufficient data columns for deterministic GST vs. "
            "Bank discrepancy check.*\n"
        )

    # ── Schema instruction block for the LLM ──────────────────────────────────
    all_schema_fields = set()
    for fields in (schemas or {}).values():
        all_schema_fields.update(fields)

    schema_instruction = ""
    if all_schema_fields:
        fields_list = "\n".join(f"  - {f}" for f in sorted(all_schema_fields))
        schema_instruction = f"""
### MANDATORY SCHEMA EXTRACTION:
The analyst has defined the following fields as critical. You MUST produce a
consolidated Markdown table — | Document Type | Field | Extracted Value | — covering
all files. Mark fields as "Not Available" if absent from the data.

{fields_list}

After the schema table, proceed with the forensic analysis below.
"""

    print(
        f"🕵️ [CSV SCHEMA PARSER] Sending {len(uploaded_files)} file(s) "
        f"({'with' if all_schema_fields else 'without'} schema) to AI..."
    )

    ai_prompt = f"""
Act as a senior credit analyst at a Tier-1 bank conducting a structured data review.
{schema_instruction}
I am providing raw CSV/Excel data from a company's financial documents.

### EXTRACTED DATA SAMPLES:
{data_samples}

### FORENSIC INSTRUCTIONS:
Analyze this structured data across ALL provided files to find inconsistencies.
Pay special attention to:
1. **Loan Shopping:** Do bank deposits reflect massive revenue while ITR net
   profit is suspiciously low due to inflated expenses?
2. **Circular Trading:** Are there same-day sweeps where money enters and exits
   to specific counterparties, leaving a tiny closing balance?
3. **Concentration Risk:** Is revenue dependent on a single suspicious entity?

Output a concise, professional assessment using 3–5 bullet points.
Do NOT hallucinate data not present in the files.
"""

    ai_analysis = analyze_text_with_fallback(ai_prompt)
    report += f"\n### AI Forensic Multi-File Analysis\n{ai_analysis}\n"

    return report

