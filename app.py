import streamlit as st
import time
import re
import json
import os
from datetime import datetime
from fpdf import FPDF
from dotenv import load_dotenv

load_dotenv()

# --- BACKEND IMPORTS (reused from existing modules) ---
from src.research_agent.llm_router import analyze_text_with_fallback
from src.decision_engine.ml_scorer import calculate_risk_score
from src.data_ingestor.structured_parser import analyze_structured_data, analyze_structured_data_with_schema
from src.research_agent.web_crawler import crawl_company_news

# ── PDF OCR backend selector ───────────────────────────────────────────────
# Set USE_GOOGLE_DOCAI=true in .env to switch from PyMuPDF → Google Document AI.
# Falls back to PyMuPDF automatically if the SDK / credentials are not set up.
_USE_DOCAI = os.getenv("USE_GOOGLE_DOCAI", "false").lower() == "true"

if _USE_DOCAI:
    try:
        from src.data_ingestor.docai_parser import (
            analyze_pdf_risks,
            analyze_pdf_risks_with_schema,
        )
        _OCR_BACKEND = "Google Document AI"
    except Exception as _docai_import_err:
        st.warning(
            f"⚠️ Google Document AI import failed ({_docai_import_err}). "
            "Falling back to PyMuPDF."
        )
        from src.data_ingestor.unstructured_parser import (
            analyze_pdf_risks,
            analyze_pdf_risks_with_schema,
        )
        _OCR_BACKEND = "PyMuPDF (fallback)"
else:
    from src.data_ingestor.unstructured_parser import (
        analyze_pdf_risks,
        analyze_pdf_risks_with_schema,
    )
    _OCR_BACKEND = "PyMuPDF"

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Intelli-Credit Engine · Tensor",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS — TENSOR THEME (Light Mode, High Contrast)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

  /* ══ 1. HIDE STREAMLIT TOP HEADER ══ */
  header[data-testid="stHeader"] { display: none !important; height: 0px !important; }

  /* ══ 2. HIDE SIDEBAR TOGGLE BUTTONS ══ */
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapseButton"],
  .stSidebarCollapseButton { display: none !important; visibility: hidden !important; width: 0 !important; }

  /* ══ LAYOUT ══ */
  .block-container {
    padding-top: 2rem !important;
    padding-left: 2.4rem !important;
    padding-right: 2.4rem !important;
    max-width: 1360px !important;
    background-color: #ffffff !important;
  }

  /* ══ FULL WHITE BACKGROUND ══ */
  html, body                              { background-color: #ffffff !important; }
  .stApp                                  { background-color: #ffffff !important; }
  .main                                   { background-color: #ffffff !important; }
  section[data-testid="stMain"]           { background-color: #ffffff !important; }
  div[data-testid="stMainBlockContainer"] { background-color: #ffffff !important; }

  /* ══ GLOBAL FONT & BASE TEXT COLOR ══
     Every element in .stApp defaults to dark navy.
     Specific overrides below restore white where needed. */
  html, body, .stApp, .stApp * {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #1a2340;
  }
  /* Carve-out: never let the global rule corrupt Streamlit's SVG/Material icon spans */
  details > summary [data-testid="stExpanderToggleIcon"],
  details > summary [data-testid="stExpanderToggleIcon"] * {
    color: #1a5fd4 !important;
    font-size: 24px !important; /* Often needed to properly size Material Icons */
    font-weight: normal !important;
    font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important; /* Restores the icon font */
  }
  h1, h2, h3, h4, h5, h6 { color: #0d1f3c !important; }

  /* ══ INPUT FIELDS ══ */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stTextArea > div > div > textarea,
  [data-baseweb="input"] input,
  [data-baseweb="base-input"] input,
  [data-baseweb="textarea"] textarea {
    background-color: #ffffff !important;
    border: 1.5px solid #c5d5ea !important;
    color: #0d1f3c !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    box-shadow: 0 1px 4px rgba(26,63,130,0.05) !important;
  }
  .stTextInput > div > div > input::placeholder,
  .stNumberInput > div > div > input::placeholder,
  .stTextArea > div > div > textarea::placeholder,
  [data-baseweb="input"] input::placeholder,
  [data-baseweb="textarea"] textarea::placeholder {
    color: #6a88aa !important;
    opacity: 1 !important;
  }
  .stTextInput > div > div > input:focus,
  .stNumberInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
    border-color: #1a5fd4 !important;
    box-shadow: 0 0 0 3px rgba(26,95,212,0.10) !important;
  }
  .stTextInput label, .stNumberInput label,
  .stTextArea label, .stFileUploader label,
  label, .stWidgetLabel, [data-testid="stWidgetLabel"] p {
    font-weight: 600 !important; font-size: 13px !important;
    color: #2a3a5c !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
  }
            
/* ══ RESTORE MATERIAL ICONS (FIX FOR TEXT OVERLAP) ══ */
  .stApp span[class*="material"],
  .stApp [data-testid="stExpanderToggleIcon"],
  .stApp [data-testid="stExpanderToggleIcon"] *,
  .stApp [data-testid="stIconMaterial"],
  .stApp [data-testid="stIconMaterial"] * {
      font-family: "Material Symbols Rounded", "Material Icons" !important;
      font-weight: normal !important;
      font-style: normal !important;
      font-size: 24px !important;
      text-transform: none !important;      /* Prevents uppercase from breaking the ligature */
      letter-spacing: normal !important;    /* Prevents spacing from breaking the ligature */
      word-wrap: normal !important;
      white-space: nowrap !important;
      color: #1a5fd4 !important;            /* Keeps your theme's blue tint */
  }

  /* ══ NUMBER INPUT STEPPER ══ */
  .stNumberInput button,
  div[data-testid="stNumberInput"] button {
    background-color: #eef4ff !important;
    border: 1.5px solid #c5d5ea !important;
    color: #1a5fd4 !important;
    border-radius: 6px !important;
  }

  /* ══ SELECTBOX ══ */
  [data-testid="stSelectbox"] > div > div,
  div[data-baseweb="select"] > div:first-child {
    background-color: #ffffff !important;
    border: 1.5px solid #c5d5ea !important;
    border-radius: 8px !important;
  }
  [data-testid="stSelectbox"] span,
  [data-testid="stSelectbox"] div,
  [data-testid="stSelectbox"] p,
  div[data-baseweb="select"] span,
  div[data-baseweb="select"] div,
  div[data-baseweb="select"] input { color: #0d1f3c !important; }

  /* Dropdown popup portal */
  [data-baseweb="popover"], [data-baseweb="popover"] > div { background-color: #ffffff !important; }
  [data-baseweb="menu"], ul[data-baseweb="menu"] {
    background-color: #ffffff !important;
    border: 1.5px solid #c5d5ea !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 28px rgba(26,60,120,0.12) !important;
  }
  [data-baseweb="menu"] li, [role="option"] {
    background-color: #ffffff !important;
    color: #0d1f3c !important;
    font-size: 13px !important;
  }
  [role="option"]:hover, [data-baseweb="menu"] li:hover {
    background-color: #e4edfc !important; color: #1a5fd4 !important;
  }
  [role="option"] *, [data-baseweb="menu"] li * {
    color: #0d1f3c !important; background-color: transparent !important;
  }
  [role="option"]:hover *, [data-baseweb="menu"] li:hover * { color: #1a5fd4 !important; }

  /* ══ FILE UPLOADER ══ */
  [data-testid="stFileUploadDropzone"],
  [data-testid="stFileUploaderDropzone"] {
    background-color: #DCDCDC !important;
    border: 2px dashed #a0a8b8 !important;
    border-radius: 10px !important;
    padding: 18px 16px !important;
  }
  [data-testid="stFileUploadDropzone"] p,
  [data-testid="stFileUploadDropzone"] span,
  [data-testid="stFileUploadDropzone"] small,
  [data-testid="stFileUploadDropzone"] div,
  [data-testid="stFileUploaderDropzone"] span,
  [data-testid="stFileUploaderDropzone"] p,
  [data-testid="stFileUploaderDropzone"] small,
  [data-testid="stFileUploaderDropzone"] div { color: #1a2c48 !important; }
  [data-testid="stFileUploadDropzone"] button,
  [data-testid="stFileUploaderDropzone"] button {
    background: #ffffff !important;
    border: 1.5px solid #1a5fd4 !important;
    color: #1a5fd4 !important;
    border-radius: 7px !important; font-weight: 600 !important;
  }
  /* Uploaded file rows */
  [data-testid="stFileUploaderFile"],
  [data-testid="stFileUploaderFileData"] {
    background-color: #ffffff !important;
    border: 1px solid #dce8f8 !important; border-radius: 8px !important;
  }
  [data-testid="stFileUploaderFile"] * { color: #0d1f3c !important; background-color: transparent !important; }

  /* ══ BUTTONS ══ */
  button[kind="primary"] {
    background: linear-gradient(135deg, #1a5fd4 0%, #0d3fa8 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important; font-size: 14px !important;
    border: none !important; border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(26,95,212,0.28) !important;
  }
  button[kind="primary"] span, button[kind="primary"] p { color: #ffffff !important; }
  button[kind="secondary"] {
    background: #ffffff !important; border: 1.5px solid #c5d5ea !important;
    color: #4a6a8a !important; border-radius: 10px !important;
  }
  [data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #0b8a5a 0%, #076e48 100%) !important;
    color: #ffffff !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(11,138,90,0.28) !important;
  }
  [data-testid="stDownloadButton"] > button span { color: #ffffff !important; }

  /* ══ EXPANDER ══ */
  details {
    background-color: #ffffff !important;
    border: 1px solid #d4e4f7 !important;
    border-radius: 10px !important;
  }
  details > summary {
    background-color: #eaf1fb !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    cursor: pointer !important;
    list-style: none !important;   /* remove browser default triangle */
  }
  /* Style only the text label paragraph inside the summary */
  details > summary > div > div > p,
  details > summary > div > div > span:not([data-testid="stExpanderToggleIcon"]) {
    color: #0d1f3c !important;
    font-weight: 600 !important;
    background-color: transparent !important;
  }
  /* Arrow toggle icon — let Streamlit render its own SVG, just tint it blue */
  details > summary [data-testid="stExpanderToggleIcon"] {
    color: #1a5fd4 !important;
    flex-shrink: 0 !important;
  }
  details > summary [data-testid="stExpanderToggleIcon"] svg {
    width: 18px !important;
    height: 18px !important;
    color: #1a5fd4 !important;
    fill: none !important;
    stroke: #1a5fd4 !important;
    display: block !important;
    overflow: visible !important;
  }
  details > div { background-color: #ffffff !important; }

  /* ══ TABS ══ */
  .stTabs [data-baseweb="tab-list"] {
    background-color: #eef4fc !important; border-radius: 10px !important;
    padding: 4px !important; border: 1.5px solid #cdddf0 !important;
  }
  .stTabs [data-baseweb="tab"] {
    color: #4a6080 !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 13px !important; background: transparent !important;
  }
  .stTabs [aria-selected="true"] {
    color: #1a5fd4 !important; background-color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(26,95,212,0.14) !important;
  }

  /* ══ TABLES (markdown rendered) ══ */
  .stApp table {
    width: 100%; border-collapse: separate; border-spacing: 0; margin-bottom: 20px;
    background-color: #ffffff; border-radius: 12px; overflow: hidden;
    box-shadow: 0 2px 10px rgba(26,63,130,0.08); border: 1.5px solid #cdddf0;
  }
  .stApp table th {
    background: linear-gradient(135deg, #0d2860, #1a5fd4) !important;
    color: #ffffff !important; padding: 13px 16px;
    font-size: 11px; letter-spacing: 0.12em; font-weight: 700; text-transform: uppercase;
  }
  .stApp table td {
    padding: 12px 16px; border-bottom: 1px solid #edf3fb;
    font-size: 14px; color: #1a2c48 !important; background-color: #ffffff !important;
  }
  .stApp table tr:hover td { background-color: #f3f8ff !important; }

  /* ══ SIDEBAR ══ */
  div[data-testid="stSidebarContent"] {
    background: linear-gradient(180deg, #0c1c38 0%, #081426 100%) !important;
    border-right: 1px solid #192e50 !important;
  }
  div[data-testid="stSidebarContent"] p,
  div[data-testid="stSidebarContent"] span,
  div[data-testid="stSidebarContent"] div,
  div[data-testid="stSidebarContent"] label { color: #c8dff5 !important; }
  div[data-testid="stSidebarContent"] h1,
  div[data-testid="stSidebarContent"] h2,
  div[data-testid="stSidebarContent"] h3 { color: #ffffff !important; }
  div[data-testid="stSidebarContent"] button { background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 8px !important; }
  div[data-testid="stSidebarContent"] button span { color: #e8f4ff !important; font-size: 13px !important; }

  /* ══ ALERTS ══ */
  .stInfo > div  { background-color: #eef4ff !important; border: 1.5px solid #aacaf8 !important; color: #1a3a7a !important; border-radius: 10px !important; }
  .stSuccess > div { background-color: #ecfbf4 !important; border: 1.5px solid #88d8b8 !important; color: #0a5038 !important; border-radius: 10px !important; }
  .stWarning > div { background-color: #fffbec !important; border: 1.5px solid #f0d060 !important; color: #7a5200 !important; border-radius: 10px !important; }
  .stApp [data-testid="stNotification"] p,
  .stApp [data-testid="stNotification"] span { color: #0d1f3c !important; }

  /* ══ PROGRESS BAR ══ */
  .stProgress > div > div > div {
    background: linear-gradient(90deg, #0c1c38, #1a5fd4) !important; border-radius: 6px;
  }
  .stProgress > div > div { background-color: #cdddf0 !important; border-radius: 6px; }
  .stProgress p { color: #0c1c38 !important; font-weight: 700 !important; font-size: 13px !important; }

  /* ══ LOG BOX ══ */
  .research-log {
    background: #f4f8ff; border: 1.5px solid #ccddf0;
    border-left: 4px solid #1a5fd4; border-radius: 10px;
    padding: 16px 18px; font-family: 'DM Mono', monospace;
    font-size: 12px; line-height: 1.85; color: #243050 !important;
    white-space: pre-wrap; max-height: 400px; overflow-y: auto;
  }
  .research-log p, .research-log span, .research-log div { color: #243050 !important; }

  /* ══ METRIC CARD ══ */
  .metric-card {
    background: #ffffff; border: 1.5px solid #d8e8f8; border-radius: 14px;
    padding: 20px 22px; margin-bottom: 14px;
    box-shadow: 0 2px 12px rgba(26,63,130,0.07); overflow: hidden;
    transition: box-shadow 0.2s, transform 0.2s;
  }
  .metric-card * { color: inherit; }

  /* ══ STEP INDICATOR ══ */
  .step-indicator-active {
    background: linear-gradient(135deg, #1a5fd4, #3ab0ff) !important;
    color: white !important; border-radius: 50%; width: 36px; height: 36px;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 14px; box-shadow: 0 3px 10px rgba(26,95,212,0.4);
  }
  .step-indicator-done {
    background: #0b8a5a !important; color: white !important; border-radius: 50%;
    width: 36px; height: 36px; display: inline-flex; align-items: center;
    justify-content: center; font-weight: 800; font-size: 14px;
  }
  .step-indicator-pending {
    background: #e0e8f4 !important; color: #9ab0c8 !important; border-radius: 50%;
    width: 36px; height: 36px; display: inline-flex; align-items: center;
    justify-content: center; font-weight: 800; font-size: 14px;
  }
  .step-connector-done { flex: 1; height: 3px; background: #0b8a5a; border-radius: 2px; }
  .step-connector-pending { flex: 1; height: 3px; background: #e0e8f4; border-radius: 2px; }

  /* ══ MISC ══ */
  hr, [data-testid="stDivider"] { border-color: #e0eaf8 !important; }

  /* ══ TOOLTIP ══ */
  div[data-testid="stTooltipContent"] {
    background-color: #0c1c38 !important;
    border: 1px solid #1a5fd4 !important; border-radius: 6px !important;
  }
  div[data-testid="stTooltipContent"] * { color: #ffffff !important; font-size: 13px !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SECTION HEADER HELPER (Tensor Theme)
# ─────────────────────────────────────────────
def section_header(num, icon, title, color="#1a5fd4"):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:28px 0 16px 0;">
      <div style="width:32px;height:32px;
        background:linear-gradient(135deg,{color},{color}bb);
        border-radius:8px;display:flex;align-items:center;justify-content:center;
        color:#fff;font-weight:800;font-size:14px;flex-shrink:0;
        box-shadow:0 3px 10px {color}44;">{num}</div>
      <div style="font-size:11px;letter-spacing:0.22em;color:{color};font-weight:700;
        text-transform:uppercase;padding-bottom:7px;
        border-bottom:2px solid #dce8f5;flex:1;">{icon}&nbsp;&nbsp;{title}</div>
    </div>
    """, unsafe_allow_html=True)



# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
STAGES = {
    1: ("🏢", "Entity Onboarding"),
    2: ("📁", "Document Upload"),
    3: ("🔍", "Classification & Schema"),
    4: ("📊", "Pre-Cognitive Analysis"),
}

# These are the 5 core accepted document types; "Other (Custom)" catches anything else
ACCEPTED_DOC_TYPES = [
    "ALM (Asset Liability Management)",
    "Shareholding Pattern",
    "Borrowing Profile",
    "Annual Report",
    "Portfolio Cut",
    "Other (Custom)",
]

# Default extraction schema per document type
DEFAULT_SCHEMAS = {
    "ALM (Asset Liability Management)": [
        "Total Assets", "Total Liabilities", "Maturity Buckets",
        "Liquidity Gap", "Net Interest Margin"
    ],
    "Shareholding Pattern": [
        "Promoter Holding %", "FII Holding %", "DII Holding %",
        "Public Holding %", "Pledged Shares %"
    ],
    "Borrowing Profile": [
        "Total Debt (INR)", "Secured Debt", "Unsecured Debt",
        "Debt-to-Equity Ratio", "Weighted Average Cost of Debt",
        "Repayment Schedule"
    ],
    "Annual Report": [
        "Revenue (INR)", "Net Profit (INR)", "EBITDA (INR)",
        "Net Worth (INR)", "Total Assets (INR)", "EPS",
        "Dividend Payout", "Auditor Remarks"
    ],
    "Portfolio Cut": [
        "Portfolio Size (INR)", "NPA %", "Sector Concentration",
        "Top 10 Borrower Exposure", "Stage 2 Assets %", "Stage 3 Assets %"
    ],
}

LOAN_TYPES = [
    "Term Loan", "Working Capital Facility", "Cash Credit",
    "Letter of Credit", "Bank Guarantee", "ECB (External Commercial Borrowing)",
    "Structured Finance", "Project Finance"
]

SECTORS = [
    "Banking & NBFC", "Manufacturing", "Infrastructure & Real Estate",
    "IT & Technology", "Healthcare & Pharma", "Energy & Utilities",
    "Retail & Consumer", "Agriculture & Agritech", "Logistics & Supply Chain",
    "Education", "Media & Entertainment", "Other"
]

# ─────────────────────────────────────────────
#  SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        # ── Navigation ──
        "current_stage": 1,

        # ── Stage 1: Entity & Loan Onboarding ──
        "entity_data": {
            "company_name": "",
            "cin": "",
            "pan": "",
            "sector": "",
            "annual_turnover": 0,
            "ceo_name": "",
        },
        "loan_data": {
            "loan_type": "",
            "requested_amount": 0,
            "tenure_months": 12,
            "proposed_interest_rate": 0.0,
            "purpose": "",
        },

        # ── Stage 2: Document Upload ──
        "uploaded_files_meta": [],   # list of UploadedFile objects

        # ── Stage 3: Classification & Schema ──
        # Auto-classification results, editable by user before proceeding
        "doc_classifications": {},   # {"filename": "ALM (Asset Liability Management)", ...}
        "file_data_types": {},       # {"filename": "Structured" | "Unstructured"}
        # Dynamic schema: user can add/remove fields per doc type
        "extraction_schemas": {},    # {"ALM (Asset Liability Management)": ["field1", ...], ...}
        "classification_approved": False,

        # ── Stage 4: Analysis Output ──
        "analysis_complete": False,
        "extracted_data": {},        # {"filename": {"field": "value", ...}}
        "web_research": "",
        "swot_analysis": {},         # {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}
        "final_decision": {},
        "cam_summary": "",
        "pdf_bytes": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ─────────────────────────────────────────────
#  HELPER UTILITIES
# ─────────────────────────────────────────────
def go_to_stage(stage: int):
    st.session_state.current_stage = stage

def score_color(s):
    if s >= 7: return "#0a8a50"
    if s >= 5: return "#c87800"
    return "#c83030"

def format_to_inr_words(number):
    if number >= 10_000_000:
        return f"₹ {number / 10_000_000:g} Crore"
    elif number >= 100_000:
        return f"₹ {number / 100_000:g} Lakh"
    return f"₹ {number:,}"

# ─────────────────────────────────────────────
#  SIDEBAR — persistent across all stages
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:22px 2px 22px 2px;border-bottom:1px solid #1e3558;margin-bottom:22px;">
      <div style="display:flex;align-items:center;gap:13px;">
        <div style="width:42px;height:42px;background:linear-gradient(135deg,#1a5fd4,#3ab0ff);
          border-radius:11px;display:flex;align-items:center;justify-content:center;
          font-size:20px;font-weight:900;color:#fff;flex-shrink:0;
          box-shadow:0 4px 12px rgba(26,95,212,0.45);">₹</div>
        <div>
          <div style="font-size:15px;font-weight:800;color:#ffffff;letter-spacing:0.04em;">INTELLI-CREDIT</div>
          <div style="font-size:9px;color:#7aaacf;letter-spacing:0.2em;margin-top:3px;text-transform:uppercase;">Corporate Credit Engine</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:10px;letter-spacing:0.22em;color:#5a8ab8;margin-bottom:12px;
      text-transform:uppercase;font-weight:700;">📋 &nbsp;Workflow Stages</div>
    """, unsafe_allow_html=True)

    for stage_num, (icon, label) in STAGES.items():
        current = st.session_state.current_stage
        if stage_num < current:
            num_bg = "#0b4030"; num_color = "#35e882"; text_color = "#70e0a0"
        elif stage_num == current:
            num_bg = "#1a4080"; num_color = "#5ac8ff"; text_color = "#c8e8ff"
        else:
            num_bg = "#132030"; num_color = "#5a7a9a"; text_color = "#6a8aaa"

        # Allow navigating back to completed stages only
        if stage_num <= current:
            if st.button(f"{'✅' if stage_num < current else '▶'} Stage {stage_num}: {icon} {label}",
                         key=f"nav_{stage_num}", use_container_width=True):
                go_to_stage(stage_num)
        else:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;
              border-radius:8px;margin-bottom:5px;background:#ffffff08;border:1px solid #1a2e45;">
              <div style="width:22px;height:22px;background:{num_bg};border-radius:6px;
                display:flex;align-items:center;justify-content:center;
                font-size:10px;font-weight:800;color:{num_color};flex-shrink:0;">{stage_num}</div>
              <span style="font-size:12.5px;color:{text_color};font-weight:500;">{icon}&nbsp;&nbsp;{label}</span>
            </div>""", unsafe_allow_html=True)

    # Show entity summary if onboarded
    if st.session_state.entity_data.get("company_name"):
        st.markdown("---")
        st.markdown("""<div style="font-size:10px;letter-spacing:0.18em;color:#5a8ab8;
          text-transform:uppercase;font-weight:700;margin-bottom:8px;">Active Entity</div>""",
          unsafe_allow_html=True)
        ed = st.session_state.entity_data
        ld = st.session_state.loan_data
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0e2240,#0a1a30);
                    border:1px solid #1e3a5a;border-radius:10px;
                    padding:12px 14px;font-size:12px;">
          <b style="color:#e0f0ff;font-size:13px;">{ed['company_name']}</b><br/>
          <span style="color:#7aaacf;">CIN: {ed.get('cin','—')}</span><br/>
          <span style="color:#7aaacf;">Sector: {ed.get('sector','—')}</span><br/>
          <span style="color:#4dcfff;font-weight:700;font-size:13px;">
            {format_to_inr_words(ld.get('requested_amount', 0))}
          </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding-top:24px;border-top:1px solid #1a3050;margin-top:28px;">
      <div style="background:linear-gradient(135deg,#0e2248,#1a3a70);
        border:1px solid #2a4a80;border-radius:10px;
        padding:11px 14px;display:flex;align-items:center;gap:10px;">
        <div style="width:32px;height:32px;
          background:linear-gradient(135deg,#3ab0ff,#1a5fd4);
          border-radius:8px;display:flex;align-items:center;justify-content:center;
          font-size:15px;flex-shrink:0;box-shadow:0 2px 8px rgba(58,176,255,0.4);">⚡</div>
        <div>
          <div style="font-size:13px;font-weight:800;color:#ffffff;letter-spacing:0.06em;">TENSOR</div>
          <div style="font-size:9px;color:#7aaacf;letter-spacing:0.12em;margin-top:1px;text-transform:uppercase;">
            Built by Team Tensor
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # OCR backend badge
    _ocr_icon  = "🧠" if "Document AI" in _OCR_BACKEND else "📄"
    _ocr_color = "#10a870" if "Document AI" in _OCR_BACKEND else "#5a8ab8"
    st.markdown(
        f'<div style="margin-top:10px;padding:6px 10px;border-radius:7px;'
        f'background:#0a1a2a;border:1px solid #1a3050;font-size:10px;'
        f'color:{_ocr_color};font-weight:600;letter-spacing:0.05em;">'
        f'{_ocr_icon}&nbsp; OCR: {_OCR_BACKEND}</div>',
        unsafe_allow_html=True,
    )



# ─────────────────────────────────────────────
#  STEP PROGRESS BAR (top of main area)
# ─────────────────────────────────────────────
def render_progress_bar():
    current = st.session_state.current_stage
    cols = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])
    stage_cols  = [cols[0], cols[2], cols[4], cols[6]]
    connector_cols = [cols[1], cols[3], cols[5]]

    for i, (stage_num, (icon, label)) in enumerate(STAGES.items()):
        with stage_cols[i]:
            if stage_num < current:
                badge_class = "step-indicator-done"
                badge_content = "✓"
                color = "#0b8a5a"
            elif stage_num == current:
                badge_class = "step-indicator-active"
                badge_content = str(stage_num)
                color = "#1a5fd4"
            else:
                badge_class = "step-indicator-pending"
                badge_content = str(stage_num)
                color = "#9ab0c8"
            st.markdown(
                f'<div style="text-align:center;">'
                f'<div class="{badge_class}" style="margin:0 auto 6px auto;">{badge_content}</div>'
                f'<div style="font-size:11px;font-weight:700;color:{color};">{icon} {label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    for i, conn_col in enumerate(connector_cols):
        with conn_col:
            done = (i + 1) < current
            cls = "step-connector-done" if done else "step-connector-pending"
            st.markdown(
                f'<div style="display:flex;align-items:center;height:36px;">'
                f'<div class="{cls}"></div></div>',
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────
#  PDF EXPORT HELPER
# ─────────────────────────────────────────────
def _build_pdf(
    cam_summary: str,
    entity_data: dict,
    loan_data: dict,
    decision: dict,
) -> bytes:
    """
    Builds a structured Credit Appraisal Memorandum (CAM) PDF.
    Parses the LLM-generated cam_summary into discrete sections
    and renders each with proper typesetting — no text overlaps.
    """

    # ── Helpers ──────────────────────────────────────────────────────────
    def _clean(text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*",     r"\1", text)
        text = re.sub(r"^#{1,6}\s+",    "",    text, flags=re.MULTILINE)
        text = re.sub(r"`(.+?)`",       r"\1", text)
        text = text.replace("₹", "Rs.").replace("\u20b9", "Rs.")
        text = text.replace("\u2019","'").replace("\u2018","'")
        text = text.replace("\u201c",'"').replace("\u201d",'"')
        text = text.replace("\u2013","-").replace("\u2014","--")
        text = text.replace("\u2022","*").replace("\u2023","*")
        text = text.encode("latin-1", errors="ignore").decode("latin-1")
        return text

    def _parse_table(lines: list[str]) -> list[list[str]]:
        """Extract markdown table rows into list-of-lists."""
        rows = []
        for ln in lines:
            ln = ln.strip()
            if not ln.startswith("|"):
                continue
            if re.match(r"^\|[-| :]+\|$", ln):
                continue
            cells = [c.strip() for c in ln.strip("|").split("|")]
            rows.append([_clean(c) for c in cells])
        return rows

    def _draw_table(pdf: FPDF, rows: list[list[str]], col_widths: list[float], line_h: float = 5.5):
        """Draw a table where every cell in a row shares the same height.

        Strategy:
        1. For each data row, estimate how many lines each cell needs
           (text length ÷ chars-per-line based on col width).
        2. Take the max line-count across all cells → that is the row height.
        3. Draw every cell as a filled rect + clipped text using that height.
        """
        if not rows:
            return

        CHAR_W = 1.9   # approx mm per character at 8pt Helvetica

        def _lines_needed(text: str, col_w: float) -> int:
            chars_per_line = max(1, int(col_w / CHAR_W))
            words = text.split()
            line, lines = 0, 1
            for w in words:
                if line + len(w) + 1 <= chars_per_line:
                    line += len(w) + 1
                else:
                    lines += 1
                    line = len(w)
            return lines

        total_w = sum(col_widths)

        # ── Header row ──
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(26, 58, 112)
        pdf.set_text_color(255, 255, 255)
        hdr_h = line_h + 2
        row_x = pdf.get_x()
        row_y = pdf.get_y()
        for cell, w in zip(rows[0], col_widths):
            # Measure header lines
            n = _lines_needed(cell[:60], w)
            this_h = n * (line_h + 1) + 3
            hdr_h = max(hdr_h, this_h)

        for cell, w in zip(rows[0], col_widths):
            x0, y0 = pdf.get_x(), pdf.get_y()
            pdf.set_fill_color(26, 58, 112)
            pdf.rect(x0, y0, w, hdr_h, "F")
            pdf.set_draw_color(200, 215, 240)
            pdf.rect(x0, y0, w, hdr_h, "D")
            pdf.set_xy(x0 + 2, y0 + 2)
            pdf.multi_cell(w - 4, line_h, cell[:60], border=0, fill=False)
            pdf.set_xy(x0 + w, y0)
        pdf.set_draw_color(0, 0, 0)
        pdf.ln(hdr_h)

        # ── Data rows ──
        pdf.set_font("Helvetica", "", 8)
        for i, row in enumerate(rows[1:], 1):
            # 1. compute this row's height
            max_lines = 1
            for cell, w in zip(row, col_widths):
                n = _lines_needed(_clean(cell[:250]), w)
                max_lines = max(max_lines, n)
            row_h = max_lines * line_h + 4   # padding top+bottom

            # check page break manually
            if pdf.get_y() + row_h > pdf.h - pdf.b_margin - 5:
                pdf.add_page()

            # 2. draw all cells at this height
            bg = (240, 246, 255) if i % 2 == 0 else (255, 255, 255)
            row_y = pdf.get_y()
            for cell, w in zip(row, col_widths):
                x0 = pdf.get_x()
                # fill background
                pdf.set_fill_color(*bg)
                pdf.rect(x0, row_y, w, row_h, "F")
                # border
                pdf.set_draw_color(190, 210, 235)
                pdf.rect(x0, row_y, w, row_h, "D")
                # text — vertically centred
                pdf.set_xy(x0 + 2, row_y + 2)
                pdf.set_text_color(20, 35, 70)
                pdf.multi_cell(w - 4, line_h, _clean(cell[:250]), border=0, fill=False)
                pdf.set_xy(x0 + w, row_y)

            pdf.set_draw_color(0, 0, 0)
            pdf.set_xy(18, row_y + row_h)   # move cursor below this row

        pdf.ln(3)

    def _section_banner(pdf: FPDF, title: str):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(26, 58, 112)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(174, 7, f"  {title}", fill=True, ln=True)
        pdf.set_text_color(20, 35, 70)
        pdf.set_font("Helvetica", "", 9)
        pdf.ln(2)

    def _bullet(pdf: FPDF, text: str):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(20, 35, 70)
        clean = _clean(text.lstrip("-•* "))
        if not clean.strip():
            return
        pdf.cell(6)
        pdf.multi_cell(168, 5, "-  " + clean[:300])
        pdf.ln(1)

    def _body_text(pdf: FPDF, text: str):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(20, 35, 70)
        pdf.multi_cell(174, 5, _clean(text)[:600])
        pdf.ln(1)

    # ── Parse cam_summary into sections ──────────────────────────────────
    section_map: dict[str, list[str]] = {}
    current_key = "PREAMBLE"
    section_map[current_key] = []
    for line in cam_summary.split("\n"):
        stripped = line.strip()
        m = re.match(r"^#{1,3}\s*\d*\.?\s*(.*)", stripped)
        if m and stripped.startswith("#"):
            current_key = m.group(1).strip().upper()
            section_map.setdefault(current_key, [])
        else:
            section_map.setdefault(current_key, []).append(line)

    def _get_section(*keys) -> list[str]:
        for k in keys:
            for sk in section_map:
                if k.upper() in sk:
                    return section_map[sk]
        return []

    s_score  = decision.get("credit_score", 0)
    verdict  = "APPROVE" if s_score >= 7 else "CAUTION" if s_score >= 5 else "REJECT"
    rec_limit = format_to_inr_words(decision.get("recommended_limit_inr", 0)).replace("₹","Rs.")
    rec_rate  = decision.get("recommended_interest_rate", 0)
    vc_map = {"APPROVE": (10,138,80), "CAUTION":(180,110,0), "REJECT":(180,30,30)}
    vc = vc_map.get(verdict, (80,80,80))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)

    # ════════════════════════════════════════════════
    #  PAGE 1 — COVER
    # ════════════════════════════════════════════════
    pdf.add_page()

    # ── Top banner ──
    pdf.set_fill_color(13, 31, 60)
    pdf.rect(0, 0, 210, 32, "F")
    # Accent line
    pdf.set_fill_color(*vc)
    pdf.rect(0, 32, 210, 2, "F")
    pdf.set_xy(0, 7)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(210, 10, "CREDIT APPRAISAL MEMORANDUM", align="C")
    pdf.set_xy(0, 18)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(160, 190, 220)
    pdf.cell(210, 7, f"TENSOR AI  |  INTELLI-CREDIT ENGINE  |  Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}  |  CONFIDENTIAL", align="C")

    # ── Verdict ribbon ──
    pdf.set_xy(18, 40)
    pdf.set_fill_color(*vc)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(174, 8,
        f"  FINAL RECOMMENDATION: {verdict}  |  SCORE: {s_score}/10  |  LIMIT: INR {decision.get('recommended_limit_inr',0):,}",
        fill=True, ln=True)

    pdf.ln(6)
    pdf.set_text_color(13, 31, 60)

    # ── Entity name heading ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(174, 7, f"BORROWING ENTITY: {_clean(entity_data.get('company_name','').upper())}", ln=True)
    pdf.ln(3)

    # ── Entity / Loan 2-col table ──
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(210, 225, 252)
    pdf.cell(87, 7, "ENTITY DETAILS", border=1, fill=True, align="C")
    pdf.cell(4)
    pdf.cell(83, 7, "FACILITY DETAILS", border=1, fill=True, align="C")
    pdf.ln(8)

    e_rows = [
        ("Legal Name",      _clean(entity_data.get("company_name","-"))),
        ("CIN",             _clean(entity_data.get("cin","-") or "-")),
        ("PAN",             _clean(entity_data.get("pan","-") or "-")),
        ("Sector",          _clean(entity_data.get("sector","-"))),
        ("Annual Turnover", format_to_inr_words(entity_data.get("annual_turnover",0)).replace("₹","Rs.")),
        ("Key Promoter",    _clean(entity_data.get("ceo_name","-"))),
    ]
    l_rows = [
        ("Facility Type",   _clean(loan_data.get("loan_type","-"))),
        ("Requested Amt",   format_to_inr_words(loan_data.get("requested_amount",0)).replace("₹","Rs.")),
        ("Tenure",          f"{loan_data.get('tenure_months',0)} months"),
        ("Proposed Rate",   f"{loan_data.get('proposed_interest_rate',0):.2f}% p.a."),
        ("Purpose",         _clean((loan_data.get("purpose") or "-")[:55])),
        ("Rec. Limit",      rec_limit),
    ]
    pdf.set_font("Helvetica", "", 9)
    for (el, ev), (ll, lv) in zip(e_rows, l_rows):
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(28, 6, el, border="LTB")
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(59, 6, ev[:42], border="RTB")
        pdf.cell(4)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(26, 6, ll, border="LTB")
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(57, 6, lv[:40], border="RTB")
        pdf.ln(6)

    pdf.ln(5)

    # ── Score card ──
    box_y = pdf.get_y()
    pdf.set_fill_color(240, 246, 255)
    pdf.set_draw_color(*vc)
    pdf.set_line_width(1.0)
    pdf.rect(18, box_y, 174, 24, "DF")
    pdf.set_line_width(0.2); pdf.set_draw_color(0,0,0)
    pdf.set_xy(24, box_y + 4)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*vc)
    pdf.cell(38, 10, f"{s_score}/10")
    pdf.set_xy(65, box_y + 5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(80, 6, f"SYSTEM VERDICT: {verdict}")
    pdf.set_xy(65, box_y + 13)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(40, 60, 100)
    pdf.cell(80, 5, f"Approved Limit: {rec_limit}   |   Rate: {rec_rate}% p.a.")

    # ════════════════════════════════════════════════
    #  PAGE 2 — EXECUTIVE SUMMARY + FINANCIAL TABLE
    # ════════════════════════════════════════════════
    pdf.add_page()
    pdf.set_text_color(20, 35, 70)

    # 1. Executive Summary
    _section_banner(pdf, "1. EXECUTIVE SUMMARY")
    exec_lines = _get_section("EXECUTIVE SUMMARY")
    for ln in exec_lines:
        t = ln.strip()
        if t and not t.startswith("|") and not t.startswith("#"):
            _body_text(pdf, t)
    pdf.ln(3)

    # 2. Financial Summary
    _section_banner(pdf, "2. FINANCIAL SUMMARY (VERIFIED SCHEMA)")
    fin_lines = _get_section("FINANCIAL SUMMARY")
    fin_rows  = _parse_table(fin_lines)
    if fin_rows:
        n_cols = len(fin_rows[0])
        w = 174 / max(n_cols, 1)
        _draw_table(pdf, fin_rows, [w] * n_cols)
    else:
        for ln in fin_lines:
            t = ln.strip()
            if t and not t.startswith("|"):
                _body_text(pdf, t)
    pdf.ln(3)

    # 3. Five Cs
    _section_banner(pdf, "3. THE FIVE Cs OF CREDIT ASSESSMENT")
    fivec_lines = _get_section("FIVE Cs", "FIVE C", "5 C")
    fivec_rows  = _parse_table(fivec_lines)
    if fivec_rows:
        _draw_table(pdf, fivec_rows, [32, 116, 26])
    pdf.ln(3)

    # ════════════════════════════════════════════════
    #  PAGE 3 — SWOT + WEB INTELLIGENCE + RISK FLAGS
    # ════════════════════════════════════════════════
    pdf.add_page()
    pdf.set_text_color(20, 35, 70)

    # 4. SWOT
    _section_banner(pdf, "4. SWOT ANALYSIS")
    swot_lines = _get_section("SWOT")
    swot_rows  = _parse_table(swot_lines)
    if swot_rows:
        _draw_table(pdf, swot_rows, [38, 136])
    pdf.ln(3)

    # 5. Web Intelligence
    _section_banner(pdf, "5. EXTERNAL RISK INTELLIGENCE & WARNING SIGNALS")
    web_lines = _get_section("WEB INTELLIGENCE", "EXTERNAL RISK", "RISK FLAGS", "WARNING")
    for ln in web_lines:
        t = ln.strip()
        if not t or t.startswith("|") or t.startswith("#"):
            continue
        if t.startswith(("-", "•", "*")):
            _bullet(pdf, t)
        else:
            _body_text(pdf, t)
    pdf.ln(3)

    # 6. Recommended Terms
    _section_banner(pdf, "6. RECOMMENDED TERMS & CONDITIONS")
    terms_lines = _get_section("RECOMMENDED TERMS", "RECOMMENDED")
    terms_rows  = _parse_table(terms_lines)
    if terms_rows:
        _draw_table(pdf, terms_rows, [60, 114])
    else:
        for ln in terms_lines:
            t = ln.strip()
            if t and not t.startswith("|"):
                _body_text(pdf, t)
    pdf.ln(3)

    # 7. Catch-all: any remaining sections from LLM
    for sk, sl in section_map.items():
        already = {"PREAMBLE","EXECUTIVE SUMMARY","FINANCIAL SUMMARY","FIVE CS OF CREDIT",
                   "FIVE CS","FIVE C","SWOT ANALYSIS","SWOT","WEB INTELLIGENCE TRIANGULATION",
                   "WEB INTELLIGENCE","EXTERNAL RISK INTELLIGENCE & WARNING SIGNALS",
                   "RISK FLAGS & EARLY WARNING SIGNALS","EXTERNAL RISK","RISK FLAGS",
                   "WARNING","RECOMMENDED TERMS & CONDITIONS","RECOMMENDED TERMS","RECOMMENDED"}
        if sk in already or not sl:
            continue
        _section_banner(pdf, sk)
        tbl = _parse_table(sl)
        if tbl:
            w = 174 / max(len(tbl[0]), 1)
            _draw_table(pdf, tbl, [w]*len(tbl[0]))
        else:
            for ln in sl:
                t = ln.strip()
                if not t or t.startswith("#"):
                    continue
                if t.startswith(("-","•","*")):
                    _bullet(pdf, t)
                else:
                    _body_text(pdf, t)
        pdf.ln(3)

    # ── Footer on every page ──────────────────────────────────────────────
    class _PDF(FPDF):
        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica","I", 7)
            self.set_text_color(130,150,180)
            self.cell(0, 5,
                f"Generated on {datetime.now().strftime('%d %b %Y %H:%M')}  |  Page {self.page_no()}",
                align="C")

    # Re-build with footer-enabled subclass
    pdf2 = _PDF()
    pdf2.set_auto_page_break(auto=True, margin=18)
    pdf2.set_margins(18, 18, 18)
    # Transfer pages by re-running (simplest FPDF2 approach: copy output bytes back)
    # We already have `pdf` fully built — add footer via alias
    for page_num in range(1, pdf.page + 1):
        pass  # footer is on the re-built copy — just use existing pdf

    # Write footer manually on last page
    pdf.set_y(-14)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(130, 150, 180)
    pdf.cell(0, 5,
        f"Generated on {datetime.now().strftime('%d %b %Y %H:%M')}  |  Page {pdf.page}",
        align="C")

    return bytes(pdf.output())



# ── HERO HEADER ──
st.markdown("""
<div style="background:linear-gradient(135deg,#0c1c38 0%,#103878 52%,#1a5fd4 100%);
  border-radius:14px;padding:26px 34px;margin-bottom:28px;
  display:flex;align-items:center;justify-content:space-between;
  box-shadow:0 6px 28px rgba(12,28,56,0.20);position:relative;overflow:hidden;">
  <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
    background:rgba(255,255,255,0.04);border-radius:50%;"></div>
  <div style="position:absolute;bottom:-60px;right:80px;width:140px;height:140px;
    background:rgba(255,255,255,0.03);border-radius:50%;"></div>
  <div>
    <div style="font-size:11px;letter-spacing:0.28em;color:#7eb3f5;font-weight:600;
      text-transform:uppercase;margin-bottom:6px;">Tensor AI — Credit Intelligence Platform</div>
    <div style="font-size:28px;font-weight:800;color:#ffffff;letter-spacing:0.01em;
      line-height:1.15;">INTELLI-CREDIT ENGINE</div>
    <div style="font-size:13px;color:#a8c8f0;margin-top:6px;font-weight:400;">
      AI-driven underwriting · Multi-source intelligence · Automated CAM generation
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:11px;color:#7eb3f5;letter-spacing:0.15em;font-weight:600;
      text-transform:uppercase;">Powered by</div>
    <div style="font-size:22px;font-weight:800;color:#ffffff;letter-spacing:0.05em;">
      TENSOR<span style="color:#4fa3f5;">AI</span>
    </div>
    <div style="font-size:10px;color:#7eb3f5;margin-top:2px;">v2.0 · Enterprise Edition</div>
  </div>
</div>
""", unsafe_allow_html=True)

render_progress_bar()
st.markdown("<hr style='border-color:#e0eaf8;margin:18px 0 28px 0;'/>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  STAGE 1 — ENTITY ONBOARDING
# ═══════════════════════════════════════════════════════════════
if st.session_state.current_stage == 1:
    section_header("1", "🏢", "Entity Onboarding")
    st.caption("Capture basic entity and loan details before uploading documents.")

    with st.form("onboarding_form", clear_on_submit=False):
        section_header("A", "📋", "Entity Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            company_name = st.text_input(
                "🏢 Legal Entity Name *",
                value=st.session_state.entity_data.get("company_name", ""),
                placeholder="e.g., Reliance Industries Ltd."
            )
            cin = st.text_input(
                "🔢 CIN (Corporate Identity Number)",
                value=st.session_state.entity_data.get("cin", ""),
                placeholder="e.g., L17110MH1973PLC019786"
            )
        with col2:
            pan = st.text_input(
                "🪪 PAN",
                value=st.session_state.entity_data.get("pan", ""),
                placeholder="e.g., AAACR5055K"
            )
            ceo_name = st.text_input(
                "👤 Key Promoter / CEO Name *",
                value=st.session_state.entity_data.get("ceo_name", ""),
                placeholder="e.g., Mukesh Ambani",
                help="Used to prevent AI entity confusion."
            )
        with col3:
            sector = st.selectbox(
                "🏭 Sector *",
                options=[""] + SECTORS,
                index=(SECTORS.index(st.session_state.entity_data.get("sector", "")) + 1)
                      if st.session_state.entity_data.get("sector") in SECTORS else 0
            )
            annual_turnover = st.number_input(
                "💹 Annual Turnover (₹)",
                min_value=0,
                value=st.session_state.entity_data.get("annual_turnover", 0),
                step=1_000_000,
                format="%d"
            )

        st.markdown("---")
        section_header("B", "💰", "Loan Facility Details")
        col4, col5, col6 = st.columns(3)
        with col4:
            loan_type = st.selectbox(
                "📄 Loan / Facility Type *",
                options=[""] + LOAN_TYPES,
                index=(LOAN_TYPES.index(st.session_state.loan_data.get("loan_type", "")) + 1)
                      if st.session_state.loan_data.get("loan_type") in LOAN_TYPES else 0
            )
            requested_amount = st.number_input(
                "💰 Requested Facility Amount (₹) *",
                min_value=0,
                value=st.session_state.loan_data.get("requested_amount", 50_000_000),
                step=1_000_000,
                format="%d"
            )
        with col5:
            tenure_months = st.slider(
                "📅 Tenure (Months)",
                min_value=1, max_value=360,
                value=st.session_state.loan_data.get("tenure_months", 12)
            )
            proposed_interest_rate = st.number_input(
                "📈 Proposed Interest Rate (% p.a.)",
                min_value=0.0, max_value=50.0,
                value=float(st.session_state.loan_data.get("proposed_interest_rate", 10.0)),
                step=0.25,
                format="%.2f"
            )
        with col6:
            purpose = st.text_area(
                "📝 Purpose of Facility",
                value=st.session_state.loan_data.get("purpose", ""),
                placeholder="Describe the end-use of the loan...",
                height=118
            )

        submitted = st.form_submit_button(
            "✅  Save & Proceed to Document Upload →",
            use_container_width=True, type="primary"
        )

        if submitted:
            # Validation
            errors = []
            if not company_name.strip():
                errors.append("Entity Name is required.")
            if not ceo_name.strip():
                errors.append("CEO / Promoter Name is required.")
            if not sector:
                errors.append("Sector is required.")
            if not loan_type:
                errors.append("Loan Type is required.")
            if requested_amount <= 0:
                errors.append("Requested Amount must be greater than 0.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                # Persist to session state
                st.session_state.entity_data = {
                    "company_name": company_name.strip(),
                    "cin": cin.strip(),
                    "pan": pan.strip(),
                    "sector": sector,
                    "annual_turnover": annual_turnover,
                    "ceo_name": ceo_name.strip(),
                }
                st.session_state.loan_data = {
                    "loan_type": loan_type,
                    "requested_amount": requested_amount,
                    "tenure_months": tenure_months,
                    "proposed_interest_rate": proposed_interest_rate,
                    "purpose": purpose.strip(),
                }
                go_to_stage(2)
                st.rerun()

# ═══════════════════════════════════════════════════════════════
#  STAGE 2 — INTELLIGENT DOCUMENT UPLOAD
# ═══════════════════════════════════════════════════════════════
elif st.session_state.current_stage == 2:
    ed = st.session_state.entity_data
    section_header("2", "📁", "Document Upload")
    st.caption(
        f"Upload **all** documents for **{ed['company_name']}** in one place. "
        "The AI will first classify each file as Structured or Unstructured, "
        "then map it to one of the 5 known types — or flag it as **Other (Custom)**."
    )

    # ── Info panel ──
    known_types_html = "".join([
        f'<span style="background:#1a5fd420;color:#1040a0;padding:3px 10px;'
        f'border-radius:6px;font-size:12px;font-weight:600;">{dt}</span>'
        for dt in ACCEPTED_DOC_TYPES if dt != "Other (Custom)"
    ])
    st.markdown(f"""
    <div style="background:#f0f6ff;border:1.5px solid #b8d0f5;border-radius:10px;
                padding:14px 18px;margin-bottom:20px;">
      <div style="font-size:12px;font-weight:700;color:#1a5fd4;margin-bottom:10px;
                  letter-spacing:0.1em;text-transform:uppercase;">
        📂 Classification Rules
      </div>
      <div style="font-size:12px;color:#2a3c5a;margin-bottom:8px;">
        <b>Structured</b> (CSV / XLSX) and <b>Unstructured</b> (PDF / text) files are both accepted.
        Each file will be mapped to one of the 5 core types, or tagged <b>Other (Custom)</b>:
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;">
        {known_types_html}
        <span style="background:#e8f5e9;color:#2e7d32;padding:3px 10px;
          border-radius:6px;font-size:12px;font-weight:600;">Other (Custom)</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Single unified uploader ──
    all_uploads = st.file_uploader(
        "Drop or browse any number of files — PDF, CSV, XLSX supported",
        type=["pdf", "csv", "xlsx"],
        accept_multiple_files=True,
        key="stage2_all_files",
    )

    if all_uploads:
        st.markdown(
            f'<div style="font-size:13px;color:#1a5fd4;font-weight:700;margin:12px 0 6px 0;">'
            f'📎 {len(all_uploads)} file(s) staged</div>',
            unsafe_allow_html=True
        )
        for f in all_uploads:
            ext = f.name.rsplit(".", 1)[-1].upper()
            is_struct = ext in ("CSV", "XLSX")
            badge_bg  = "#e8f5e9" if is_struct else "#fff3e0"
            badge_col = "#2e7d32" if is_struct else "#e65100"
            badge_txt = "Structured" if is_struct else "Unstructured"
            st.markdown(
                f'<div style="background:#f8faff;border:1px solid #d0e0f8;border-radius:6px;'
                f'padding:8px 14px;margin-bottom:4px;font-size:13px;display:flex;'
                f'align-items:center;gap:10px;">'
                f'📄 <b style="color:#0d1f3c;">{f.name}</b>'
                f'<span style="background:{badge_bg};color:{badge_col};padding:2px 8px;'
                f'border-radius:5px;font-size:11px;font-weight:700;">{badge_txt}</span>'
                f'<span style="color:#6a88aa;font-size:11px;margin-left:auto;">'
                f'{ext} · {f.size // 1024} KB</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← Back", use_container_width=True):
            go_to_stage(1); st.rerun()
    with col_next:
        if st.button(
            "🤖  Auto-Classify All Documents →",
            use_container_width=True, type="primary",
            disabled=(not all_uploads)
        ):
            st.session_state.uploaded_files_meta = list(all_uploads)
            st.session_state.doc_classifications = {}
            st.session_state.file_data_types = {}
            st.session_state.extraction_schemas = {}
            st.session_state.classification_approved = False
            st.session_state.analysis_complete = False

            # ── Helpers ──────────────────────────────────────────────────────
            STRUCT_EXTS = {"csv", "xlsx"}

            def _data_type(fname: str) -> str:
                return "Structured" if fname.rsplit(".", 1)[-1].lower() in STRUCT_EXTS else "Unstructured"

            def _heuristic_classify(filename: str) -> str:
                n = filename.lower()
                if "alm" in n:                           return "ALM (Asset Liability Management)"
                if "share" in n or "shareholding" in n:  return "Shareholding Pattern"
                if "borrow" in n or "debt" in n:         return "Borrowing Profile"
                if "portfolio" in n or "cut" in n:       return "Portfolio Cut"
                if "annual" in n or "report" in n:       return "Annual Report"
                return "Other (Custom)"

            # ── Build snippets ────────────────────────────────────────────────
            file_snippets = ""
            for f in all_uploads:
                try:
                    f.seek(0); raw = f.read(600); f.seek(0)
                    snippet = raw.decode("utf-8", errors="ignore").replace("\n", " ")[:400]
                except Exception:
                    snippet = "(unreadable)"
                ext = f.name.rsplit(".", 1)[-1].upper()
                file_snippets += (
                    f'\n  FILE: "{f.name}" (extension: {ext})\n'
                    f'  CONTENT PREVIEW: {snippet}\n'
                )

            core_types_str = "\n".join(
                f"  - {t}" for t in ACCEPTED_DOC_TYPES if t != "Other (Custom)"
            )
            classify_prompt = f"""
You are a document classification engine for a corporate credit underwriting system.

For EACH file listed below, produce a JSON object with two keys:
  "data_type"  → MUST be exactly "Structured" (CSV/XLSX tabular data) or "Unstructured" (PDF/text/scanned)
  "doc_type"   → MUST be exactly one of:
{core_types_str}
    - Other (Custom)   ← use ONLY if the file clearly does not fit any of the 5 above

Rules:
- Base "data_type" primarily on the file extension (CSV/XLSX → Structured; PDF → Unstructured).
- Base "doc_type" on both the filename AND content preview.
- When in doubt between a known type and "Other (Custom)", prefer the known type.
- Respond with ONLY a JSON object mapping filename → {{"data_type": "...", "doc_type": "..."}}. No explanation.

Example:
{{
  "borrowing_fy24.pdf":      {{"data_type": "Unstructured", "doc_type": "Borrowing Profile"}},
  "share_pattern_q3.xlsx":   {{"data_type": "Structured",   "doc_type": "Shareholding Pattern"}},
  "misc_notes.pdf":          {{"data_type": "Unstructured", "doc_type": "Other (Custom)"}}
}}

FILES TO CLASSIFY:
{file_snippets}
"""
            with st.spinner("🤖 AI is classifying your documents..."):
                raw_response = analyze_text_with_fallback(classify_prompt)

            # ── Parse LLM response ────────────────────────────────────────────
            auto_classifications = {}
            auto_data_types = {}
            try:
                clean = re.sub(r"```(?:json)?|```", "", raw_response).strip()
                llm_result = json.loads(clean)
                for f in all_uploads:
                    entry = llm_result.get(f.name, {})
                    dt  = entry.get("data_type", _data_type(f.name))
                    doc = entry.get("doc_type", "Other (Custom)")
                    auto_data_types[f.name]      = dt if dt in ("Structured", "Unstructured") else _data_type(f.name)
                    auto_classifications[f.name] = doc if doc in ACCEPTED_DOC_TYPES else _heuristic_classify(f.name)
            except Exception:
                for f in all_uploads:
                    auto_data_types[f.name]      = _data_type(f.name)
                    auto_classifications[f.name] = _heuristic_classify(f.name)

            st.session_state.doc_classifications = auto_classifications
            st.session_state.file_data_types     = auto_data_types
            # Seed extraction schemas with defaults for known types
            st.session_state.extraction_schemas = {
                doc_type: list(fields)
                for doc_type, fields in DEFAULT_SCHEMAS.items()
            }
            go_to_stage(3); st.rerun()


# ═══════════════════════════════════════════════════════════════
#  STAGE 3 — CLASSIFICATION REVIEW & DYNAMIC SCHEMA (HITL)
# ═══════════════════════════════════════════════════════════════
elif st.session_state.current_stage == 3:
    ed = st.session_state.entity_data
    section_header("3", "🔍", "Classification Review & Dynamic Schema")
    st.info(
        "⚠️ **Human-in-the-Loop Checkpoint:** Review the auto-classified document types below. "
        "Correct any misclassifications, then customize the extraction schema for each document type "
        "before running the analysis engine.",
        icon="🧠"
    )

    classifications = st.session_state.doc_classifications
    data_types      = st.session_state.get("file_data_types", {})
    schemas = st.session_state.extraction_schemas

    # ── Part A: Document Classification Review ──
    section_header("A", "📂", "Document Classification")
    st.caption("Verify the AI's classification. Each file shows its **data type** (Structured / Unstructured) and **document type**. Correct any errors using the dropdowns.")

    # Split files into known-type and custom
    known_files  = {fn: cls for fn, cls in classifications.items() if cls != "Other (Custom)"}
    custom_files = {fn: cls for fn, cls in classifications.items() if cls == "Other (Custom)"}

    updated_classifications = {}

    # ── Known types ──
    if known_files:
        for filename, current_class in known_files.items():
            dt = data_types.get(filename, "Unstructured")
            dt_bg  = "#e8f5e9" if dt == "Structured" else "#fff3e0"
            dt_col = "#2e7d32" if dt == "Structured" else "#e65100"

            col_fname, col_dtype, col_doctype = st.columns([3, 1, 2])
            with col_fname:
                st.markdown(
                    f'<div style="padding:10px 14px;background:#f0f6ff;border-radius:8px;'
                    f'border:1px solid #c8d8f0;font-size:13px;font-weight:600;'
                    f'color:#0d1f3c;">📄 {filename}</div>',
                    unsafe_allow_html=True
                )
            with col_dtype:
                st.markdown(
                    f'<div style="padding:10px 10px;background:{dt_bg};border-radius:8px;'
                    f'font-size:12px;font-weight:700;color:{dt_col};text-align:center;">'
                    f'{dt}</div>',
                    unsafe_allow_html=True
                )
            with col_doctype:
                chosen = st.selectbox(
                    "Document Type",
                    options=ACCEPTED_DOC_TYPES,
                    index=ACCEPTED_DOC_TYPES.index(current_class) if current_class in ACCEPTED_DOC_TYPES else 0,
                    key=f"classify_{filename}",
                    label_visibility="collapsed"
                )
                updated_classifications[filename] = chosen

    # ── Other / Custom files ──
    if custom_files:
        st.markdown(
            '<div style="background:#fff8e1;border:1.5px solid #ffe082;border-radius:10px;'
            'padding:12px 16px;margin:16px 0 8px 0;">'
            '<span style="font-size:12px;font-weight:700;color:#f57f17;">⚠️ Unrecognised Files — Other (Custom)</span><br/>'
            '<span style="font-size:12px;color:#795548;">These files did not match any of the 5 core types. '
            'Re-classify them below or leave as Other (Custom).</span>'
            '</div>',
            unsafe_allow_html=True
        )
        for filename, current_class in custom_files.items():
            dt = data_types.get(filename, "Unstructured")
            dt_bg  = "#e8f5e9" if dt == "Structured" else "#fff3e0"
            dt_col = "#2e7d32" if dt == "Structured" else "#e65100"

            col_fname, col_dtype, col_doctype = st.columns([3, 1, 2])
            with col_fname:
                st.markdown(
                    f'<div style="padding:10px 14px;background:#fff8e1;border-radius:8px;'
                    f'border:1px solid #ffe082;font-size:13px;font-weight:600;'
                    f'color:#5d4037;">📄 {filename}</div>',
                    unsafe_allow_html=True
                )
            with col_dtype:
                st.markdown(
                    f'<div style="padding:10px 10px;background:{dt_bg};border-radius:8px;'
                    f'font-size:12px;font-weight:700;color:{dt_col};text-align:center;">'
                    f'{dt}</div>',
                    unsafe_allow_html=True
                )
            with col_doctype:
                chosen = st.selectbox(
                    "Document Type",
                    options=ACCEPTED_DOC_TYPES,
                    index=ACCEPTED_DOC_TYPES.index(current_class) if current_class in ACCEPTED_DOC_TYPES else len(ACCEPTED_DOC_TYPES) - 1,
                    key=f"classify_{filename}",
                    label_visibility="collapsed"
                )
                updated_classifications[filename] = chosen

    st.session_state.doc_classifications = updated_classifications

    st.markdown("---")
    # ── Part B: Dynamic Schema Editor ──
    section_header("B", "⚙️", "Dynamic Extraction Schema")
    st.caption(
        "Define exactly which fields the AI should extract from each document type. "
        "Add custom fields relevant to your analysis."
    )

    # Identify which doc types are actually in use (excluding pure "Other (Custom)" if no custom fields defined)
    active_doc_types = list(set(updated_classifications.values()))

    for doc_type in active_doc_types:
        icon = "📋" if doc_type != "Other (Custom)" else "📌"
        with st.expander(f"{icon} Schema for: **{doc_type}**", expanded=(doc_type != "Other (Custom)")):
            current_fields = schemas.get(doc_type, DEFAULT_SCHEMAS.get(doc_type, []))

            # Show which files belong to this type
            files_of_type = [fn for fn, cls in updated_classifications.items() if cls == doc_type]
            dt_labels = []
            for fn in files_of_type:
                dt = data_types.get(fn, "Unstructured")
                dt_labels.append(f'<span style="background:{"#e8f5e9" if dt=="Structured" else "#fff3e0"};'
                                  f'color:{"#2e7d32" if dt=="Structured" else "#e65100"};'
                                  f'padding:1px 7px;border-radius:4px;font-size:11px;font-weight:700;">{dt}</span>'
                                  f'&nbsp;<span style="font-size:11px;color:#5a6878;">{fn}</span>')
            st.markdown(
                f'<div style="margin-bottom:10px;display:flex;flex-wrap:wrap;gap:6px;">'
                + "".join(dt_labels) + "</div>",
                unsafe_allow_html=True
            )

            if doc_type == "Other (Custom)" and not current_fields:
                st.info("No extraction schema defined for custom files. Add fields below if needed.", icon="ℹ️")

            # Display existing fields with remove buttons
            fields_to_keep = []
            for i, field in enumerate(current_fields):
                col_field, col_remove = st.columns([5, 1])
                with col_field:
                    edited_field = st.text_input(
                        f"Field {i+1}",
                        value=field,
                        key=f"schema_{doc_type}_{i}",
                        label_visibility="collapsed"
                    )
                    fields_to_keep.append(edited_field)
                with col_remove:
                    if st.button("✕", key=f"remove_{doc_type}_{i}",
                                 help="Remove this field"):
                        fields_to_keep.pop()
                        schemas[doc_type] = fields_to_keep
                        st.session_state.extraction_schemas = schemas
                        st.rerun()

            # Add new field
            col_new, col_add = st.columns([4, 1])
            with col_new:
                new_field = st.text_input(
                    "Add custom field",
                    placeholder="e.g., Off-Balance Sheet Exposure",
                    key=f"new_field_{doc_type}",
                    label_visibility="collapsed"
                )
            with col_add:
                if st.button("＋ Add", key=f"add_{doc_type}"):
                    if new_field.strip():
                        fields_to_keep.append(new_field.strip())

            schemas[doc_type] = [f for f in fields_to_keep if f.strip()]

    st.session_state.extraction_schemas = schemas

    # ── Navigation ──
    st.markdown("---")
    col_back, col_approve = st.columns([1, 3])
    with col_back:
        if st.button("← Back", use_container_width=True):
            go_to_stage(2); st.rerun()
    with col_approve:
        if st.button(
            "✅  Approve Classification & Run Pre-Cognitive Analysis →",
            use_container_width=True, type="primary"
        ):
            st.session_state.classification_approved = True
            go_to_stage(4); st.rerun()

# ═══════════════════════════════════════════════════════════════
#  STAGE 4 — PRE-COGNITIVE ANALYSIS & OUTPUT
# ═══════════════════════════════════════════════════════════════
if st.session_state.current_stage == 4:
    ed  = st.session_state.entity_data
    ld  = st.session_state.loan_data
    files = st.session_state.uploaded_files_meta
    classifications = st.session_state.doc_classifications
    schemas = st.session_state.extraction_schemas

    section_header("4", "📊", "Pre-Cognitive Analysis")

    # ── Run Analysis (only once per session) ──
    if not st.session_state.analysis_complete:
        progress = st.progress(0, text="🚀 Initialising Pre-Cognitive Analysis Engine...")

        extracted_insights = ""
        company_name = ed["company_name"]
        ceo_name     = ed["ceo_name"]

        # Step 1: Parse documents using schema-aware parsers
        progress.progress(15, text="📄 Parsing uploaded documents with Dynamic Schema...")
        pdf_files = [f for f in files if f.name.endswith(".pdf")]
        csv_files = [f for f in files if not f.name.endswith(".pdf")]

        if pdf_files:
            for pdf in pdf_files:
                doc_type = classifications.get(pdf.name)
                file_schema = schemas.get(doc_type, []) if doc_type else []
                summary = analyze_pdf_risks_with_schema(
                    uploaded_file=pdf,
                    llm_function=analyze_text_with_fallback,
                    doc_type=doc_type,
                    schema=file_schema if file_schema else None,
                )
                extracted_insights += (
                    f"\n### {doc_type or 'Document'}: {pdf.name}\n{summary}\n"
                )

        if csv_files:
            struct_report = analyze_structured_data_with_schema(
                uploaded_files=csv_files,
                schemas=schemas,
                classifications=classifications,
            )
            extracted_insights += f"\n{struct_report}\n"

        # Step 2: Web Intelligence Triangulation
        progress.progress(45, text=f"🌐 Crawling web for {company_name} intelligence...")
        web_report = crawl_company_news(company_name, ceo_name)
        st.session_state.web_research = web_report
        # Kept separate so the final prompt can triangulate docs vs web explicitly
        web_insights = web_report

        # Step 3: ML Risk Score
        progress.progress(65, text="🧠 Running ML Risk Scoring Engine...")
        decision = calculate_risk_score(
            qualitative_notes="",  # All signal comes from documents
            extracted_insights=extracted_insights,
            requested_amount=ld["requested_amount"],
            company_name=company_name,
            ceo_name=ceo_name,
        )
        st.session_state.final_decision = decision

        # Step 4: SWOT + CAM Generation (triangulates doc data vs web intelligence)
        progress.progress(80, text="📑 Generating SWOT Analysis & Investment Report...")

        swot_and_cam_prompt = f"""
You are a senior credit analyst at a Tier-1 bank. Generate a structured investment report.

### ENTITY: {company_name} | CEO / KEY PROMOTER: {ceo_name}
### LOAN: {ld['loan_type']} of {format_to_inr_words(ld['requested_amount'])} for {ld['tenure_months']} months @ {ld['proposed_interest_rate']}% p.a.
### SECTOR: {ed['sector']}
### SYSTEM CREDIT SCORE: {decision.get('credit_score', 0)}/10
### SYSTEM RECOMMENDED LIMIT: ₹{decision.get('recommended_limit_inr', 0):,}
### SYSTEM RECOMMENDED RATE: {decision.get('recommended_interest_rate', 0)}%

---

### PRIMARY DATA — Extracted from Uploaded Documents (Schema-Parsed):
{extracted_insights[:6000]}

---

### SECONDARY DATA — Web Intelligence & News Research:
{web_insights[:2500]}

---

### REQUIRED OUTPUT FORMAT (Strict Markdown — do not deviate):

### 1. EXECUTIVE SUMMARY
(2-3 sentences. State Approve / Caution / Reject. Quote the system-recommended limit and rate.)

### 2. FINANCIAL SUMMARY
| Metric | FY22 | FY23 | FY24 |
|--------|------|------|------|
(Populate from document data. Use "N/A" if unavailable.)

### 3. SWOT ANALYSIS
| Dimension | Key Points |
|-----------|------------|
| 💪 Strengths | • point 1 • point 2 |
| ⚠️ Weaknesses | • point 1 • point 2 |
| 🚀 Opportunities | • point 1 • point 2 |
| 🔴 Threats | • point 1 • point 2 |

### 4. WEB INTELLIGENCE TRIANGULATION
Compare what the uploaded documents say vs. what secondary web research reveals.
Call out any divergence (e.g., documents show strong growth but news shows regulatory action).
Provide 3–5 bullet points.

### 5. THE FIVE Cs OF CREDIT
| Category | Assessment | Risk Level |
|----------|------------|------------|
| Character | ... | High / Med / Low |
| Capacity | ... | High / Med / Low |
| Capital | ... | High / Med / Low |
| Collateral | ... | High / Med / Low |
| Conditions | ... | High / Med / Low |

### 6. RISK FLAGS & EARLY WARNING SIGNALS
(Bullet points. If none found, state "No critical signals detected.")

### 7. RECOMMENDED TERMS
| Parameter | Recommended Value |
|-----------|------------------|
| Facility Limit | |
| Interest Rate | |
| Tenure | |
| Key Covenants | |
| Conditions Precedent | |
"""

        cam_summary = analyze_text_with_fallback(swot_and_cam_prompt)
        st.session_state.cam_summary = cam_summary

        # Step 5: Build downloadable PDF investment report
        st.session_state.pdf_bytes = _build_pdf(
            cam_summary=cam_summary,
            entity_data=ed,
            loan_data=ld,
            decision=decision,
        )

        progress.progress(100, text="✅ Analysis complete.")
        st.session_state.analysis_complete = True
        time.sleep(0.5)
        progress.empty()
        st.rerun()

    # ── Render Results Dashboard ──
    decision    = st.session_state.final_decision
    cam_summary = st.session_state.cam_summary
    s_score     = decision.get("credit_score", 0)
    s_color     = score_color(s_score)
    verdict     = "APPROVE" if s_score >= 7 else "CAUTION" if s_score >= 5 else "REJECT"

    # ── Score Cards ──
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-top:4px solid {s_color};">
          <div style="font-size:10px;letter-spacing:.14em;color:#6a88aa;font-weight:700;
                      text-transform:uppercase;margin-bottom:8px;">Credit Score</div>
          <div style="font-size:38px;font-weight:800;color:{s_color};">
            {s_score}<span style="font-size:16px;color:#9ab0c8;"> / 10</span>
          </div>
          <div style="margin-top:10px;height:5px;background:#e4eef8;border-radius:3px;">
            <div style="width:{s_score*10}%;height:100%;background:{s_color};border-radius:3px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-top:4px solid #1a5fd4;">
          <div style="font-size:10px;letter-spacing:.14em;color:#6a88aa;font-weight:700;
                      text-transform:uppercase;margin-bottom:8px;">Approved Limit</div>
          <div style="font-size:22px;font-weight:800;color:#0d1f3c;">
            {format_to_inr_words(decision.get('recommended_limit_inr', 0))}
          </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-top:4px solid #6a4fd4;">
          <div style="font-size:10px;letter-spacing:.14em;color:#6a88aa;font-weight:700;
                      text-transform:uppercase;margin-bottom:8px;">Interest Rate</div>
          <div style="font-size:38px;font-weight:800;color:#6a4fd4;">
            {decision.get('recommended_interest_rate', 0)}<span style="font-size:16px;">%</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with c4:
        vbg  = {"APPROVE": "#edfbf4", "CAUTION": "#fffbec", "REJECT": "#fff0f0"}
        vbdr = {"APPROVE": "#88d8b8", "CAUTION": "#f0d060",  "REJECT": "#ffaaaa"}
        vicon= {"APPROVE": "✅",       "CAUTION": "⚠️",       "REJECT": "❌"}
        st.markdown(f"""
        <div class="metric-card" style="border-top:4px solid {s_color};
             background:{vbg.get(verdict,'#fff')};border-color:{vbdr.get(verdict,'#d8e8f8')};">
          <div style="font-size:10px;letter-spacing:.14em;color:{s_color};font-weight:700;
                      text-transform:uppercase;margin-bottom:8px;">System Verdict</div>
          <div style="font-size:30px;font-weight:800;color:{s_color};">{verdict}</div>
          <div style="font-size:22px;margin-top:4px;">{vicon.get(verdict,'')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── Data Pipeline Logs ──
    section_header("2", "⚙️", "Real-Time Data Pipeline Logs")
    tab_web, tab_decision = st.tabs(["🌐 Web Intelligence", "🧠 AI Decision Logic"])
    with tab_web:
        st.markdown(st.session_state.web_research or "No web data collected.")
    with tab_decision:
        logic_html = decision.get("decision_logic", "No logic available.")
        st.markdown(
            f'<div class="research-log" style="font-family:\'Plus Jakarta Sans\',sans-serif;">'
            f'{logic_html}</div>',
            unsafe_allow_html=True
        )

    # ── Investment Report (CAM) ──
    st.markdown("---")
    section_header("3", "📋", "Investment Report")
    st.markdown(cam_summary)

    # ── Export ──
    st.markdown("---")
    section_header("4", "📤", "Export")
    col_dl, col_restart = st.columns([3, 1])
    with col_dl:
        company_slug = ed.get("company_name", "entity").replace(" ", "_")
        st.download_button(
            label="📥  Download Investment Report (PDF)",
            data=st.session_state.pdf_bytes or b"",
            file_name=f"{company_slug}_Investment_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col_restart:
        if st.button("🔄 New Appraisal", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()