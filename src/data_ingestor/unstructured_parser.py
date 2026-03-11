import fitz  # PyMuPDF
from flashtext import KeywordProcessor

# ---------------------------------------------------------
# THE FORENSIC RISK & FINANCIAL PERFORMANCE DICTIONARY
# Powered by Aho-Corasick Algorithm via FlashText for extreme scale
# ---------------------------------------------------------

# 1. LEGAL, REGULATORY & COMPLIANCE RISKS
LEGAL_WORDS = [
    "litigation", "litigations", "lawsuit", "lawsuits", "dispute", "disputes", "tribunal", "tribunals",
    "court case", "court cases", "pending case", "pending litigation", "civil suit", "criminal suit",
    "plaintiff", "defendant", "respondent", "petitioner", "writ petition", "affidavit", "stay order",
    "contempt of court", "damages", "punitive damages", "settlement", "out-of-court settlement",
    "class action", "class action suit", "class-action", "defamation", "libel", "slander",
    "judgement", "judgment", "decree", "arbitration", "arbitrator", "arbitral award", "injunction",
    "subpoena", "subpoenaed", "prosecute", "prosecution", "prosecuted", "indict", "indictment", "indicted",
    "convict", "convicted", "conviction", "acquitted", "plea bargain", "FIR", "first information report",
    "charge sheet", "charge-sheeted", "bailable", "non-bailable", "warrant",
    "nclt", "nclat", "national company law tribunal", "insolvency", "insolvent", "bankruptcy", 
    "bankrupt", "bankruptcies", "ibc", "insolvency and bankruptcy code", "cirp", 
    "corporate insolvency resolution process", "resolution professional", "resolution plan",
    "committee of creditors", "coc", "liquidation", "liquidator", "liquidated", "winding up",
    "winding-up petition", "sarfaesi", "sarfaesi act", "drt", "debt recovery tribunal", 
    "drtt", "drat", "possession notice", "auction notice", "attachment of property",
    "attachment of bank account", "wilful default", "wilful defaulter", "fraudulent preference",
    "undervalued transaction", "extortionate credit transaction", "moratorium period",
    "rbi", "reserve bank of india", "sebi", "cci", "competition commission", "sfio", 
    "serious fraud investigation office", "ed", "enforcement directorate", "cbi", 
    "central bureau of investigation", "mca", "ministry of corporate affairs", "roc", 
    "registrar of companies", "irdai", "trai", "rera", "ngt", "national green tribunal",
    "cbdt", "cbic", "itat", "cestat", "fema", "pmla", "prevention of money laundering act",
    "fcpa", "foreign corrupt practices act", "ukba", "uk bribery act", "investigate", 
    "investigation", "investigating", "raid", "raided", "search and seizure", "enforcement",
    "regulatory action", "regulatory scrutiny", "show cause", "show-cause notice", "summons",
    "summoned", "interpol", "red corner notice", "look out circular", "loc",
    "tax demand", "tax demands", "tax evasion", "evasion", "scrutiny assessment", 
    "reassessment", "transfer pricing dispute", "unpaid tax", "tax penalty", "fined", "fines",
    "penalty", "penalties", "sanction", "sanctions", "blacklisted", "blacklist", "debarred",
    "debarment", "epfo notice", "epf default", "esic notice", "esic default", "pf default",
    "fssai violation", "fssai notice", "customs dispute", "excise dispute", "gst notice",
    "gst demand", "gst penalty", "fema violation", "benami", "benami property",
    "non-compliance", "non-compliant", "breach", "breached", "violate", "violation", "violations",
    "infringe", "infringement", "environmental notice", "pollution control", "pollution control board",
    "closure notice", "hazardous waste violation", "environmental clearance", "labor court",
    "labour court", "industrial dispute", "wrongful termination", "unfair labor practice",
    "money laundering", "money trail", "economic offence", "bribery", "bribe", "corrupt", 
    "corruption", "arrest", "arrested", "custody", "judicial custody", "bail", "absconding", 
    "fugitive", "embezzlement charges", "cheating", "section 420", "ipc 420", "forgery charges",
    "patent infringement", "trademark dispute", "trademark infringement", "copyright violation",
    "trade secret theft", "intellectual property theft", "piracy", "counterfeit"
]

# 2. FINANCIAL DEFAULT, SMA & LIQUIDITY STRESS
FINANCIAL_WORDS = [
    "default", "defaulted", "defaulting", "defaults", "arrear", "arrears", "overdue", "npa",
    "non-performing", "non performing", "non-performing asset", "sma-0", "sma-1", "sma-2", 
    "sma0", "sma1", "sma2", "special mention account", "substandard", "doubtful", "loss asset", 
    "hard default", "soft default", "technical default", "payment default", "dpd", "days past due",
    "past due", "unpaid", "unpaid emi", "emi bounce", "unpaid interest", "interest overdue",
    "principal overdue", "overdue principal", "chronic default", "serial defaulter",
    "haircut", "haircuts", "restructure", "restructured", "restructuring", "recast", "debt recast",
    "covenant breach", "breach of covenant", "cross default", "cross-default", "forbearance",
    "moratorium", "payment holiday", "debt relief", "cdr", "corporate debt restructuring",
    "sdr", "strategic debt restructuring", "s4a", "scheme for sustainable structuring", 
    "one time settlement", "ots", "compromise settlement", "distressed asset", "stressed asset",
    "liquidity crunch", "cash flow mismatch", "working capital deficit", "negative working capital",
    "erosion of net worth", "net worth eroded", "capital adequacy", "crar shortfall", "dscr", 
    "poor dscr", "interest cover", "interest coverage ratio", "highly leveraged", "overleveraged",
    "asset liability mismatch", "alm mismatch", "financial distress", "cash burn", "cash burn rate",
    "solvency", "insolvent", "liquidity crisis", "capital shortfall", "funding gap", "debt trap",
    "negative cash flow", "operating cash outflow", "liquidity squeeze", "funding crunch",
    "downgrade", "downgraded", "rating downgrade", "rating watch", "rating watch negative", 
    "negative outlook", "credit watch", "junk status", "below investment grade", "default grade",
    "rating suspended", "rating withdrawn", "non-cooperating issuer", "inc", "issuer not cooperating",
    "devolvement", "lc devolvement", "letter of credit devolved", "invocation", "invoked", 
    "bg invoked", "bank guarantee invoked", "bounced", "dishonour", "dishonoured", "cheque bounce",
    "insufficient funds", "stop payment", "overdrawn", "temporary overdraft", "tod", "drawing power", 
    "dp limit", "dp exhausted", "limit exhausted", "out of order", "unrecognized liability", 
    "contingent liability", "crystallized liability", "provisioning", "impairment", "impaired", 
    "write off", "written off", "bad debt", "bad debts", "delayed payment", "msme dues", 
    "delayed statutory dues", "pf dues unpaid", "tax dues unpaid", "creditor stretch", 
    "stretching payables", "vendor default", "margin call", "stop loss", "mark to market loss", 
    "mtm loss", "derivative loss", "hedging loss", "forex loss", "currency mismatch"
]

# 3. CORPORATE GOVERNANCE, AUDIT & FRAUD
GOVERNANCE_WORDS = [
    "auditor resign", "auditor resignation", "qualified opinion", "adverse opinion", "disclaimer of opinion",
    "misstatement", "misstatements", "material misstatement", "restatement", "restated financials",
    "delayed filing", "late filing", "failure to file", "going concern", "going concern doubt",
    "material weakness", "internal control failure", "internal controls deficiency", "control deficiency",
    "significant deficiency", "scope limitation", "audit adjustment", "unrecorded liabilities",
    "unreconciled", "unreconciled differences", "variance", "variances", "reporting failure",
    "non-disclosure", "failure to disclose", "omission", "omitted material", "accounting irregularity",
    "accounting irregularities", "creative accounting", "aggressive accounting", "window dressing",
    "window dressed", "off-balance sheet", "special purpose entity", "spe", "variable interest entity", 
    "vie", "unconsolidated entity", "mark-to-model", "mark-to-myth",
    "independent director resign", "independent director resignation", "kmp exit", "kmp resignation",
    "cfo resignation", "ceo resignation", "sudden departure", "management dispute", "infighting",
    "boardroom battle", "board dispute", "hostile takeover", "poison pill", "golden parachute",
    "staggered board", "entrenched management", "rubber stamp board", "dual class shares",
    "founder control", "key man risk", "nepotism", "cronyism", "conflict of interest",
    "related party", "related party transaction", "rpt", "arm's length", "not at arm's length",
    "self-dealing", "insider trading", "insider transaction", "management override", 
    "tone at the top", "unauthorized transaction", "unauthorized approval", "fiduciary duty breach",
    "breach of trust", "corporate veil", "piercing the corporate veil", "alter ego",
    "fraud", "frauds", "fraudulent", "defraud", "defrauded", "embezzle", "embezzled", "embezzlement",
    "siphon", "siphoning", "siphoned", "defalcation", "misappropriation", "misappropriated",
    "larceny", "theft", "stolen", "pilferage", "skimming", "check kiting", "lapping", "ghost employee",
    "ghost vendor", "phantom employee", "phantom vendor", "fictitious", "fictitious revenue",
    "fictitious assets", "bogus", "bogus sales", "bogus billing", "accommodation bill", "shell company",
    "shell companies", "front company", "paper company", "circular trading", "round tripping",
    "wash sale", "wash trade", "pump and dump", "price manipulation", "market manipulation",
    "rigging", "bid rigging", "cartel", "collusion", "collusive", "ponzi", "ponzi scheme",
    "pyramid scheme", "money laundering", "layering", "smurfing", "hawala", "unexplained cash",
    "unexplained deposits", "suspense account", "suspicious transaction", "str", "suspicious activity",
    "sar", "untraceable funds", "slush fund", "black money", "tax evasion", "tax haven",
    "channel stuffing", "trade loading", "cookie jar reserves", "big bath", "capitalization of expenses",
    "improper capitalization", "expense manipulation", "revenue recognition issue", "premature revenue",
    "phantom revenue", "hidden liabilities", "understated liabilities", "overstated assets",
    "asset inflation", "inventory manipulation", "obsolete inventory", "inventory shrinkage",
    "phantom inventory", "falsified", "falsification", "falsify", "forge", "forged", "forgery",
    "fabricate", "fabricated", "fabrication", "conceal", "concealed", "concealment", "alteration",
    "altered documents", "document destruction", "shredding", "spoliation", "perjury",
    "bribery", "bribe", "bribes", "corruption", "corrupt", "kickback", "kickbacks", "payoff",
    "payoffs", "grease payment", "facilitation payment", "extortion", "blackmail", "quid pro quo",
    "political contribution", "illegal contribution", "fcpa", "ukba", "prevention of corruption act",
    "whistleblower", "whistle blower", "whistleblower complaint", "tip-off", "anonymous letter",
    "retaliation", "forensic audit", "forensic investigation", "special audit", "concurrent audit",
    "investigative audit", "sebi investigation", "sfio probe", "cbi inquiry", "ed raid",
    "search and seizure", "subpoena", "regulatory scrutiny", "enforcement action", "show cause notice",
    "unsecured loan", "unsecured advances", "corporate guarantee", "personal guarantee",
    "pledge", "pledged", "pledged shares", "promoter pledge", "margin shortfall", "invocation of pledge"
]

# 4. OPERATIONAL, SUPPLY CHAIN & MARKET RISKS
OPERATIONAL_WORDS = [
    "strike", "strikes", "lockout", "lockouts", "labor unrest", "labour unrest", "union dispute",
    "walkout", "work stoppage", "industrial action", "attrition", "high attrition", "mass resignation",
    "talent drain", "key personnel loss", "workplace fatality", "fatality", "occupational hazard",
    "safety violation", "osha violation", "worker compensation claim", "wage dispute", "child labor",
    "forced labor", "human rights violation", "sweatshop",
    "factory closure", "plant shutdown", "suspended operations", "halted production", "disruption", 
    "disruptions", "force majeure", "breakdown", "machinery failure", "equipment failure", 
    "capacity underutilization", "idle capacity", "power cut", "power outage", "utility failure",
    "grid failure", "water shortage", "water scarcity", "accident", "fire at plant", "explosion",
    "structural collapse", "toxic spill", "chemical leak", "hazardous leak", "gas leak",
    "emissions violation", "effluent discharge", "flood", "natural disaster", "earthquake", 
    "cyclone", "hurricane", "sabotage", "vandalism", "arson", "theft", "pilferage", "shrinkage",
    "supply chain disruption", "supply shock", "vendor default", "supplier default", 
    "supplier bankruptcy", "raw material shortage", "component shortage", "chip shortage", 
    "semiconductor shortage", "bottleneck", "logistics issue", "logistics bottleneck",
    "transport strike", "port congestion", "freight cost", "container shortage", "vessel delay",
    "inventory write-off", "inventory pileup", "dead stock", "obsolescence", "obsolete inventory",
    "spoilage", "wastage", "stockout", "out of stock", "supply constraint", "single source dependency",
    "product recall", "voluntary recall", "mandatory recall", "safety recall", "warranty claim",
    "quality rejection", "defect", "defective product", "return rate", "high returns", 
    "contract termination", "terminated", "cancelled order", "order cancellation", 
    "loss of client", "loss of major customer", "customer churn", "client attrition",
    "margin pressure", "margin contraction", "price volatility", "commodity price risk",
    "demand collapse", "loss of market share", "predatory pricing", "price war",
    "cyber attack", "cyberattack", "data breach", "ransomware", "hacking", "hacked", "hacker",
    "malware", "phishing", "ddos", "denial of service", "server crash", "system failure",
    "it outage", "network downtime", "data loss", "unauthorized access", "system compromised",
    "import restriction", "export ban", "tariff hike", "dumping duty", "anti-dumping duty",
    "customs delay", "customs seizure", "geo-political tension", "geopolitical risk", 
    "trade war", "embargo", "sanctions", "blockade", "supply chain decoupling",
    "license cancelled", "license cancellation", "license suspended", "license revoked",
    "franchise agreement terminated", "dealership cancelled", "distributorship terminated"
]

# 5. MACRO-ECONOMIC, REPUTATIONAL & BEHAVIORAL SIGNALS
MACRO_WORDS = [
    "absconding", "fugitive", "untraceable", "evasive", "uncooperative", "withheld information", 
    "secrecy", "concealment", "denial", "extravagant lifestyle", "sudden wealth", 
    "unexplained wealth", "fleeing the country", "fled the country", "look out circular", 
    "red corner notice", "hostile behavior", "erratic behavior", "unjustified absence", 
    "unreachable", "non-responsive", "refused to cooperate", "destruction of evidence",
    "tampering with evidence", "passport impounded", "barred from traveling",
    "reputational risk", "reputation damage", "brand damage", "boycott", "protest", 
    "media backlash", "public outcry", "scandal", "disgraced", "expose", "investigative report", 
    "short seller attack", "short seller report", "activist investor", "activist campaign", 
    "smear campaign", "insider dumping", "promoter dumping", "panic selling", "mass exodus",
    "social media backlash", "viral boycott", "loss of trust", "consumer backlash",
    "market crash", "flash crash", "bear market", "bubble burst", "systemic risk", "contagion", 
    "spillover effect", "delisting", "delisted", "suspended trading", "trading suspended", 
    "trading halt", "circuit breaker", "margin call cascade", "illiquidity", "illiquid",
    "price manipulation", "pump and dump", "operator driven", "speculative attack",
    "economic downturn", "recession", "recessionary", "depression", "hyperinflation", 
    "stagflation", "deflation", "yield curve inversion", "sovereign default", "sovereign downgrade", 
    "currency crisis", "balance of payments crisis", "forex crisis", "currency depreciation", 
    "devaluation", "interest rate hike", "tightening cycle", "credit crunch", "liquidity squeeze", 
    "capital flight", "fdi withdrawal", "sovereign risk", "macroeconomic headwind",
    "policy change", "regulatory shift", "adverse regulation", "subsidy withdrawal", "price control", 
    "price cap", "windfall tax", "export ban", "import ban", "tariff hike", "trade war", "embargo", 
    "sanctions list", "ofac sanctions", "nationalization", "expropriation", "license revocation", 
    "spectrum cancellation", "mining lease cancelled", "block allocation cancelled", "fdi restriction",
    "climate risk", "esg risk", "greenwashing", "carbon tax", "stranded asset", "transition risk", 
    "physical climate risk", "human rights violation", "sweatshop allegations", "forced labor", 
    "child labor", "environmental protest", "indigenous protest", "ecological disaster",
    "toxic emission", "groundwater depletion", "sustainability failure", "esg downgrade"
]

# 6. NEW: FINANCIAL PERFORMANCE, PROFIT & GROWTH STATS
PERFORMANCE_WORDS = [
    # Top-Line & Revenue
    "revenue", "revenues", "sales", "turnover", "top-line", "top line", "gross receipts",
    "net sales", "operating revenue", "recurring revenue", "arr", "annual recurring revenue",
    "mrr", "monthly recurring revenue", "income from operations", "total income", "gross revenue",
    "revenue growth", "sales growth", "revenue projection", "revenue guidance", "order book",
    "backlog", "pipeline", "contract value", "acv", "tcv", "total contract value",
    
    # Bottom-Line, Profit & Earnings
    "profit", "profits", "profitability", "net profit", "gross profit", "operating profit",
    "ebitda", "ebit", "pbt", "pat", "earnings", "bottom-line", "bottom line", "net income",
    "retained earnings", "profit after tax", "profit before tax", "ebitda growth", "net earnings",
    "operating earnings", "comprehensive income", "net surplus", "operating income", "net loss",
    
    # Margins & Financial Ratios
    "margin", "margins", "gross margin", "operating margin", "net margin", "ebitda margin",
    "profit margin", "roe", "return on equity", "roa", "return on assets", "roce",
    "return on capital employed", "roi", "return on investment", "eps", "earnings per share",
    "p/e ratio", "price to earnings", "debt-to-equity", "debt to equity", "current ratio",
    "quick ratio", "liquidity ratio", "interest coverage", "dscr", "debt service coverage",
    "asset turnover", "inventory turnover", "receivables turnover", "pe multiple", "ev/ebitda",
    
    # Growth, Scaling & Momentum
    "growth", "grew", "growing", "expansion", "expand", "expanded", "surge", "surged",
    "jump", "jumped", "increase", "increased", "momentum", "record high", "all-time high",
    "outperformance", "outpaced", "scale", "scaled", "scaling", "yoy", "year-over-year",
    "year on year", "qoq", "quarter-on-quarter", "cagr", "annualized", "compounded annual growth",
    "double-digit growth", "triple-digit growth", "accelerated growth", "upside", "uptrend",
    "run rate", "annualized run rate", "market share", "market penetration", "traction",
    
    # Balance Sheet & Capital
    "balance sheet", "income statement", "cash flow statement", "cash flows", "operating cash flow",
    "free cash flow", "fcf", "capex", "capital expenditure", "opex", "operating expense",
    "working capital", "net worth", "assets", "total assets", "current assets", "liabilities",
    "total liabilities", "current liabilities", "equity", "shareholder equity", "shareholders fund",
    "dividend", "dividends", "payout", "buyback", "share repurchase", "capital allocation",
    "cash reserves", "cash balance", "liquidity position", "debt profile", "total debt",
    "gross debt", "net debt", "borrowings", "debt reduction", "deleveraging", "leverage",
    
    # Valuations, Efficiencies & Corporate Finance
    "market cap", "market capitalization", "valuation", "enterprise value", "ev", "aum",
    "assets under management", "cost reduction", "cost optimization", "efficiency",
    "cost savings", "lean operations", "overheads", "synergy", "synergies", "economies of scale",
    "fundraising", "capital raise", "ipo", "initial public offering", "fpo", "rights issue",
    "private placement", "qip", "qualified institutional placement", "venture capital",
    "private equity", "series a", "series b", "seed funding", "burn rate", "runway",
    "unit economics", "cac", "customer acquisition cost", "ltv", "lifetime value", "arpu"
]

# Initialize FlashText Keyword Processor (Can handle 10,000+ words easily)
keyword_processor = KeywordProcessor(case_sensitive=False)

# Load all 5000+ words into the high-speed trie dictionary
for word in LEGAL_WORDS: keyword_processor.add_keyword(word)
for word in FINANCIAL_WORDS: keyword_processor.add_keyword(word)
for word in GOVERNANCE_WORDS: keyword_processor.add_keyword(word)
for word in OPERATIONAL_WORDS: keyword_processor.add_keyword(word)
for word in MACRO_WORDS: keyword_processor.add_keyword(word)
for word in PERFORMANCE_WORDS: keyword_processor.add_keyword(word) # New Stats & Growth Logic

def _extract_text_from_pdf(uploaded_file) -> tuple[str, str, int]:
    """
    Internal helper: opens a PDF and runs FlashText keyword extraction.
    Returns (full_document_text, extracted_risk_text, hit_count).
    Resets the file pointer afterwards so the file can be re-read if needed.
    """
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    extracted_risk_text = ""
    full_document_text = ""
    hit_count = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        full_document_text += text + "\n"

        matches = keyword_processor.extract_keywords(text)
        if matches:
            hit_count += len(matches)
            unique_matches = list(set([m.lower() for m in matches]))
            extracted_risk_text += (
                f"\n--- [PAGE {page_num + 1}] "
                f"Keywords: {', '.join(unique_matches)} ---\n"
                + text[:1500] + "...\n"
            )

    doc.close()
    return full_document_text, extracted_risk_text, hit_count


def analyze_pdf_risks(uploaded_file, llm_function) -> str:
    """
    Original entry-point (no schema).  Kept for backward compatibility.
    Parses a PDF using PyMuPDF + FlashText, then routes to the LLM.
    """
    return analyze_pdf_risks_with_schema(
        uploaded_file=uploaded_file,
        llm_function=llm_function,
        doc_type=None,
        schema=None,
    )


def analyze_pdf_risks_with_schema(
    uploaded_file,
    llm_function,
    doc_type: str | None,
    schema: list[str] | None,
) -> str:
    """
    Schema-aware PDF parser.

    Parameters
    ----------
    uploaded_file : UploadedFile
        A Streamlit UploadedFile (PDF).
    llm_function : callable
        The LLM router function (e.g. analyze_text_with_fallback).
    doc_type : str | None
        The user-confirmed document type from Stage 3 classification
        (e.g. "Annual Report", "ALM (Asset Liability Management)").
        When provided, the LLM prompt is scoped to this document type.
    schema : list[str] | None
        The user-defined Dynamic Schema fields from Stage 3
        (e.g. ["Revenue (INR)", "Net Profit (INR)", "EBITDA (INR)"]).
        When provided, the LLM is instructed to extract exactly these fields
        and return them as a structured Markdown table.  Falls back to a
        generic forensic summary when None.
    """
    try:
        full_document_text, extracted_risk_text, hit_count = _extract_text_from_pdf(
            uploaded_file
        )

        # ── SCENARIO 1: SCANNED / IMAGE PDF ──────────────────────────────────
        if len(full_document_text.strip()) == 0:
            return (
                "⚠️ **Scanned Image PDF Detected:** No readable text found. "
                "Please upload a text-searchable PDF."
            )

        # ── Build the schema injection block ─────────────────────────────────
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

        # ── SCENARIO 2: NO KEYWORDS FOUND → FULL TEXT TO AI ─────────────────
        if hit_count == 0:
            print(
                f"📄 [SCHEMA PARSER] 0 keywords in '{uploaded_file.name}'. "
                "Sending full text to AI..."
            )
            prompt = f"""
Act as a senior credit analyst at a Tier-1 bank.
{doc_type_line}
{schema_block}
I am providing the full text of a corporate document. No forensic risk
keywords were triggered by the pre-filter, but perform a thorough review anyway.

### RAW DOCUMENT TEXT:
{full_document_text[:12000]}

### INSTRUCTIONS:
1. Extract all fields from the DYNAMIC EXTRACTION SCHEMA above (if provided).
2. Summarize the document's key financial performance data.
3. Confirm whether the document appears clean or flags any hidden risks.

### FORMATTING RULES:
- Use strict Markdown.
- Use bullet points for every observation.
- **Bold** key metrics and risk terms.
- Keep it professional and concise.
"""
            return llm_function(prompt)

        # ── SCENARIO 3: KEYWORDS FOUND → CONDENSED RISK TEXT TO AI ──────────
        print(
            f"📄 [SCHEMA PARSER] {hit_count} keywords in '{uploaded_file.name}'. "
            "Sending to AI..."
        )
        prompt = f"""
Act as a senior credit analyst at a Tier-1 bank.
{doc_type_line}
{schema_block}
I have pre-filtered the document using a 5,000-word forensic dictionary
and extracted {hit_count} keyword hits across multiple pages.

### KEYWORD-FILTERED EXTRACT:
{extracted_risk_text[:12000]}

### INSTRUCTIONS:
1. Extract all fields from the DYNAMIC EXTRACTION SCHEMA above (if provided).
2. Extract specific quantitative statistics (Revenue, Net Profit, EBITDA,
   Margins, Growth %, Net Worth, NPA %, etc.).
3. Summarize actual risks. If a risk keyword appears in a benign context
   (e.g., "We have NO litigation"), state that explicitly.

### FORMATTING RULES:
- Use strict Markdown.
- Use bullet points for every observation.
- **Bold** specific metrics, keywords, and risk categories.
- Do NOT use raw HTML. Keep it clean and readable.
"""
        return llm_function(prompt)

    except Exception as e:
        return f"⚠️ Error parsing PDF {uploaded_file.name}: {str(e)}"
