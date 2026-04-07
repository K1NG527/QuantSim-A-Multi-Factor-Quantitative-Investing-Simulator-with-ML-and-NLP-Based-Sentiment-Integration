"""
PDF Parser Module — Stage 1: Document Understanding (NLP-Enhanced)
Extracts text from brokerage PDFs using pdfplumber, identifies document type
and broker using NLP techniques, and segments content.
"""

import re
import pdfplumber
from difflib import SequenceMatcher


# ── Document type detection keywords ──────────────────────────────────────────
DOCTYPE_KEYWORDS = {
    "Holdings Statement": [
        "holdings", "portfolio", "demat", "current value", "market value",
        "avg.? ?price", "average price", "isin"
    ],
    "Order History": [
        "order book", "order history", "trade book", "trade history",
        "executed", "order id", "order no"
    ],
    "Capital Gains": [
        "capital gain", "short term", "long term", "stcg", "ltcg",
        "acquisition cost", "sale consideration"
    ],
    "Profit & Loss": [
        "profit", "loss", "p&l", "p & l", "realised", "realized",
        "unrealised", "unrealized", "net gain"
    ],
    "Contract Notes": [
        "contract note", "contract no", "settlement", "brokerage",
        "stt", "exchange turnover"
    ],
}

# ── Broker detection keywords ─────────────────────────────────────────────────
BROKER_KEYWORDS = {
    "Zerodha": ["zerodha", "kite", "coin by zerodha"],
    "Groww": ["groww", "nextbillion"],
    "Upstox": ["upstox", "rksv"],
    "Angel One": ["angel", "angel broking", "angel one"],
    "ICICI Direct": ["icici direct", "icici securities"],
    "HDFC Securities": ["hdfc securities"],
    "Kotak Securities": ["kotak securities", "kotak"],
    "5Paisa": ["5paisa", "5 paisa"],
    "Motilal Oswal": ["motilal", "motilal oswal"],
    "Sharekhan": ["sharekhan"],
}

# ── Common Indian stock names for fuzzy matching ──────────────────────────────
CANONICAL_STOCKS = {
    "RELIANCE": ["reliance", "reliance ind", "reliance industries", "ril"],
    "TCS": ["tcs", "tata consultancy", "tata consultancy services"],
    "INFY": ["infy", "infosys", "infosys technologies"],
    "HDFCBANK": ["hdfc bank", "hdfcbank"],
    "ICICIBANK": ["icici bank", "icicibank"],
    "HINDUNILVR": ["hul", "hindustan unilever", "hindunilvr"],
    "SBIN": ["sbi", "state bank", "state bank of india", "sbin"],
    "BHARTIARTL": ["bharti airtel", "airtel", "bhartiartl"],
    "ITC": ["itc"],
    "KOTAKBANK": ["kotak mahindra", "kotak bank", "kotakbank"],
    "LT": ["l&t", "larsen", "larsen & toubro", "larsen and toubro"],
    "WIPRO": ["wipro"],
    "AXISBANK": ["axis bank", "axisbank"],
    "MARUTI": ["maruti", "maruti suzuki"],
    "TATAMOTORS": ["tata motors", "tatamotors"],
    "TATASTEEL": ["tata steel", "tatasteel"],
    "SUNPHARMA": ["sun pharma", "sun pharmaceutical", "sunpharma"],
    "BAJFINANCE": ["bajaj finance", "bajfinance"],
    "HCLTECH": ["hcl tech", "hcl technologies", "hcltech"],
    "ASIANPAINT": ["asian paints", "asianpaint", "asian paint"],
    "ADANIENT": ["adani enterprises", "adanient"],
    "ADANIPORTS": ["adani ports", "adaniports"],
    "POWERGRID": ["power grid", "powergrid"],
    "NTPC": ["ntpc"],
    "ONGC": ["ongc", "oil and natural gas"],
    "COALINDIA": ["coal india", "coalindia"],
    "TECHM": ["tech mahindra", "techm"],
    "ULTRACEMCO": ["ultratech cement", "ultracemco"],
    "TITAN": ["titan", "titan company"],
    "NESTLEIND": ["nestle", "nestle india", "nestleind"],
    "BAJAJFINSV": ["bajaj finserv", "bajajfinsv"],
    "DRREDDY": ["dr reddys", "dr reddy", "drreddy"],
    "DIVISLAB": ["divis lab", "divi's laboratories", "divislab"],
    "CIPLA": ["cipla"],
    "EICHERMOT": ["eicher motors", "eichermot"],
    "HEROMOTOCO": ["hero motocorp", "heromotoco"],
    "BPCL": ["bpcl", "bharat petroleum"],
    "GRASIM": ["grasim", "grasim industries"],
    "JSWSTEEL": ["jsw steel", "jswsteel"],
    "VEDL": ["vedanta", "vedl"],
    "ZOMATO": ["zomato"],
    "PAYTM": ["paytm", "one97"],
}


def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from a PDF file (path string or file-like object)."""
    text_pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    return "\n".join(text_pages)


def extract_tables_from_pdf(pdf_file) -> list:
    """Extract tabular data from all pages."""
    all_tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                all_tables.extend(tables)
    return all_tables


def detect_document_types(text: str) -> list:
    """Identify document type(s) from extracted text using pattern matching."""
    text_lower = text.lower()
    detected = []
    for doc_type, keywords in DOCTYPE_KEYWORDS.items():
        for kw in keywords:
            if re.search(kw, text_lower):
                detected.append(doc_type)
                break
    return detected if detected else ["Unknown"]


def detect_broker(text: str) -> str:
    """Identify the broker/source from extracted text."""
    text_lower = text.lower()
    for broker, keywords in BROKER_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return broker
    return "Unknown"


def resolve_stock_name(raw_name: str) -> dict:
    """
    NLP-style fuzzy matching to resolve a raw stock name to canonical form.
    Returns dict with canonical_symbol, confidence, and match_type.
    """
    if not raw_name:
        return {"canonical_symbol": None, "confidence": 0, "match_type": "none"}

    clean = raw_name.strip().lower()
    clean = re.sub(r"\s+(ltd\.?|limited|inc\.?|corp\.?|nse|bse)\s*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()

    # Exact match
    for symbol, aliases in CANONICAL_STOCKS.items():
        if clean in aliases or clean == symbol.lower():
            return {"canonical_symbol": symbol, "confidence": 1.0, "match_type": "exact"}

    # Fuzzy match
    best_score = 0
    best_symbol = None
    for symbol, aliases in CANONICAL_STOCKS.items():
        for alias in aliases:
            score = SequenceMatcher(None, clean, alias).ratio()
            if score > best_score:
                best_score = score
                best_symbol = symbol

    if best_score >= 0.75:
        return {"canonical_symbol": best_symbol, "confidence": round(best_score, 3), "match_type": "fuzzy"}

    # No match — return cleaned name as-is
    return {
        "canonical_symbol": raw_name.strip().upper().split()[0],
        "confidence": 0.5,
        "match_type": "inferred",
    }


def infer_table_headers(table: list) -> list | None:
    """
    NLP heuristic: if first row doesn't look like headers (no text-heavy cells),
    try to infer column types from data patterns.
    """
    if not table or len(table) < 2:
        return None

    first_row = table[0]
    # Check if first row looks like headers (mostly text, few numbers)
    text_cells = sum(1 for c in first_row if c and not re.match(r"^[\d,.\-₹$%]+$", str(c).strip()))

    if text_cells >= len(first_row) * 0.5:
        return None  # First row is likely headers, no inference needed

    # Infer from data patterns
    num_cols = len(first_row)
    inferred = []
    for col_idx in range(num_cols):
        col_values = [str(row[col_idx]).strip() if col_idx < len(row) and row[col_idx] else ""
                      for row in table[1:min(5, len(table))]]

        # Date pattern
        if any(re.match(r"\d{2}[/-]\d{2}[/-]\d{4}", v) for v in col_values):
            inferred.append("date")
        # All integers
        elif all(re.match(r"^\d+$", v) for v in col_values if v):
            inferred.append("quantity")
        # Decimal numbers
        elif all(re.match(r"^[\d,]+\.\d+$", v) for v in col_values if v):
            if len(inferred) == 0 or "price" not in str(inferred):
                inferred.append("price")
            else:
                inferred.append("value")
        # Text
        else:
            inferred.append("stock_name")

    return inferred


def segment_text(text: str) -> dict:
    """Segment extracted text into logical sections."""
    lines = text.split("\n")
    tables_text = []
    transaction_lines = []
    summary_lines = []

    date_pattern = re.compile(r"\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2}")
    summary_keywords = re.compile(
        r"\b(total|summary|net|grand total|overall|aggregate)\b", re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        num_count = len(re.findall(r"[\d,]+\.?\d*", stripped))
        has_date = bool(date_pattern.search(stripped))

        if summary_keywords.search(stripped):
            summary_lines.append(stripped)
        elif has_date and num_count >= 2:
            transaction_lines.append(stripped)
        elif num_count >= 3:
            tables_text.append(stripped)
        else:
            summary_lines.append(stripped)

    return {
        "tables_text": tables_text,
        "transaction_lines": transaction_lines,
        "summary_lines": summary_lines,
    }


def parse_pdf(pdf_files) -> dict:
    """
    Full Stage 1 pipeline with NLP enhancements.
    Supports a single file-like object or a list/tuple of files.
    Aggregates text, tables, and metadata across all documents.
    """
    # Robustly handle single file vs list/tuple
    print(f"[Parser] Received pdf_files type: {type(pdf_files)}")
    
    if hasattr(pdf_files, "seek") or not isinstance(pdf_files, (list, tuple)):
        pdf_files = [pdf_files]
    
    # Flatten if it's a list containing a list (though shouldn't happen now)
    if len(pdf_files) == 1 and isinstance(pdf_files[0], (list, tuple)):
        pdf_files = pdf_files[0]

    aggregated_text = []
    aggregated_tables = []
    aggregated_doc_types = set()
    aggregated_brokers = set()
    
    for pdf_file in pdf_files:
        try:
            raw_text = extract_text_from_pdf(pdf_file)
            raw_tables = extract_tables_from_pdf(pdf_file)
            doc_types = detect_document_types(raw_text)
            broker = detect_broker(raw_text)
            
            aggregated_text.append(raw_text)
            aggregated_tables.extend(raw_tables)
            aggregated_doc_types.update(doc_types)
            if broker != "Unknown":
                aggregated_brokers.update([broker])
        except Exception as e:
            print(f"[Parser] Error processing a file: {e}")

    full_text = "\n\n--- NEXT DOCUMENT ---\n\n".join(aggregated_text)
    sections = segment_text(full_text)

    # Infer headers for tables missing them
    inferred_headers = []
    for table in aggregated_tables:
        headers = infer_table_headers(table)
        inferred_headers.append(headers)

    return {
        "raw_text": full_text,
        "raw_tables": aggregated_tables,
        "inferred_headers": inferred_headers,
        "document_types": list(aggregated_doc_types) if aggregated_doc_types else ["Unknown"],
        "broker": ", ".join(aggregated_brokers) if aggregated_brokers else "Unknown",
        "sections": sections,
    }
