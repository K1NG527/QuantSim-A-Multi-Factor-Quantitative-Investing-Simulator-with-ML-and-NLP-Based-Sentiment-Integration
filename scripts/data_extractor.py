"""
Data Extractor Module — Stages 2-3: Data Extraction & Cleaning (NLP-Enhanced)
Parses structured financial data from raw PDF text, resolves stock symbols
using fuzzy matching, and adds confidence scores.
"""

import re
from datetime import datetime
from scripts.pdf_parser import resolve_stock_name


# ── Stock name normalization ──────────────────────────────────────────────────
STRIP_SUFFIXES = re.compile(
    r"\s+(LTD\.?|LIMITED|INC\.?|INCORPORATED|CORP\.?|CORPORATION|PLC|NV|SA|AG|CO\.?|COMPANY)\s*$",
    re.IGNORECASE,
)


def normalize_stock_name(name: str) -> str:
    """Strip common corporate suffixes and extra whitespace."""
    if not name:
        return name
    cleaned = STRIP_SUFFIXES.sub("", name.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def safe_float(val) -> float | None:
    """Parse a numeric string to float, handling commas and currency symbols."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip()
    val = re.sub(r"[₹$€£,\s]", "", val)
    val = val.replace("(", "-").replace(")", "")
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_date(date_str: str) -> str | None:
    """Parse various date formats into YYYY-MM-DD."""
    if not date_str:
        return None
    date_str = date_str.strip()
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%d-%b-%Y", "%d %b %Y", "%d-%B-%Y", "%d %B %Y",
        "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y",
        "%d-%m-%y", "%d/%m/%y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ── Holdings Extraction ───────────────────────────────────────────────────────

def extract_holdings(raw_text: str, raw_tables: list) -> list:
    """
    Extract holdings data from PDF content.
    Returns list of dicts with: stock_name, symbol, standardized_symbol,
    quantity, average_buy_price, current_price, total_value, portfolio_weight,
    confidence, match_type.
    """
    holdings = []

    # Try table-based extraction first
    for table in raw_tables:
        if not table or len(table) < 2:
            continue
        header = [str(c).lower().strip() if c else "" for c in table[0]]

        has_qty = any("qty" in h or "quantity" in h or "quant" in h for h in header)
        has_price = any("price" in h or "avg" in h or "cost" in h for h in header)

        if not (has_qty or has_price):
            continue

        col_map = {}
        for i, h in enumerate(header):
            if "stock" in h or "instrument" in h or "scrip" in h or "company" in h or "name" in h:
                col_map["stock_name"] = i
            elif "symbol" in h or "ticker" in h or "code" in h:
                col_map["symbol"] = i
            elif "qty" in h or "quantity" in h or "quant" in h:
                col_map["quantity"] = i
            elif ("avg" in h or "average" in h or "cost" in h or "buy" in h) and "price" in h:
                col_map["average_buy_price"] = i
            elif ("current" in h or "market" in h or "ltp" in h or "close" in h) and ("price" in h or "value" not in h):
                col_map["current_price"] = i
            elif "invested" in h or "cost" in h and "value" in h:
                col_map["average_buy_price"] = i # Map invested to buy price/value for simplicity
            elif "value" in h or "amount" in h or "mkt" in h:
                col_map["total_value"] = i
            elif "weight" in h or "%" in h or "alloc" in h:
                col_map["portfolio_weight"] = i

        for row in table[1:]:
            if not row or all(c is None or str(c).strip() == "" for c in row):
                continue
            entry = {
                "stock_name": None, "symbol": None, "standardized_symbol": None,
                "quantity": None, "average_buy_price": None, "current_price": None,
                "total_value": None, "portfolio_weight": None,
                "confidence": None, "match_type": None,
            }
            for field, idx in col_map.items():
                if idx < len(row) and row[idx] is not None:
                    val = str(row[idx]).strip()
                    if field == "stock_name":
                        entry[field] = normalize_stock_name(val)
                    elif field == "symbol":
                        entry[field] = val.upper()
                    elif field in ("quantity",):
                        entry[field] = safe_float(val)
                        if entry[field] is not None:
                            entry[field] = int(entry[field])
                    else:
                        entry[field] = safe_float(val)

            # NLP: Resolve stock name to canonical symbol
            name_for_resolve = entry["stock_name"] or entry.get("symbol", "")
            if name_for_resolve:
                resolved = resolve_stock_name(name_for_resolve)
                entry["standardized_symbol"] = resolved["canonical_symbol"]
                entry["confidence"] = resolved["confidence"]
                entry["match_type"] = resolved["match_type"]

            if entry["stock_name"] or entry["symbol"]:
                holdings.append(entry)

    # Fallback: regex-based extraction from raw text
    if not holdings:
        holdings = _extract_holdings_from_text(raw_text)

    # Compute portfolio weights if not present
    total_val = sum(h["total_value"] for h in holdings if h["total_value"])
    if total_val > 0:
        for h in holdings:
            if h["total_value"] and h.get("portfolio_weight") is None:
                h["portfolio_weight"] = round((h["total_value"] / total_val) * 100, 2)

    return holdings


def _extract_holdings_from_text(text: str) -> list:
    """Fallback regex extraction for holdings from raw text lines."""
    holdings = []
    row_pattern = re.compile(
        r"^([A-Z][A-Za-z\s&.\-]+?)\s+"
        r"(\d[\d,]*)\s+"
        r"([\d,]+\.?\d*)\s+"
        r"([\d,]+\.?\d*)\s+"
        r"([\d,]+\.?\d*)",
        re.MULTILINE,
    )
    for match in row_pattern.finditer(text):
        name = normalize_stock_name(match.group(1))
        resolved = resolve_stock_name(name)
        holdings.append({
            "stock_name": name,
            "symbol": None,
            "standardized_symbol": resolved["canonical_symbol"],
            "quantity": int(safe_float(match.group(2)) or 0),
            "average_buy_price": safe_float(match.group(3)),
            "current_price": safe_float(match.group(4)),
            "total_value": safe_float(match.group(5)),
            "portfolio_weight": None,
            "confidence": resolved["confidence"],
            "match_type": resolved["match_type"],
        })
    return holdings


# ── Order / Trade History Extraction ──────────────────────────────────────────

def extract_trades(raw_text: str, raw_tables: list) -> list:
    """
    Extract order/trade history.
    Returns list of dicts with: stock_name, standardized_symbol, date,
    transaction_type, quantity, price, confidence.
    """
    trades = []

    for table in raw_tables:
        if not table or len(table) < 2:
            continue
        header = [str(c).lower().strip() if c else "" for c in table[0]]

        has_type = any("buy" in h or "sell" in h or "type" in h or "side" in h for h in header)
        has_date_col = any("date" in h or "time" in h for h in header)

        if not (has_type or has_date_col):
            continue

        col_map = {}
        for i, h in enumerate(header):
            if "stock" in h or "instrument" in h or "scrip" in h or "symbol" in h or "name" in h:
                col_map["stock_name"] = i
            elif "date" in h and "settle" not in h:
                col_map["date"] = i
            elif "type" in h or "side" in h or "action" in h:
                col_map["transaction_type"] = i
            elif "qty" in h or "quantity" in h:
                col_map["quantity"] = i
            elif "price" in h and "avg" not in h:
                col_map["price"] = i

        for row in table[1:]:
            if not row or all(c is None or str(c).strip() == "" for c in row):
                continue
            entry = {
                "stock_name": None, "standardized_symbol": None,
                "date": None, "transaction_type": None,
                "quantity": None, "price": None, "confidence": None,
            }
            for field, idx in col_map.items():
                if idx < len(row) and row[idx] is not None:
                    val = str(row[idx]).strip()
                    if field == "stock_name":
                        entry[field] = normalize_stock_name(val)
                    elif field == "date":
                        entry[field] = parse_date(val)
                    elif field == "transaction_type":
                        entry[field] = "BUY" if "buy" in val.lower() else "SELL"
                    elif field == "quantity":
                        entry[field] = safe_float(val)
                        if entry[field] is not None:
                            entry[field] = int(entry[field])
                    elif field == "price":
                        entry[field] = safe_float(val)

            if entry["stock_name"]:
                resolved = resolve_stock_name(entry["stock_name"])
                entry["standardized_symbol"] = resolved["canonical_symbol"]
                entry["confidence"] = resolved["confidence"]
                trades.append(entry)

    if not trades:
        trades = _extract_trades_from_text(raw_text)

    return trades


def _extract_trades_from_text(text: str) -> list:
    """Fallback regex extraction for trades from raw text."""
    trades = []
    date_pat = re.compile(
        r"(\d{2}[/-]\d{2}[/-]\d{4})\s+"
        r"([A-Z][A-Za-z\s&.\-]+?)\s+"
        r"(BUY|SELL|B|S)\s+"
        r"(\d[\d,]*)\s+"
        r"([\d,]+\.?\d*)",
        re.IGNORECASE,
    )
    for m in date_pat.finditer(text):
        txn_type = m.group(3).upper()
        if txn_type in ("B",):
            txn_type = "BUY"
        elif txn_type in ("S",):
            txn_type = "SELL"
        name = normalize_stock_name(m.group(2))
        resolved = resolve_stock_name(name)
        trades.append({
            "stock_name": name,
            "standardized_symbol": resolved["canonical_symbol"],
            "date": parse_date(m.group(1)),
            "transaction_type": txn_type,
            "quantity": int(safe_float(m.group(4)) or 0),
            "price": safe_float(m.group(5)),
            "confidence": resolved["confidence"],
        })
    return trades


# ── Capital Gains / P&L Extraction ────────────────────────────────────────────

def extract_pnl(raw_text: str, raw_tables: list) -> dict:
    """Extract capital gains / P&L data."""
    result = {
        "realized_profit_loss": None,
        "unrealized_profit_loss": None,
        "holding_period": None,
        "total_return_percentage": None,
        "records": [],
    }

    text_lower = raw_text.lower()

    for pattern in [
        r"(?:realised|realized)\s*(?:p\s*&?\s*l|profit|gain)[:\s]*([-₹\d,. ]+)",
        r"(?:total\s+)?(?:realised|realized)[:\s]*([-₹\d,. ]+)",
    ]:
        m = re.search(pattern, text_lower)
        if m:
            result["realized_profit_loss"] = safe_float(m.group(1))
            break

    for pattern in [
        r"(?:unrealised|unrealized)\s*(?:p\s*&?\s*l|profit|gain)[:\s]*([-₹\d,. ]+)",
        r"(?:total\s+)?(?:unrealised|unrealized)[:\s]*([-₹\d,. ]+)",
    ]:
        m = re.search(pattern, text_lower)
        if m:
            result["unrealized_profit_loss"] = safe_float(m.group(1))
            break

    if "short term" in text_lower or "stcg" in text_lower:
        result["holding_period"] = "short"
    elif "long term" in text_lower or "ltcg" in text_lower:
        result["holding_period"] = "long"
    if "short term" in text_lower and "long term" in text_lower:
        result["holding_period"] = "mixed"

    m = re.search(r"(?:total\s+)?return[:\s]*([-\d,.]+)\s*%", text_lower)
    if m:
        result["total_return_percentage"] = safe_float(m.group(1))

    for table in raw_tables:
        if not table or len(table) < 2:
            continue
        header = [str(c).lower().strip() if c else "" for c in table[0]]
        has_pnl = any("p&l" in h or "profit" in h or "loss" in h or "gain" in h for h in header)
        if not has_pnl:
            continue

        col_map = {}
        for i, h in enumerate(header):
            if "stock" in h or "scrip" in h or "instrument" in h or "name" in h:
                col_map["stock_name"] = i
            elif "p&l" in h or "profit" in h or "gain" in h or "loss" in h:
                col_map["pnl"] = i
            elif ("buy" in h or "invested" in h) and "value" in h or h == "invested":
                col_map["buy_value"] = i
            elif ("sell" in h or "current" in h) and "value" in h:
                col_map["sell_value"] = i

        for row in table[1:]:
            if not row or all(c is None or str(c).strip() == "" for c in row):
                continue
            rec = {}
            for field, idx in col_map.items():
                if idx < len(row) and row[idx] is not None:
                    val = str(row[idx]).strip()
                    if field == "stock_name":
                        rec[field] = normalize_stock_name(val)
                    else:
                        rec[field] = safe_float(val)
            if rec.get("stock_name"):
                result["records"].append(rec)

    return result


# ── Master extraction function ────────────────────────────────────────────────

def extract_all(raw_text: str, raw_tables: list, document_types: list) -> dict:
    """Run all extractors and return structured data with quality flags."""
    data = {
        "holdings": [],
        "trades": [],
        "pnl": {
            "realized_profit_loss": None,
            "unrealized_profit_loss": None,
            "holding_period": None,
            "total_return_percentage": None,
            "records": [],
        },
        "data_quality_flags": [],
    }

    data["holdings"] = extract_holdings(raw_text, raw_tables)
    data["trades"] = extract_trades(raw_text, raw_tables)
    data["pnl"] = extract_pnl(raw_text, raw_tables)

    if not data["holdings"]:
        data["data_quality_flags"].append("No holdings data found")
    if not data["trades"]:
        data["data_quality_flags"].append("No trade/order data found")
    if data["pnl"]["realized_profit_loss"] is None and not data["pnl"]["records"]:
        data["data_quality_flags"].append("No P&L data found")

    # Check confidence levels
    low_conf = sum(1 for h in data["holdings"] if (h.get("confidence") or 0) < 0.75)
    if low_conf > 0:
        data["data_quality_flags"].append(f"{low_conf} stock(s) matched with low confidence")

    # Deduplicate holdings by stock name
    seen = set()
    unique_holdings = []
    for h in data["holdings"]:
        key = (h.get("stock_name"), h.get("symbol"))
        if key not in seen:
            seen.add(key)
            unique_holdings.append(h)
    data["holdings"] = unique_holdings

    return data
