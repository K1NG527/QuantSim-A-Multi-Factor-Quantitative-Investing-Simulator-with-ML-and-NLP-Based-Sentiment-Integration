"""
Document Analyzer — Full SaaS Pipeline Orchestrator
Stages: Document Understanding → Data Extraction → Market Enrichment →
        DB Storage → Feature Engineering → Risk Scoring → Longitudinal →
        NLP Insights → Recommendations → JSON Output
"""

import json
from datetime import datetime
from scripts.pdf_parser import parse_pdf
from scripts.data_extractor import extract_all
from scripts.risk_engine import analyze
from scripts.market_data import enrich_holdings, compute_sector_exposure
from scripts.longitudinal_engine import load_history, save_snapshot, compute_longitudinal


def analyze_document(pdf_files, user_id: str = "default", enrich: bool = True) -> dict:
    """
    End-to-end analysis pipeline.

    Args:
        pdf_files: a single file or a list of files (from st.file_uploader)
        user_id: user identifier for DB tracking
        enrich: whether to fetch real-time market data

    Returns:
        Stage 10 JSON-compatible dict.
    """
    timestamp = datetime.utcnow().isoformat()

    # ── Stage 1: Document Understanding ───────────────────────────────────
    # Supports multiple files automatically now
    parsed = parse_pdf(pdf_files)

    # ── Stage 2-3: Data Extraction & Cleaning ─────────────────────────────
    extracted = extract_all(
        raw_text=parsed["raw_text"],
        raw_tables=parsed["raw_tables"],
        document_types=parsed["document_types"],
    )

    # ── Stage 3: Real-Time Data Enrichment ────────────────────────────────
    enrichment_status = "not_enriched"
    if enrich and extracted["holdings"]:
        try:
            extracted["holdings"] = enrich_holdings(extracted["holdings"])
            enriched_count = sum(
                1 for h in extracted["holdings"]
                if h.get("market_data", {}).get("enriched") in ("yes", "partial")
            )
            if enriched_count > 0:
                enrichment_status = f"enriched ({enriched_count}/{len(extracted['holdings'])} stocks)"
            else:
                enrichment_status = "attempted_but_failed"
        except Exception as e:
            enrichment_status = f"error: {str(e)}"

    # ── Stages 4-8: Risk Analysis ─────────────────────────────────────────
    analysis = analyze(extracted)

    # ── Stage 7: Longitudinal Analysis ────────────────────────────────────
    history = load_history(user_id)
    
    # Build preliminary result for longitudinal comparison
    preliminary = {
        "portfolio_metrics": analysis["portfolio_metrics"],
        "behavioral_metrics": analysis["behavioral_metrics"],
        "performance_metrics": analysis["performance_metrics"],
        "risk_analysis": analysis["risk_analysis"],
        "investor_profile": analysis["investor_profile"],
    }
    longitudinal = compute_longitudinal(preliminary, history)

    # ── Stage 10: Assemble Output ─────────────────────────────────────────
    output = {
        "user_id": user_id,
        "document_summary": {
            "document_types": parsed["document_types"],
            "broker": parsed["broker"],
            "enrichment_status": enrichment_status,
        },
        "snapshot": {
            "timestamp": timestamp,
            "portfolio_value": str(analysis["portfolio_metrics"].get("total_portfolio_value", 0)),
            "risk_score": str(analysis["risk_analysis"]["risk_score"]),
            "risk_trend": longitudinal.get("risk_trend", "stable"),
        },
        "extracted_data": {
            "holdings_count": len(extracted["holdings"]),
            "trades_count": len(extracted["trades"]),
            "pnl_summary": {
                "realized_profit_loss": extracted["pnl"].get("realized_profit_loss"),
                "unrealized_profit_loss": extracted["pnl"].get("unrealized_profit_loss"),
                "holding_period": extracted["pnl"].get("holding_period"),
                "total_return_percentage": extracted["pnl"].get("total_return_percentage"),
            },
            "holdings": extracted["holdings"],
            "trades": extracted["trades"],
            "data_quality_flags": extracted.get("data_quality_flags", []),
        },
        "portfolio_metrics": analysis["portfolio_metrics"],
        "behavioral_metrics": analysis["behavioral_metrics"],
        "performance_metrics": analysis["performance_metrics"],
        "risk_analysis": {
            "risk_score": str(analysis["risk_analysis"]["risk_score"]),
            "sub_scores": {
                k: str(v) for k, v in analysis["risk_analysis"]["sub_scores"].items()
            },
            "explanation": analysis["risk_analysis"]["explanation"],
        },
        "investor_profile": analysis["investor_profile"],
        "historical_comparison": {
            "has_history": longitudinal["has_history"],
            "snapshots_count": longitudinal["snapshots_count"],
            "risk_trend": longitudinal["risk_trend"],
            "value_change": longitudinal.get("value_change"),
            "risk_change": longitudinal.get("risk_change"),
            "behavioral_shifts": longitudinal.get("behavioral_shifts", []),
            "previous_snapshot": longitudinal.get("previous_snapshot"),
        },
        "key_insights": analysis["insights"]["key_strengths"],
        "risk_factors": analysis["insights"]["key_risks"],
        "behavioral_observations": analysis["insights"]["behavioral_observations"],
        "recommendations": analysis["recommendations"],
    }

    # ── Stage 4: Save to Database ─────────────────────────────────────────
    try:
        from utils.db_models import init_db
        init_db()  # Ensure tables exist
        snapshot_id = save_snapshot(user_id, output)
        if snapshot_id:
            output["snapshot"]["snapshot_id"] = snapshot_id
    except Exception as e:
        output["snapshot"]["db_status"] = f"save_failed: {str(e)}"

    return output


def analyze_to_json(pdf_files, user_id: str = "default", enrich: bool = True) -> str:
    """Run analysis and return as formatted JSON string."""
    result = analyze_document(pdf_files, user_id=user_id, enrich=enrich)
    return json.dumps(result, indent=2, default=str)
