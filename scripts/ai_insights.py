"""
AI Portfolio Insights — QuantSim

Uses OpenAI GPT to generate intelligent portfolio analysis including:
    - Risk summary
    - Diversification analysis
    - Actionable improvement suggestions

Features:
    - Structured prompt engineering for financial analysis
    - Graceful fallback to rule-based insights if API unavailable
    - Rate limiting and retry logic
    - Configurable model and temperature

Usage:
    from scripts.ai_insights import generate_ai_insights
    insights = generate_ai_insights(portfolio_data)
"""

import os
import json
import logging
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior quantitative portfolio analyst with 20 years 
of experience in factor investing, risk management, and portfolio construction. 
You provide institutional-quality analysis in a clear, actionable format.

Your analysis must include:
1. RISK SUMMARY — Key risk exposures and their implications
2. DIVERSIFICATION ANALYSIS — Asset allocation assessment, sector concentration, 
   factor exposures with specific recommendations
3. IMPROVEMENT SUGGESTIONS — 3-5 specific, actionable steps ranked by priority

Use precise financial terminology. Cite specific metrics from the data provided. 
Be concise but thorough. Format your response with clear headers."""


def _build_analysis_prompt(portfolio_data: dict) -> str:
    """Build the analysis prompt from portfolio data."""
    data_str = json.dumps(portfolio_data, indent=2, default=str)
    return f"""Analyze this portfolio and provide comprehensive insights:

PORTFOLIO DATA:
{data_str}

Provide your analysis with:
1. **Risk Summary** — Evaluate the risk profile, identify key risk drivers, 
   and assess whether the risk level is appropriate.
2. **Diversification Analysis** — Assess sector concentration, position sizing, 
   factor tilts, and correlation exposure.
3. **Improvement Suggestions** — Provide 3-5 specific, actionable recommendations 
   ranked by impact. Include specific allocation changes where relevant.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# OPENAI API INTERACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _call_openai(prompt: str, system_prompt: str = SYSTEM_PROMPT,
                 model: str = "gpt-4o-mini", temperature: float = 0.4,
                 max_tokens: int = 1500, max_retries: int = 3) -> Optional[str]:
    """
    Call OpenAI API with retry logic.

    Args:
        prompt:        User prompt
        system_prompt: System prompt for role definition
        model:         OpenAI model name
        temperature:   Sampling temperature (0-1)
        max_tokens:    Max response tokens
        max_retries:   Number of retry attempts

    Returns:
        Response text string, or None on failure
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_key_here":
        logger.warning("OPENAI_API_KEY not configured")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content
            logger.info(f"OpenAI response received ({len(result)} chars)")
            return result

        except Exception as e:
            logger.warning(f"OpenAI API attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error("All OpenAI API attempts failed")
                return None


# ═══════════════════════════════════════════════════════════════════════════════
# RULE-BASED FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_rule_based_insights(portfolio_data: dict) -> dict:
    """
    Generate rule-based portfolio insights as fallback when OpenAI is unavailable.

    Args:
        portfolio_data: Dict with portfolio_metrics, risk_analysis, factor_exposures

    Returns:
        Dict with risk_summary, diversification_analysis, suggestions
    """
    metrics = portfolio_data.get("portfolio_metrics", {})
    risk = portfolio_data.get("risk_analysis", {})
    sector_exp = metrics.get("sector_exposure", {})

    risk_score = risk.get("risk_score", 5)
    n_stocks = metrics.get("number_of_stocks", 0)
    conc = metrics.get("top_3_concentration_ratio", 0)
    div_score = metrics.get("diversification_score", 0)
    vol = metrics.get("portfolio_volatility")

    # Risk summary
    if risk_score >= 7:
        risk_summary = (
            f"⚠️ HIGH RISK: Portfolio risk score is {risk_score}/10. "
            f"Volatility {'is elevated at ' + f'{vol:.2%}' if vol else 'data unavailable'}. "
            f"Top-3 concentration at {conc:.1f}% indicates significant exposure to individual names."
        )
    elif risk_score >= 4:
        risk_summary = (
            f"📊 MODERATE RISK: Portfolio risk score is {risk_score}/10. "
            f"Risk levels are within acceptable bounds but could be improved with better diversification."
        )
    else:
        risk_summary = (
            f"✅ LOW RISK: Portfolio risk score is {risk_score}/10. "
            f"Well-positioned with {'strong' if div_score > 0.7 else 'adequate'} diversification."
        )

    # Diversification analysis
    div_parts = [f"Portfolio holds {n_stocks} stocks with diversification score of {div_score:.2f}."]
    if sector_exp:
        top_sector = max(sector_exp.items(), key=lambda x: x[1], default=("N/A", 0))
        div_parts.append(f"Top sector: {top_sector[0]} at {top_sector[1]:.1f}%.")
        if top_sector[1] >= 50:
            div_parts.append("Significant sector concentration risk detected.")
    diversification = " ".join(div_parts)

    # Suggestions
    suggestions = []
    if conc >= 50:
        suggestions.append(f"Reduce top-3 concentration from {conc:.1f}% to below 40% by redistributing capital.")
    if n_stocks < 10:
        suggestions.append(f"Increase portfolio breadth from {n_stocks} to 15-20 holdings for better risk-adjusted returns.")
    if vol and vol > 0.3:
        suggestions.append(f"Add low-volatility holdings (utilities, consumer staples) to reduce portfolio volatility from {vol:.2%}.")
    if sector_exp:
        top = max(sector_exp.items(), key=lambda x: x[1], default=("N/A", 0))
        if top[1] >= 40:
            suggestions.append(f"Diversify sector exposure: {top[0]} at {top[1]:.1f}% is too concentrated.")
    if not suggestions:
        suggestions.append("Portfolio is well-structured. Continue monitoring and maintain discipline.")

    return {
        "risk_summary": risk_summary,
        "diversification_analysis": diversification,
        "suggestions": suggestions,
        "source": "rule_based_fallback",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ai_insights(portfolio_data: dict,
                         model: str = "gpt-4o-mini",
                         temperature: float = 0.4) -> dict:
    """
    Generate AI-powered portfolio insights.

    Tries OpenAI GPT first; falls back to rule-based analysis if unavailable.

    Args:
        portfolio_data: Dict containing:
            - portfolio_metrics: dict with total_value, n_stocks, concentration, etc.
            - risk_analysis: dict with risk_score, sub_scores
            - factor_exposures: dict with factor weights (optional)
            - behavioral_metrics: dict with trading patterns (optional)
        model:       OpenAI model name (default: gpt-4o-mini)
        temperature: Sampling temperature (default: 0.4)

    Returns:
        Dict with:
            - risk_summary (str)
            - diversification_analysis (str)
            - suggestions (list[str])
            - raw_response (str, only if OpenAI used)
            - source ("openai" or "rule_based_fallback")

    Example:
        >>> data = {
        ...     "portfolio_metrics": {"total_value": 500000, "number_of_stocks": 12},
        ...     "risk_analysis": {"risk_score": 6.5},
        ... }
        >>> insights = generate_ai_insights(data)
        >>> print(insights["risk_summary"])
    """
    # Try OpenAI first
    prompt = _build_analysis_prompt(portfolio_data)
    response = _call_openai(prompt, model=model, temperature=temperature)

    if response:
        # Parse structured response
        sections = _parse_ai_response(response)
        sections["raw_response"] = response
        sections["source"] = "openai"
        logger.info("AI insights generated via OpenAI")
        return sections

    # Fallback to rule-based
    logger.info("Falling back to rule-based insights")
    return _generate_rule_based_insights(portfolio_data)


def _parse_ai_response(response: str) -> dict:
    """
    Parse OpenAI response into structured sections.

    Tries to extract Risk Summary, Diversification Analysis, and Suggestions
    from the markdown-formatted response.
    """
    result = {
        "risk_summary": "",
        "diversification_analysis": "",
        "suggestions": [],
    }

    lines = response.strip().split("\n")
    current_section = None

    for line in lines:
        line_lower = line.lower().strip()

        if "risk summary" in line_lower or "risk" in line_lower and "##" in line:
            current_section = "risk"
            continue
        elif "diversification" in line_lower and ("##" in line or "**" in line):
            current_section = "diversification"
            continue
        elif ("suggestion" in line_lower or "improvement" in line_lower or
              "recommendation" in line_lower) and ("##" in line or "**" in line):
            current_section = "suggestions"
            continue

        # Accumulate content
        clean = line.strip()
        if not clean:
            continue

        if current_section == "risk":
            result["risk_summary"] += clean + " "
        elif current_section == "diversification":
            result["diversification_analysis"] += clean + " "
        elif current_section == "suggestions":
            # Collect numbered/bulleted items
            if clean.startswith(("-", "•", "*")) or (len(clean) > 2 and clean[0].isdigit() and clean[1] in ".):"):
                suggestion = clean.lstrip("-•* 0123456789.):").strip()
                if suggestion:
                    result["suggestions"].append(suggestion)

    result["risk_summary"] = result["risk_summary"].strip()
    result["diversification_analysis"] = result["diversification_analysis"].strip()

    # If parsing failed, use the whole response
    if not result["risk_summary"] and not result["diversification_analysis"]:
        result["risk_summary"] = response[:500]
        result["diversification_analysis"] = "See raw response for details."
        result["suggestions"] = ["Review the full AI analysis for detailed recommendations."]

    return result


def generate_quick_summary(portfolio_data: dict) -> str:
    """
    Generate a short one-paragraph AI summary of the portfolio.

    Args:
        portfolio_data: Portfolio metrics dict

    Returns:
        String with 2-3 sentence portfolio summary

    Example:
        >>> summary = generate_quick_summary({"portfolio_metrics": {...}})
        >>> print(summary)
    """
    prompt = f"""Provide a 2-3 sentence summary of this portfolio's health and key action items:
{json.dumps(portfolio_data, indent=2, default=str)}"""

    response = _call_openai(prompt, max_tokens=200, temperature=0.3)
    if response:
        return response.strip()

    # Fallback
    metrics = portfolio_data.get("portfolio_metrics", {})
    risk = portfolio_data.get("risk_analysis", {})
    return (
        f"Portfolio has {metrics.get('number_of_stocks', 'N/A')} holdings "
        f"with a risk score of {risk.get('risk_score', 'N/A')}/10. "
        f"Top-3 concentration is {metrics.get('top_3_concentration_ratio', 'N/A')}%."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example portfolio data
    sample_data = {
        "portfolio_metrics": {
            "total_portfolio_value": 1245000,
            "number_of_stocks": 12,
            "top_3_concentration_ratio": 45.2,
            "diversification_score": 0.72,
            "sector_exposure": {
                "Technology": 38.5,
                "Healthcare": 22.1,
                "Financial": 18.3,
                "Consumer": 12.8,
                "Energy": 8.3,
            },
            "portfolio_volatility": 0.28,
        },
        "risk_analysis": {
            "risk_score": 5.5,
            "sub_scores": {
                "concentration_score": 6,
                "behavior_score": 4,
                "performance_score": 5,
                "market_risk_score": 6,
            },
        },
        "factor_exposures": {
            "value": 0.35,
            "momentum": 0.55,
            "quality": 0.68,
            "volatility": 0.42,
            "volume": 0.30,
        },
    }

    print("=== AI Portfolio Insights ===")
    insights = generate_ai_insights(sample_data)
    print(f"\nSource: {insights['source']}")
    print(f"\nRisk Summary:\n{insights['risk_summary']}")
    print(f"\nDiversification:\n{insights['diversification_analysis']}")
    print(f"\nSuggestions:")
    for i, s in enumerate(insights["suggestions"], 1):
        print(f"  {i}. {s}")
