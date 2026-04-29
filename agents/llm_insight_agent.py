"""
LLM-Powered Insight Agent — Drop into agents/agentic_pipeline.py
Replaces the string-formatting briefing with real Gemini LLM output.

Setup:
  pip install google-generativeai
  
  Add to your .env or environment:
  GEMINI_API_KEY=your_key_here
  
  Get free key: https://aistudio.google.com/apikey
"""

import os
import json
from datetime import datetime

# ── Option A: Google Gemini (Free tier = 15 RPM) ──
def get_llm_briefing(state):
    """Generate a natural-language race briefing using Gemini LLM"""
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("⚠️ No GEMINI_API_KEY found — falling back to template briefing")
            return get_template_briefing(state)
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Build context from agent state
        data_health = state.get("data_health", {})
        features = state.get("new_features_accepted", [])
        features_rejected = state.get("new_features_rejected", [])
        model_perf = state.get("model_performance", {})
        
        prompt = f"""You are an F1 race strategy analyst. Based on the following AI agent outputs, 
write a concise pre-race strategy briefing (200-300 words) that a pit wall engineer could use.

DATA QUALITY REPORT:
- Status: {data_health.get('status', 'N/A')}
- Total records: {data_health.get('row_count', 'N/A')}
- Null rates: {json.dumps(data_health.get('null_rates', {}), indent=2)[:500]}

FEATURE DISCOVERY:
- Accepted features: {json.dumps(features, indent=2)[:500]}
- Rejected features: {json.dumps(features_rejected, indent=2)[:300]}

MODEL PERFORMANCE:
- Position MAE (2025-2026): {model_perf.get('position_predictor', {}).get('mae_2025', 'N/A')}
- Worst circuits: {json.dumps(model_perf.get('worst_circuits', []), indent=2)[:300]}
- Hardest drivers: {json.dumps(model_perf.get('hardest_drivers', []), indent=2)[:300]}

Write the briefing with these sections:
1. DATA HEALTH — one sentence summary
2. NEW INTELLIGENCE — what the Feature Agent discovered and why it matters
3. MODEL CONFIDENCE — where predictions are strong vs weak
4. RECOMMENDED ACTIONS — 2-3 specific things to watch for

Keep it professional but accessible. Use F1 terminology where appropriate.
Do NOT include any disclaimers or meta-commentary."""

        response = model.generate_content(prompt)
        briefing = response.text
        
        # Add timestamp
        briefing = f"═══ AI STRATEGY BRIEFING ═══\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nPowered by: Gemini 2.0 Flash + LangGraph Agents\n{'═' * 40}\n\n{briefing}"
        
        print("✅ LLM briefing generated successfully")
        return briefing
        
    except ImportError:
        print("⚠️ google-generativeai not installed — pip install google-generativeai")
        return get_template_briefing(state)
    except Exception as e:
        print(f"⚠️ LLM error: {e} — falling back to template")
        return get_template_briefing(state)


def get_template_briefing(state):
    """Fallback template briefing when LLM is unavailable"""
    data_health = state.get("data_health", {})
    features = state.get("new_features_accepted", [])
    model_perf = state.get("model_performance", {})
    mae = model_perf.get("position_predictor", {}).get("mae_2025", "N/A")
    
    return f"""═══ STRATEGY BRIEFING (Template) ═══
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

DATA HEALTH: {data_health.get('status', 'UNKNOWN')}
Records: {data_health.get('row_count', 'N/A')}

NEW FEATURES DISCOVERED: {len(features)} accepted
{chr(10).join(f"  • {f.get('name', f)}: corr={f.get('correlation', 'N/A')}" for f in features) if features else '  None this cycle'}

MODEL PERFORMANCE: MAE = {mae}
Retrain triggered: {state.get('retrain_triggered', False)}

Note: Set GEMINI_API_KEY for AI-powered briefings."""


# ── Integration into your existing insight_agent function ──
def insight_agent_with_llm(state):
    """
    Drop-in replacement for your existing insight_agent function.
    
    In agents/agentic_pipeline.py, replace:
        def insight_agent(state):
            ...template string...
    
    With:
        from llm_insight_agent import insight_agent_with_llm as insight_agent
    """
    # Generate briefing
    briefing = get_llm_briefing(state)
    state["latest_briefing"] = briefing
    
    # Save to memory (if memory module is available)
    try:
        from agent_memory import save_agent_memory
        save_agent_memory(state)
    except ImportError:
        pass
    
    return state


if __name__ == "__main__":
    # Test with sample state
    test_state = {
        "data_health": {"status": "HEALTHY", "row_count": 3524, 
                        "null_rates": {"q3_seconds": 0.515, "grid": 0.002}},
        "new_features_accepted": [
            {"name": "constructor_dominance", "correlation": 0.594},
            {"name": "home_advantage", "correlation": 0.167},
        ],
        "new_features_rejected": [
            {"name": "quali_trend", "correlation": 0.044},
        ],
        "model_performance": {
            "position_predictor": {"mae_2025": 2.096},
            "worst_circuits": ["Zandvoort (4.1)", "Silverstone (3.8)"],
            "hardest_drivers": ["Antonelli (rookie)"]
        },
        "retrain_triggered": False
    }
    
    result = get_llm_briefing(test_state)
    print(result)
