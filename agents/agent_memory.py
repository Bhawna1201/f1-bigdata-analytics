"""
Agent Memory System — Persistent memory for AI agents
Agents save their outputs after each run. Future agents read past outputs
to detect trends, compare performance, and learn from previous cycles.

Setup:
  Drop this file into agents/agent_memory.py
  
  In your agentic_pipeline.py, add:
    from agent_memory import save_agent_memory, load_memory, get_memory_context

Usage:
  # At end of pipeline (after insight agent):
  save_agent_memory(state)
  
  # At start of pipeline (before data agent):
  past_runs = load_memory()
  state["memory_context"] = get_memory_context(past_runs)
"""

import json
import os
from datetime import datetime

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "agent_memory.json")


def load_memory():
    """Load all past agent runs from memory file"""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_agent_memory(state):
    """
    Save current agent run to persistent memory.
    Call this at the end of the agentic pipeline, after all agents have run.
    """
    memory = load_memory()
    
    # Extract key metrics from state
    entry = {
        "timestamp": datetime.now().isoformat(),
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        
        # Data Agent outputs
        "data_health": {
            "status": state.get("data_health", {}).get("status", "UNKNOWN"),
            "row_count": state.get("data_health", {}).get("row_count", 0),
            "key_null_rates": {
                k: round(v, 3) for k, v in 
                list(state.get("data_health", {}).get("null_rates", {}).items())[:5]
            }
        },
        
        # Feature Agent outputs
        "features_proposed": len(state.get("new_features_proposed", [])),
        "features_accepted": [
            {"name": f.get("name", str(f)), "correlation": round(f.get("correlation", 0), 3)}
            if isinstance(f, dict) else {"name": str(f), "correlation": state.get("feature_test_results", {}).get(str(f), {}).get("correlation", 0)}
            for f in state.get("new_features_accepted", [])
        ],
        "features_rejected": [
            {"name": f.get("name", str(f)), "correlation": round(f.get("correlation", 0), 3)}
            if isinstance(f, dict) else {"name": str(f), "correlation": state.get("feature_test_results", {}).get(str(f), {}).get("correlation", 0)}
            for f in state.get("new_features_rejected", state.get("new_features_proposed", []))
            if f not in state.get("new_features_accepted", [])
        ],
        
        # Model Agent outputs
        "model_mae": state.get("model_performance", {}).get(
            "position_predictor", {}).get("mae_2025", None),
        "worst_circuits": state.get("model_performance", {}).get("worst_circuits", [])[:3],
        "retrain_triggered": state.get("retrain_triggered", False),
        
        # Briefing summary (first 300 chars)
        "briefing_preview": state.get("latest_briefing", "")[:300],
    }
    
    memory.append(entry)
    
    # Keep last 20 runs to prevent file from growing forever
    memory = memory[-20:]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)
    
    print(f"✅ Agent memory saved — {len(memory)} runs in history")
    return entry


def get_memory_context(memory=None):
    """
    Generate a context string from past agent runs.
    Pass this to the Insight Agent or LLM for trend-aware briefings.
    
    Returns a string like:
    "In the last 5 runs: MAE improved from 2.3 to 2.1. 
     Feature Agent discovered 8 features total, accepted 5.
     No retraining was triggered. Data health was consistently HEALTHY."
    """
    if memory is None:
        memory = load_memory()
    
    if not memory:
        return "No previous agent runs in memory. This is the first execution."
    
    # Analyze trends
    recent = memory[-5:]  # Last 5 runs
    
    context_parts = [f"Agent Memory: {len(memory)} total runs, analyzing last {len(recent)}."]
    
    # MAE trend
    maes = [r.get("model_mae") for r in recent if r.get("model_mae") is not None]
    if len(maes) >= 2:
        trend = "improving" if maes[-1] < maes[0] else "degrading" if maes[-1] > maes[0] else "stable"
        context_parts.append(
            f"Position MAE trend: {maes[0]:.3f} → {maes[-1]:.3f} ({trend})."
        )
    elif len(maes) == 1:
        context_parts.append(f"Latest Position MAE: {maes[0]:.3f}.")
    
    # Feature discovery
    total_accepted = sum(len(r.get("features_accepted", [])) for r in recent)
    total_proposed = sum(r.get("features_proposed", 0) for r in recent)
    if total_proposed > 0:
        context_parts.append(
            f"Feature Agent: {total_accepted} accepted out of {total_proposed} proposed "
            f"across last {len(recent)} runs."
        )
    
    # Data health
    health_statuses = [r.get("data_health", {}).get("status", "UNKNOWN") for r in recent]
    if all(s == "HEALTHY" for s in health_statuses):
        context_parts.append("Data health: Consistently HEALTHY across all recent runs.")
    else:
        unhealthy = sum(1 for s in health_statuses if s != "HEALTHY")
        context_parts.append(f"Data health: {unhealthy}/{len(recent)} recent runs had issues.")
    
    # Retraining
    retrain_count = sum(1 for r in recent if r.get("retrain_triggered", False))
    if retrain_count > 0:
        context_parts.append(f"Retraining was triggered {retrain_count} times in last {len(recent)} runs.")
    else:
        context_parts.append("No retraining triggered recently — model performance is acceptable.")
    
    # Worst circuits consistency
    all_worst = []
    for r in recent:
        all_worst.extend(r.get("worst_circuits", []))
    if all_worst:
        from collections import Counter
        recurring = Counter(all_worst).most_common(3)
        if recurring:
            circuits = ", ".join(f"{c[0]} ({c[1]}x)" for c in recurring)
            context_parts.append(f"Recurring weak circuits: {circuits}.")
    
    return "\n".join(context_parts)


def get_memory_for_display():
    """
    Format memory for Streamlit dashboard display.
    Returns a list of dicts ready for st.dataframe().
    """
    memory = load_memory()
    if not memory:
        return []
    
    display = []
    for run in memory:
        display.append({
            "Date": run.get("run_date", "N/A"),
            "Data Health": run.get("data_health", {}).get("status", "N/A"),
            "MAE": run.get("model_mae", "N/A"),
            "Features Found": len(run.get("features_accepted", [])),
            "Retrain": "Yes" if run.get("retrain_triggered") else "No",
            "Worst Circuit": run.get("worst_circuits", ["N/A"])[0] if run.get("worst_circuits") else "N/A",
        })
    
    return display


# ── Standalone test ──
if __name__ == "__main__":
    # Simulate saving a run
    test_state = {
        "data_health": {"status": "HEALTHY", "row_count": 3524,
                        "null_rates": {"q3_seconds": 0.515, "grid": 0.002}},
        "new_features_proposed": [
            {"name": "constructor_dominance", "correlation": 0.594},
            {"name": "home_advantage", "correlation": 0.167},
            {"name": "grid_volatility", "correlation": 0.064},
            {"name": "quali_trend", "correlation": 0.044},
            {"name": "points_acceleration", "correlation": 0.019},
        ],
        "new_features_accepted": [
            {"name": "constructor_dominance", "correlation": 0.594},
            {"name": "home_advantage", "correlation": 0.167},
            {"name": "grid_volatility", "correlation": 0.064},
        ],
        "model_performance": {
            "position_predictor": {"mae_2025": 2.096},
            "worst_circuits": ["Zandvoort (4.1)", "Silverstone (3.8)", "Las Vegas (3.8)"],
        },
        "retrain_triggered": False,
        "latest_briefing": "Data quality is healthy. Constructor dominance discovered as strong feature...",
    }
    
    print("Saving test run...")
    save_agent_memory(test_state)
    
    print("\nMemory context:")
    print(get_memory_context())
    
    print("\nDisplay data:")
    for row in get_memory_for_display():
        print(row)
