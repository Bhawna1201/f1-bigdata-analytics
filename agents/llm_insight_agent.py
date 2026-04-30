import os
import json
from datetime import datetime


def get_llm_briefing(state):
    """Generate a natural-language race briefing using Groq LLaMA"""
    try:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print("No GROQ_API_KEY found - falling back to template briefing")
            return get_template_briefing(state)

        client = Groq(api_key=api_key)

        data_health = state.get("data_health", {})
        features = state.get("new_features_accepted", [])
        features_rejected = state.get("new_features_rejected", [])
        model_perf = state.get("model_performance", {})
        test_results = state.get("feature_test_results", {})

        # Build feature strings safely
        accepted_str = ""
        for f in features:
            name = f if isinstance(f, str) else f.get("name", str(f))
            corr = test_results.get(name, {}).get("correlation", "N/A") if isinstance(test_results, dict) else "N/A"
            accepted_str += "  - " + str(name) + " (correlation: " + str(corr) + ")\n"

        rejected_str = ""
        for f in features_rejected:
            name = f if isinstance(f, str) else f.get("name", str(f))
            corr = test_results.get(name, {}).get("correlation", "N/A") if isinstance(test_results, dict) else "N/A"
            rejected_str += "  - " + str(name) + " (correlation: " + str(corr) + ")\n"

        prompt = (
            "You are an F1 race strategy analyst. Based on the following AI agent outputs, "
            "write a concise pre-race strategy briefing (200-300 words) that a pit wall engineer could use.\n\n"
            "DATA QUALITY REPORT:\n"
            "- Status: " + str(data_health.get("status", "N/A")) + "\n"
            "- Total records: " + str(data_health.get("row_count", data_health.get("total_rows", "N/A"))) + "\n"
            "- Issues: " + str(data_health.get("issues_count", 0)) + "\n\n"
            "FEATURE DISCOVERY:\n"
            "Accepted features:\n" + (accepted_str if accepted_str else "  None\n") +
            "Rejected features:\n" + (rejected_str if rejected_str else "  None\n") + "\n"
            "MODEL PERFORMANCE:\n"
            "- Position MAE: " + str(model_perf.get("position_predictor", {}).get("mae_2025", "N/A")) + "\n"
            "- Worst circuits: " + str(model_perf.get("worst_circuits_2025", "N/A")) + "\n"
            "- Hardest drivers: " + str(model_perf.get("hardest_to_predict_2025", "N/A")) + "\n\n"
            "Write the briefing with these sections:\n"
            "1. DATA HEALTH - one sentence summary\n"
            "2. NEW INTELLIGENCE - what the Feature Agent discovered and why it matters\n"
            "3. MODEL CONFIDENCE - where predictions are strong vs weak\n"
            "4. RECOMMENDED ACTIONS - 2-3 specific things to watch for\n\n"
            "Keep it professional but accessible. Use F1 terminology where appropriate. "
            "Do NOT include any disclaimers or meta-commentary."
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        briefing = response.choices[0].message.content

        return briefing

    except ImportError:
        print("groq not installed - pip install groq")
        return get_template_briefing(state)
    except Exception as e:
        print("LLM error: " + str(e) + " - falling back to template")
        return get_template_briefing(state)


def get_template_briefing(state):
    """Fallback template briefing when LLM is unavailable"""
    data_health = state.get("data_health", {})
    features = state.get("new_features_accepted", [])
    test_results = state.get("feature_test_results", {})
    model_perf = state.get("model_performance", {})
    mae = model_perf.get("position_predictor", {}).get("mae_2025", "N/A")

    feature_lines = ""
    for f in features:
        name = f if isinstance(f, str) else f.get("name", str(f))
        corr = test_results.get(name, {}).get("correlation", "N/A") if isinstance(test_results, dict) else "N/A"
        feature_lines += "  - " + str(name) + ": correlation = " + str(corr) + "\n"

    return (
        "DATA HEALTH: " + str(data_health.get("status", "UNKNOWN")) + "\n"
        "Records: " + str(data_health.get("row_count", data_health.get("total_rows", "N/A"))) + "\n\n"
        "NEW FEATURES DISCOVERED: " + str(len(features)) + " accepted\n"
        + (feature_lines if feature_lines else "  None this cycle\n") + "\n"
        "MODEL PERFORMANCE: MAE = " + str(mae) + "\n"
        "Retrain triggered: " + str(state.get("retrain_triggered", False)) + "\n\n"
        "Note: Set GROQ_API_KEY for AI-powered briefings."
    )


if __name__ == "__main__":
    test_state = {
        "data_health": {"status": "HEALTHY", "row_count": 3524,
                        "null_rates": {"q3_seconds": 0.515, "grid": 0.002}},
        "new_features_accepted": ["constructor_dominance", "home_advantage"],
        "feature_test_results": {
            "constructor_dominance": {"correlation": 0.594},
            "home_advantage": {"correlation": 0.167},
        },
        "model_performance": {
            "position_predictor": {"mae_2025": 2.096},
            "worst_circuits_2025": {"Zandvoort": 4.1, "Silverstone": 3.8},
            "hardest_to_predict_2025": {"Antonelli": 3.9}
        },
        "retrain_triggered": False
    }

    result = get_llm_briefing(test_state)
    print(result)
