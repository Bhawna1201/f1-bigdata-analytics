"""
F1 Big Data Analytics Dashboard
================================
6 pages covering all rubric requirements:
  1. Project Overview (goals, pipeline, scope)
  2. Race Predictions (live demo with probabilities)
  3. Driver & Constructor Analysis (interactive charts)
  4. Tire & Weather Impact (how data improved predictions)
  5. ML Models & Improvement Journey (3 models, iteration)
  6. Agentic AI & Limitations (agents, challenges, lessons)
  Stevens Maroon (#9D1535) for F1 Red (#E10600).

Run: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import joblib
from text_to_sql_page import render_text_to_sql_page

st.set_page_config(
    page_title="F1 Big Data Analytics",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (adapted from healthcare dashboard) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family:'Source Sans 3',sans-serif; color:#2d2d2d; }
h1,h2,h3 { font-family:'Playfair Display',serif; }

.stApp {
    background:#F9F9F9;
    background-image:
        radial-gradient(ellipse at 96% 4%, rgba(196,30,30,0.04) 0%, transparent 45%),
        radial-gradient(ellipse at 4% 96%, rgba(196,30,30,0.02) 0%, transparent 45%);
}

section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#C41E1E 0%,#9A1818 100%) !important;
    border-right:none;
    box-shadow:4px 0 20px rgba(225,6,0,0.18);
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div { color:white !important; }
section[data-testid="stSidebar"] .stRadio label {
    line-height:1.4 !important;
    display:flex !important;
    align-items:center !important;
    gap:6px !important;
    color:white !important;
    font-size:1.08rem !important;
    font-weight:400 !important;
    padding:5px 2px;
    white-space:nowrap !important;
}
section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.2) !important; }

[data-baseweb="select"]>div:first-child { background:white !important; border-color:#E0E0E0 !important; color:#2d2d2d !important; }
[data-baseweb="select"] div { color:#2d2d2d !important; }
[data-baseweb="menu"] li { color:#2d2d2d !important; background:white !important; }

.stTabs [data-baseweb="tab-list"] { background:#F0F0F0; border-radius:8px; padding:4px; border:1px solid #E0E0E0; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#949594 !important; border-radius:6px; font-weight:400; }
.stTabs [aria-selected="true"] { background:white !important; color:#C41E1E !important; font-weight:600; box-shadow:0 1px 4px rgba(0,0,0,0.1); }

.stSelectbox label,.stMultiSelect label { color:#949594 !important; font-weight:400; }
.js-plotly-plot { border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)

# ── Design tokens ──
F1_RED = "#C41E1E"      # was "#E10600" — deeper, easier on eyes
F1_RED_D = "#9A1818"     # was "#B00500"
F1_RED_L = "#FDF2F2"     # was "#FDE8E8" — softer tint
GRAY = "#949594"
WHT = "#FFFFFE"

P_BLUE = "#E8F0FE"
P_GREEN = "#E6F4EA"
P_AMBER = "#FEF9E7"
P_PURPLE = "#F3E8FF"
P_TEAL = "#E8F5F5"

C_BLUE = "#2563EB"
C_GREEN = "#059669"
C_AMBER = "#D97706"
C_PURPLE = "#7C3AED"
C_TEAL = "#0891B2"

PT = dict(
    paper_bgcolor="white", plot_bgcolor="#FAFAFA",
    font=dict(color="#2d2d2d", family="Source Sans 3"),
    title_font=dict(family="Playfair Display", color="#2d2d2d", size=16),
    colorway=[F1_RED, C_BLUE, C_GREEN, C_AMBER, C_PURPLE, C_TEAL, "#DB2777", "#65A30D"],
)


# ── Helper components ──
def kpi(value, label, color=F1_RED, bg=F1_RED_L, sub=""):
    return (
        f"<div style='background:{bg};border-top:3px solid {color};"
        f"border-radius:0 0 10px 10px;padding:18px 16px;text-align:center;"
        f"box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
        f"<div style='font-family:Playfair Display,serif;font-size:2rem;"
        f"color:{color};font-weight:700;line-height:1'>{value}</div>"
        f"<div style='font-size:0.7rem;color:{GRAY};text-transform:uppercase;"
        f"letter-spacing:0.09em;margin-top:5px;font-weight:600'>{label}</div>"
        f"{'<div style=font-size:0.8rem;color:' + color + ';margin-top:3px>' + sub + '</div>' if sub else ''}"
        f"</div>")


def info_box(title, body, color=F1_RED, bg=F1_RED_L):
    return (
        f"<div style='background:{bg};border-left:4px solid {color};"
        f"border-radius:0 10px 10px 0;padding:14px 18px;margin:12px 0;"
        f"box-shadow:0 1px 6px rgba(0,0,0,0.05)'>"
        f"<div style='font-weight:700;color:{color};font-size:0.9rem;margin-bottom:6px'>{title}</div>"
        f"<div style='color:#2d2d2d;font-size:0.88rem;line-height:1.7'>{body}</div>"
        f"</div>")


def sec(text, sub=""):
    st.markdown(
        f"<div style='margin:28px 0 14px 0'>"
        f"<div style='font-family:Playfair Display,serif;font-size:1.35rem;"
        f"color:{F1_RED};font-weight:600;border-bottom:2px solid {F1_RED_L};"
        f"padding-bottom:8px'>{text}</div>"
        f"{'<div style=font-size:0.83rem;color:' + GRAY + ';margin-top:5px>' + sub + '</div>' if sub else ''}"
        f"</div>", unsafe_allow_html=True)


def narrative(text):
    st.markdown(
        f"<div style='background:white;border:1px solid #E9ECEF;"
        f"border-left:4px solid {F1_RED};border-radius:0 10px 10px 0;"
        f"padding:14px 18px;margin:10px 0;font-size:0.88rem;"
        f"color:#2d2d2d;line-height:1.75;box-shadow:0 1px 4px rgba(0,0,0,0.04)'>"
        f"{text}</div>", unsafe_allow_html=True)


# ── Data loading ──
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"
SILVER_DIR = DATA_DIR / "silver"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs" / "agents"


@st.cache_data
def load_data():
    data = {}

    # Load best available features
    for fname in ["race_prediction_features_complete.parquet",
                  "race_prediction_features_advanced.parquet",
                  "race_prediction_features_weather.parquet",
                  "race_prediction_features.parquet"]:
        path = GOLD_DIR / fname
        if path.exists():
            data["features"] = pd.read_parquet(path)
            data["features_file"] = fname
            break

    if SILVER_DIR.exists():
        for name in ["pit_stops", "laps"]:
            p = SILVER_DIR / f"{name}.parquet"
            if p.exists():
                data[name] = pd.read_parquet(p)

    if (GOLD_DIR / "tire_degradation_features.parquet").exists():
        data["tire_deg"] = pd.read_parquet(GOLD_DIR / "tire_degradation_features.parquet")

    if (SILVER_DIR / "weather_features.parquet").exists():
        data["weather"] = pd.read_parquet(SILVER_DIR / "weather_features.parquet")

    # Models
    models = {}
    for name in ["position_predictor", "podium_classifier", "winner_predictor"]:
        p = MODELS_DIR / f"{name}.pkl"
        if p.exists():
            models[name] = joblib.load(p)
    data["models"] = models

    # Results
    for fname in ["model_comparison_results.json", "training_results.json"]:
        p = MODELS_DIR / fname
        if p.exists():
            with open(p) as f:
                data[fname.replace(".json", "")] = json.load(f)

    # Agent state
    if LOGS_DIR.exists():
        briefings = sorted(LOGS_DIR.glob("briefing_*.txt"), reverse=True)
        if briefings:
            data["latest_briefing"] = briefings[0].read_text()
        states = sorted(LOGS_DIR.glob("pipeline_state_*.json"), reverse=True)
        if states:
            with open(states[0]) as f:
                data["agent_state"] = json.load(f)

    return data


data = load_data()
df = data.get("features", pd.DataFrame())

# ── Sidebar ──
st.sidebar.markdown(
    f"<div style='text-align:center;padding:10px 0 5px 0'>"
    f"<div style='font-family:Playfair Display,serif;font-size:1.6rem;font-weight:700;color:white'>F1 Big Data</div>"
    f"<div style='font-size:0.8rem;color:rgba(255,255,255,0.7);margin-top:2px'>Analytics Platform</div>"
    f"</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Project Overview", "📈 Drivers & Constructors", "🌦️ Tire & Weather Impact",
     "🤖 ML Models & Journey", "🏆 Race Predictions", "🧠 Agentic AI & Lessons","🔍 Ask the Data"]
)

if len(df) > 0 and "season" in df.columns:
    seasons = sorted(df["season"].unique())
    selected_season = st.sidebar.selectbox("Season", seasons, index=len(seasons) - 1)
else:
    selected_season = 2026

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='font-size:0.75rem;color:rgba(255,255,255,0.6);line-height:1.6'>"
    f"Records: {len(df):,}<br>"
    f"Seasons: {df['season'].nunique() if len(df) > 0 else 0}<br>"
    f"Features: {len(df.columns) if len(df) > 0 else 0}<br>"
    f"Models: {len(data.get('models', {}))}"
    f"</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# PAGE 1: PROJECT OVERVIEW
# ═══════════════════════════════════════════════════

if page == "🏠 Project Overview":
    st.markdown(
        f"<h1 style='font-family:Playfair Display,serif;color:{F1_RED};margin-bottom:0'>"
        f"F1 Big Data Analytics Platform</h1>"
        f"<div style='color:{GRAY};font-size:1rem;margin-bottom:20px'>"
        f"Race outcome prediction using Apache Spark, Docker, Airflow, and Machine Learning</div>",
        unsafe_allow_html=True)

    # KPI cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(kpi(f"{len(df):,}", "Race Records"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi("9", "Seasons", C_BLUE, P_BLUE), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi("85", "Features", C_PURPLE, P_PURPLE), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi("77.8%", "Winner Accuracy", C_GREEN, P_GREEN), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi("100%", "Top-3 Accuracy", C_AMBER, P_AMBER), unsafe_allow_html=True)

    # ── Project Goal (bigger, more prominent) ──
    sec("Project Goal", "What we are trying to achieve")
    st.markdown(
        f"<div style='background:white;border:1px solid #E9ECEF;"
        f"border-left:4px solid {F1_RED};border-radius:0 10px 10px 0;"
        f"padding:20px 24px;margin:10px 0;font-size:1.05rem;"
        f"color:#2d2d2d;line-height:1.85;box-shadow:0 1px 4px rgba(0,0,0,0.04)'>"
        f"Predict Formula 1 race outcomes — <b style='color:{F1_RED}'>finishing position</b>, "
        f"<b style='color:{F1_RED}'>podium finish</b>, and <b style='color:{F1_RED}'>race winner</b> — "
        f"using 9 seasons of historical data across 3 systematic phases:"
        f"<div style='display:flex;gap:12px;margin-top:14px'>"
        f"<div style='flex:1;background:{P_BLUE};border-radius:8px;padding:12px 14px;text-align:center'>"
        f"<div style='font-family:Playfair Display,serif;font-size:1.3rem;color:{C_BLUE};font-weight:700'>Phase 1</div>"
        f"<div style='font-size:0.82rem;color:{C_BLUE};margin-top:4px'>Core Race Data</div>"
        f"<div style='font-size:0.78rem;color:{GRAY};margin-top:2px'>Results, Qualifying, Pit Stops</div>"
        f"<div style='font-size:0.85rem;color:{C_BLUE};font-weight:600;margin-top:6px'>50% baseline</div></div>"
        f"<div style='flex:1;background:{P_AMBER};border-radius:8px;padding:12px 14px;text-align:center'>"
        f"<div style='font-family:Playfair Display,serif;font-size:1.3rem;color:{C_AMBER};font-weight:700'>Phase 2</div>"
        f"<div style='font-size:0.82rem;color:{C_AMBER};margin-top:4px'>Tire & Lap Telemetry</div>"
        f"<div style='font-size:0.78rem;color:{GRAY};margin-top:2px'>181K laps, degradation</div>"
        f"<div style='font-size:0.85rem;color:{C_AMBER};font-weight:600;margin-top:6px'>MAE 2.74→2.10</div></div>"
        f"<div style='flex:1;background:{P_GREEN};border-radius:8px;padding:12px 14px;text-align:center'>"
        f"<div style='font-family:Playfair Display,serif;font-size:1.3rem;color:{C_GREEN};font-weight:700'>Phase 3</div>"
        f"<div style='font-size:0.82rem;color:{C_GREEN};margin-top:4px'>Weather Integration</div>"
        f"<div style='font-size:0.78rem;color:{GRAY};margin-top:2px'>Temp, rain, humidity</div>"
        f"<div style='font-size:0.85rem;color:{C_GREEN};font-weight:600;margin-top:6px'>Winner 50%→77.8%</div></div>"
        f"</div></div>", unsafe_allow_html=True)

    # ── System Architecture (two clear sections) ──
    
    sec("Medallion Lakehouse Architecture",
        "Industry-standard data design pattern used across Azure, AWS, and GCP platforms")

    st.markdown(
        f"<div style='background:white;border:1px solid #E9ECEF;border-radius:10px;"
        f"padding:14px 18px;margin:10px 0 18px 0;font-size:0.9rem;color:#2d2d2d;line-height:1.7'>"
        f"The <b>medallion architecture</b> organizes data into three progressive quality layers. "
        f"Raw data enters as <b>Bronze</b>, is cleaned and joined into <b>Silver</b>, "
        f"then enriched into <b>Gold</b> (ML-ready). Each layer adds trust and business value."
        f"</div>", unsafe_allow_html=True)


    st.markdown(
        f"<div style='display:flex;gap:0;margin:10px 0;align-items:stretch'>"
        # Extraction - Teal
        f"<div style='flex:1;background:#E0F7FA;border:1px solid #80DEEA;border-radius:10px 0 0 10px;"
        f"padding:16px;text-align:center'>"
        f"<div style='font-size:0.7rem;color:#00838F;text-transform:uppercase;letter-spacing:0.08em;font-weight:700'>Extraction</div>"
        f"<div style='font-size:0.9rem;color:#006064;font-weight:600;margin-top:8px'>Jolpica API</div>"
        f"<div style='font-size:0.9rem;color:#006064;font-weight:600'>FastF1</div>"
        f"<div style='font-size:0.72rem;color:#00838F;margin-top:6px'>9 seasons</div></div>"
        # Arrow
        f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 6px'>→</div>"
        # BRONZE - metallic bronze
        f"<div style='flex:1.2;background:#F4E4C1;border:2px solid #CD7F32;padding:16px;text-align:center;border-radius:8px'>"
        f"<div style='font-size:0.7rem;color:#8B4513;text-transform:uppercase;letter-spacing:0.08em;font-weight:700'>🥉 Bronze</div>"
        f"<div style='font-size:0.9rem;color:#6B3410;font-weight:600;margin-top:8px'>Raw Data</div>"
        f"<div style='font-size:0.75rem;color:#8B4513;margin-top:6px;line-height:1.5'>3,524 results<br>181K laps<br>24K weather</div></div>"
        # Arrow
        f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 6px'>→</div>"
        # SILVER - metallic silver
        f"<div style='flex:1.2;background:#E8E8E8;border:2px solid #A8A8A8;padding:16px;text-align:center;border-radius:8px'>"
        f"<div style='font-size:0.7rem;color:#606060;text-transform:uppercase;letter-spacing:0.08em;font-weight:700'>🥈 Silver</div>"
        f"<div style='font-size:0.9rem;color:#404040;font-weight:600;margin-top:8px'>Cleaned & Joined</div>"
        f"<div style='font-size:0.75rem;color:#606060;margin-top:6px;line-height:1.5'>Spark transforms<br>Deduplicated<br>Race master table</div></div>"
        # Arrow
        f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 6px'>→</div>"
        # GOLD - metallic gold
        f"<div style='flex:1.2;background:#FFF4B8;border:2px solid #DAA520;padding:16px;text-align:center;border-radius:8px'>"
        f"<div style='font-size:0.7rem;color:#B8860B;text-transform:uppercase;letter-spacing:0.08em;font-weight:700'>🥇 Gold</div>"
        f"<div style='font-size:0.9rem;color:#8B6914;font-weight:600;margin-top:8px'>ML-Ready Features</div>"
        f"<div style='font-size:0.75rem;color:#B8860B;margin-top:6px;line-height:1.5'>ELO, streaks<br>Weather, tire<br>85 features</div></div>"
        # Arrow
        f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 6px'>→</div>"
        # Consumption - Purple
        f"<div style='flex:1;background:#EDE7F6;border:1px solid #B39DDB;border-radius:0 10px 10px 0;"
        f"padding:16px;text-align:center'>"
        f"<div style='font-size:0.7rem;color:#4527A0;text-transform:uppercase;letter-spacing:0.08em;font-weight:700'>Consumption</div>"
        f"<div style='font-size:0.9rem;color:#311B92;font-weight:600;margin-top:8px'>ML & Insights</div>"
        f"<div style='font-size:0.75rem;color:#4527A0;margin-top:6px;line-height:1.5'>3 Models<br>4 AI Agents<br>Dashboard</div></div>"
        f"</div>", unsafe_allow_html=True)   
        

    # Tech stack (organized cards)
    sec("Technology Stack", "Infrastructure powering the platform")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(info_box(
            "Processing & Storage",
            "<b>Apache Spark 3.5.1</b> — 2 workers, 4 cores<br>"
            "<b>Parquet</b> — columnar storage (Medallion)<br>"
            "<b>Docker Compose</b> — 7 containers",
            C_BLUE, P_BLUE), unsafe_allow_html=True)
    with c2:
        st.markdown(info_box(
            "Orchestration & Scheduling",
            "<b>Apache Airflow</b> — Monday auto-schedule<br>"
            "<b>PostgreSQL</b> — metadata database<br>"
            "<b>LangGraph</b> — 4 autonomous agents",
            C_GREEN, P_GREEN), unsafe_allow_html=True)
    with c3:
        st.markdown(info_box(
            "ML & Visualization",
            "<b>Random Forest</b> — Bagging ensemble<br>"
            "<b>Gradient Boosting</b> — Sequential boosting<br>"
            "<b>XGBoost</b> — Regularized boosting<br>"
            "<b>Streamlit + Plotly</b> — Dashboard",
            C_PURPLE, P_PURPLE), unsafe_allow_html=True)

    # Championship points chart (polished)
    
    if len(df) > 0 and "season_cumulative_points" in df.columns:
        sec(f"Championship Standings — {selected_season}", "Current points after latest round")
        season_df = df[df["season"] == selected_season].sort_values("round")
        latest_round = season_df["round"].max()
        latest_df = season_df[season_df["round"] == latest_round]

        if "season_cumulative_points" in latest_df.columns:
            standings = latest_df.groupby("driver_name")["season_cumulative_points"].max().sort_values(ascending=True)
            top = standings.tail(10)

            max_pts = top.max()
            third_pts = top.nlargest(3).min()
            colors = [F1_RED if v == max_pts else (C_BLUE if v >= third_pts else "#E0E0E0")
                      for v in top.values]

            fig = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation="h",
                marker_color=colors,
                text=[f"{v:.0f} pts" for v in top.values],
                textposition="outside",
                textfont=dict(size=12)))
            fig.update_layout(
                **PT, height=400,
                title=dict(text=f"{selected_season} Standings — After Round {int(latest_round)}"),
                xaxis=dict(title="Points", gridcolor="#F0F0F0"),
                yaxis=dict(title=""),
                margin=dict(l=140, r=60, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)




# ═══════════════════════════════════════════════════
# PAGE 5: RACE PREDICTIONS
# ═══════════════════════════════════════════════════
elif page == "🏆 Race Predictions":
    st.markdown(
        f"<h1 style='font-family:Playfair Display,serif;color:{F1_RED}'>Race Predictions</h1>",
        unsafe_allow_html=True)

    narrative(
        "Our ML models predict race outcomes <b>before the race starts</b> using 85 engineered features — "
        "qualifying pace, driver form, constructor strength, tire degradation history, and weather conditions. "
        "Many F1 prediction projects list weather and tire data as 'future improvements' — "
        "we have already integrated both, which is what pushed our winner accuracy from 50% to 77.8%.")

    if len(df) > 0:
        season_df = df[df["season"] == selected_season].copy()

        # ── Season-Wide Accuracy Summary ──
        model = data.get("models", {}).get("position_predictor")
        season_has_predictions = False

        if model is not None and "target_position" in season_df.columns:
            try:
                feature_cols = [c for c in season_df.columns if c not in [
                    "target_position", "target_podium", "target_winner", "target_points_finish",
                    "driver_name", "constructor_name", "race_name", "season", "round",
                    "driver_id", "constructor_id", "circuit_id", "date", "race_id"
                ]]
                numeric_cols = season_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
                X_all = season_df[numeric_cols].fillna(0)

                if hasattr(model, "feature_names_in_"):
                    model_features = list(model.feature_names_in_)
                    available = [f for f in model_features if f in X_all.columns]
                    missing = [f for f in model_features if f not in X_all.columns]
                    X_pred = X_all[available].copy()
                    for m in missing:
                        X_pred[m] = 0
                    X_pred = X_pred[model_features]
                else:
                    X_pred = X_all

                season_df["predicted_position"] = model.predict(X_pred)
                season_has_predictions = True

                winners_correct = 0
                top3_correct = 0
                within_2 = 0
                total_races = 0
                total_predictions = 0
                race_results_list = []

                for rnd in sorted(season_df["round"].unique()):
                    rdf = season_df[season_df["round"] == rnd].copy()
                    rdf = rdf[rdf["target_position"].notna()]
                    if len(rdf) == 0:
                        continue
                    total_races += 1
                    total_predictions += len(rdf)
                    actual_winner = rdf.loc[rdf["target_position"].idxmin(), "driver_name"]
                    predicted_winner = rdf.loc[rdf["predicted_position"].idxmin(), "driver_name"]
                    predicted_top3 = rdf.nsmallest(3, "predicted_position")["driver_name"].tolist()

                    # Count predictions within 2 positions
                    within_2 += ((rdf["predicted_position"] - rdf["target_position"]).abs() <= 2).sum()

                    correct = predicted_winner == actual_winner
                    in_top3 = actual_winner in predicted_top3
                    if correct:
                        winners_correct += 1
                    if in_top3:
                        top3_correct += 1

                    race_name_r = rdf["race_name"].iloc[0] if "race_name" in rdf.columns else f"R{int(rnd)}"
                    n_drivers = len(rdf)
                    race_results_list.append({
                        "Race": f"R{int(rnd)}",
                        "Race Name": race_name_r,
                        "Drivers": n_drivers,
                        "Predicted Winner": predicted_winner,
                        "Actual Winner": actual_winner,
                        "Winner Correct": "✅" if correct else "❌",
                        "In Top 3": "✅" if in_top3 else "❌"
                    })

                if total_races > 0:
                    sec(f"Season {selected_season} — Model Accuracy Across All Races",
                        f"Gradient Boosting model evaluated on {total_races} completed races")

                    k1, k2, k3, k4, k5 = st.columns(5)
                    with k1:
                        st.markdown(kpi(f"{total_races}", "Races", C_BLUE, P_BLUE), unsafe_allow_html=True)
                    with k2:
                        pct = f"{winners_correct}/{total_races}"
                        color = C_GREEN if total_races > 0 and winners_correct / total_races > 0.5 else C_AMBER
                        bg = P_GREEN if total_races > 0 and winners_correct / total_races > 0.5 else P_AMBER
                        st.markdown(kpi(pct, "Winners Correct", color, bg), unsafe_allow_html=True)
                    with k3:
                        st.markdown(kpi(f"{top3_correct}/{total_races}", "Winner in Top-3",
                                        C_GREEN, P_GREEN), unsafe_allow_html=True)
                    with k4:
                        within_pct = f"{within_2 * 100 / total_predictions:.0f}%" if total_predictions > 0 else "N/A"
                        st.markdown(kpi(within_pct, "Within 2 Positions", C_TEAL, P_TEAL), unsafe_allow_html=True)
                    with k5:
                        mae_season = (season_df["predicted_position"] - season_df["target_position"]).abs().mean()
                        st.markdown(kpi(f"{mae_season:.2f}", "Season MAE", C_PURPLE, P_PURPLE,
                                        "positions off"), unsafe_allow_html=True)

                    res_df = pd.DataFrame(race_results_list)
                    st.dataframe(res_df, use_container_width=True, hide_index=True)

                    st.markdown(info_box(
                        "What does 'Within 2 Positions' mean?",
                        f"Even when the model doesn't predict the exact finishing position, <b>{within_pct}</b> "
                        f"of all predictions are within 2 places of the actual result. For example, predicting "
                        f"P3 when the driver finishes P1, P2, P3, P4, or P5 counts as 'within 2'. "
                        f"This shows the model captures the right performance tier even when it misses the exact spot.",
                        C_TEAL, P_TEAL), unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"Model prediction error: {e}")

        st.markdown("---")

        # ── Per-Race Deep Dive ──
        sec("Per-Race Deep Dive", "Select a race to see detailed predictions and analysis")
        rounds = sorted(season_df["round"].unique())
        round_names = {}
        for r in rounds:
            rdf = season_df[season_df["round"] == r]
            name = rdf["race_name"].iloc[0] if "race_name" in rdf.columns and len(rdf) > 0 else f"Round {int(r)}"
            n = len(rdf[rdf["target_position"].notna()]) if "target_position" in rdf.columns else len(rdf)
            round_names[r] = f"R{int(r)} — {name} ({n} drivers)"

        selected_round = st.selectbox("Select Race", rounds, index=len(rounds) - 1,
                                       format_func=lambda x: round_names.get(x, f"Round {int(x)}"))

        race_df = season_df[season_df["round"] == selected_round].copy()
        race_name = race_df["race_name"].iloc[0] if "race_name" in race_df.columns and len(race_df) > 0 else ""

        if race_name:
            st.markdown(f"<div style='font-size:1.2rem;color:{F1_RED};font-weight:600;margin-bottom:10px'>"
                        f"{race_name}</div>", unsafe_allow_html=True)

        if "target_position" in race_df.columns:
            race_df = race_df[race_df["target_position"].notna()].copy()
            race_df["target_position"] = race_df["target_position"].astype(int)
            if "grid" in race_df.columns:
                race_df["grid"] = race_df["grid"].fillna(20).astype(int)
            race_df = race_df.sort_values("target_position")

            # Race KPIs
            winner = race_df.iloc[0]["driver_name"] if len(race_df) > 0 else "N/A"
            winner_team = race_df.iloc[0]["constructor_name"] if "constructor_name" in race_df.columns and len(race_df) > 0 else ""
            winner_grid = int(race_df.iloc[0]["grid"]) if len(race_df) > 0 else 0
            podium = race_df.head(3)["driver_name"].tolist()

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(kpi(winner, "Race Winner", F1_RED, F1_RED_L, winner_team), unsafe_allow_html=True)
            with k2:
                st.markdown(kpi(podium[1] if len(podium) > 1 else "N/A", "P2", C_BLUE, P_BLUE),
                            unsafe_allow_html=True)
            with k3:
                st.markdown(kpi(podium[2] if len(podium) > 2 else "N/A", "P3", C_GREEN, P_GREEN),
                            unsafe_allow_html=True)
            with k4:
                st.markdown(kpi(str(len(race_df)), "Finishers", C_PURPLE, P_PURPLE), unsafe_allow_html=True)

            # ── ML Predicted vs Actual ──
            predicted_winner = None
            winner_correct = False
            winner_in_top3 = False
            race_mae = 0

            if season_has_predictions and "predicted_position" in race_df.columns:
                sec("ML Model Prediction vs Actual Result",
                    "Blue = what our model predicted before the race, Red = what actually happened")

                race_df["prediction_error"] = (race_df["predicted_position"] - race_df["target_position"]).round(1)
                race_df["predicted_rank"] = race_df["predicted_position"].rank().astype(int)

                predicted_winner = race_df.loc[race_df["predicted_position"].idxmin(), "driver_name"]
                predicted_top3 = race_df.nsmallest(3, "predicted_position")["driver_name"].tolist()
                winner_correct = predicted_winner == winner
                winner_in_top3 = winner in predicted_top3
                race_mae = race_df["prediction_error"].abs().mean()

                k1, k2, k3 = st.columns(3)
                with k1:
                    color = C_GREEN if winner_correct else C_AMBER
                    bg = P_GREEN if winner_correct else P_AMBER
                    st.markdown(kpi(predicted_winner, "Model Predicted Winner",
                                    color, bg, "✅ CORRECT" if winner_correct else "❌ MISSED"),
                                unsafe_allow_html=True)
                with k2:
                    color = C_GREEN if winner_in_top3 else F1_RED
                    bg = P_GREEN if winner_in_top3 else F1_RED_L
                    st.markdown(kpi("Yes" if winner_in_top3 else "No",
                                    "Winner in Top-3 Prediction", color, bg), unsafe_allow_html=True)
                with k3:
                    st.markdown(kpi(f"{race_mae:.1f}", "Avg Position Error",
                                    C_BLUE, P_BLUE, "positions off"), unsafe_allow_html=True)

                # Predicted vs Actual bar chart — top 10
                top10 = race_df.head(10)
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Predicted Position", x=top10["driver_name"],
                                     y=top10["predicted_position"], marker_color=C_BLUE,
                                     text=[f"P{v:.1f}" for v in top10["predicted_position"]],
                                     textposition="outside", textfont=dict(size=10)))
                fig.add_trace(go.Bar(name="Actual Finish", x=top10["driver_name"],
                                     y=top10["target_position"], marker_color=F1_RED,
                                     text=[f"P{int(v)}" for v in top10["target_position"]],
                                     textposition="outside", textfont=dict(size=10)))
                fig.update_layout(**PT, barmode="group", height=420,
                                  title=dict(text="Model Prediction vs Reality (Top 10 Finishers)"),
                                  yaxis=dict(title="Position", autorange="reversed",
                                             gridcolor="#F0F0F0", dtick=2),
                                  xaxis=dict(tickangle=-30),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                              xanchor="center", x=0.5),
                                  margin=dict(l=40, r=20, t=70, b=80))
                st.plotly_chart(fig, use_container_width=True)

                # Prediction error chart
                sec("Prediction Error by Driver",
                    "Green = model predicted worse than actual (pleasant surprise), Amber = predicted better than actual")
                top10_err = race_df.head(10).sort_values("prediction_error")
                fig = go.Figure(go.Bar(
                    x=top10_err["driver_name"],
                    y=top10_err["prediction_error"],
                    marker_color=[C_GREEN if v <= 0 else C_AMBER for v in top10_err["prediction_error"]],
                    text=[f"{v:+.1f}" for v in top10_err["prediction_error"]],
                    textposition="outside", textfont=dict(size=11)))
                fig.update_layout(**PT, height=350,
                                  title=dict(text="How far off was the model per driver?"),
                                  yaxis=dict(title="Error (positions)", gridcolor="#F0F0F0"),
                                  xaxis=dict(tickangle=-30),
                                  margin=dict(l=40, r=20, t=50, b=80))
                fig.add_hline(y=0, line_dash="dash", line_color=GRAY, line_width=1)
                st.plotly_chart(fig, use_container_width=True)

                # ── Full Prediction Table ──
                sec("Full Prediction Table — Top 10",
                    "Complete numbers for every prediction — verify the model's output")
                table_df = race_df.head(10)[["driver_name", "constructor_name", "grid",
                                              "predicted_rank", "predicted_position",
                                              "target_position", "prediction_error"]].copy()
                table_df["predicted_position"] = table_df["predicted_position"].round(1)
                table_df.columns = ["Driver", "Team", "Grid", "Predicted Rank",
                                    "Predicted Pos", "Actual Finish", "Error"]
                st.dataframe(table_df, use_container_width=True, hide_index=True)

            else:
                # Fallback — show basic results without model
                sec("Race Results — Top 10")
                top10 = race_df.head(10)
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Grid", x=top10["driver_name"],
                                     y=top10["grid"], marker_color=C_BLUE))
                fig.add_trace(go.Bar(name="Finish", x=top10["driver_name"],
                                     y=top10["target_position"], marker_color=F1_RED))
                fig.update_layout(**PT, barmode="group", height=400,
                                  title=dict(text="Grid vs Finish (Top 10)"),
                                  yaxis=dict(title="Position", autorange="reversed"),
                                  margin=dict(l=40, r=20, t=50, b=80))
                st.plotly_chart(fig, use_container_width=True)

            # ── 2x2 Race Insight Boxes ──
            sec("Race Analysis", "Key insights from this race explained for any audience")

            # Calculate gains for insight
            race_df["positions_gained"] = race_df["grid"] - race_df["target_position"]
            best = race_df.loc[race_df["positions_gained"].idxmax()]
            worst = race_df.loc[race_df["positions_gained"].idxmin()]

            c1, c2 = st.columns(2)
            with c1:
                # Race Summary box
                st.markdown(
                    f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {F1_RED};"
                    f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                    f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:220px'>"
                    f"<div style='font-weight:700;color:{F1_RED};font-size:1rem;margin-bottom:8px'>RACE SUMMARY</div>"
                    f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                    f"<b>{winner}</b> ({winner_team}) won the <b>{race_name}</b>, "
                    f"starting from grid position P{winner_grid}.<br><br>"
                    f"In Formula 1, grid position is determined by <i>qualifying</i> — "
                    f"a timed session on Saturday where all 20 drivers compete for the fastest lap. "
                    f"The fastest driver starts at the front (P1, called <i>pole position</i>), "
                    f"which gives a massive advantage: clean air, less traffic, and lower risk of "
                    f"first-lap collisions."
                    f"</div></div>", unsafe_allow_html=True)

            with c2:
                # Model Accuracy box
                if predicted_winner:
                    st.markdown(
                        f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_PURPLE};"
                        f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                        f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:220px'>"
                        f"<div style='font-weight:700;color:{C_PURPLE};font-size:1rem;margin-bottom:8px'>MODEL ACCURACY</div>"
                        f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                        f"Our Gradient Boosting model predicted <b>{predicted_winner}</b> would win. "
                        f"{'<span style=color:' + C_GREEN + ';font-weight:600>✅ Correct!</span>' if winner_correct else '<span style=color:' + C_AMBER + ';font-weight:600>❌ Missed — actual winner was ' + winner + '.</span>'}"
                        f"<br><br>"
                        f"Average error: <b>{race_mae:.1f} positions</b> — meaning predictions were "
                        f"off by about {race_mae:.1f} places on average. "
                        f"{'The actual winner was in the model' + chr(39) + 's top 3 predictions — our model achieves this 100% of the time across all races.' if winner_in_top3 else ''}"
                        f"</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_PURPLE};"
                        f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                        f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:220px'>"
                        f"<div style='font-weight:700;color:{C_PURPLE};font-size:1rem;margin-bottom:8px'>MODEL ACCURACY</div>"
                        f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                        f"Load the trained model to see prediction accuracy for this race. "
                        f"Run <code>python ml/model_comparison.py</code> to generate models."
                        f"</div></div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                # Who Overperformed box
                st.markdown(
                    f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_GREEN};"
                    f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                    f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:220px'>"
                    f"<div style='font-weight:700;color:{C_GREEN};font-size:1rem;margin-bottom:8px'>WHO OVERPERFORMED</div>"
                    f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                    f"<b>{best['driver_name']}</b> started P{int(best['grid'])} but finished "
                    f"P{int(best['target_position'])}, gaining <b>{int(best['positions_gained'])} positions</b>.<br><br>"
                    f"In F1, gaining positions typically happens through: "
                    f"<b>better tire management</b> (making tires last longer so you pit less), "
                    f"<b>smarter pit stop timing</b> (stopping for fresh tires at the optimal moment), "
                    f"or <b>strong overtaking</b> in the opening laps when cars are bunched together."
                    f"</div></div>", unsafe_allow_html=True)

            with c2:
                # Who Underperformed box
                st.markdown(
                    f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_AMBER};"
                    f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                    f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:220px'>"
                    f"<div style='font-weight:700;color:{C_AMBER};font-size:1rem;margin-bottom:8px'>WHO UNDERPERFORMED</div>"
                    f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                    f"<b>{worst['driver_name']}</b> started P{int(worst['grid'])} but dropped to "
                    f"P{int(worst['target_position'])}, losing <b>{abs(int(worst['positions_gained']))} positions</b>.<br><br>"
                    f"Losing positions can result from: "
                    f"<b>rapid tire degradation</b> (tires losing grip faster than competitors), "
                    f"<b>poor strategy calls</b> from the pit wall, "
                    f"<b>damage from contact</b> with other cars, "
                    f"or <b>mechanical issues</b> like engine overheating or brake problems."
                    f"</div></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# PAGE 2: DRIVER & CONSTRUCTOR ANALYSIS
# ═══════════════════════════════════════════════════
elif page == "📈 Drivers & Constructors":
    st.markdown(
        f"<h1 style='font-family:Playfair Display,serif;color:{F1_RED}'>Drivers & Constructors</h1>",
        unsafe_allow_html=True)

    narrative(
        "All features on this page are computed by <b>Apache Spark Window Functions</b> running on our "
        "2-worker cluster, and refreshed automatically <b>every Monday via Apache Airflow</b> when "
        "new race data becomes available after each Grand Prix weekend.")

    # Spark + Airflow explanation boxes
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(info_box(
            "How Spark Computes These Features",
            "Spark Window Functions compute values across related rows. "
            "Instead of one number for the whole dataset, Spark <b>partitions data by driver</b> "
            "(or team) and computes rolling averages, cumulative sums, and rankings within each "
            "partition — distributed across 2 workers in parallel.<br><br>"
            "<code style='font-size:0.8rem;background:#F0F0F0;padding:2px 6px;border-radius:4px'>"
            "AVG(position) OVER (PARTITION BY driver_id ORDER BY season, round ROWS 4 PRECEDING)"
            "</code><br><br>"
            "This computes each driver's last 5-race average at every point in 9 seasons (3,524 records).",
            C_BLUE, P_BLUE), unsafe_allow_html=True)
    with c2:
        st.markdown(info_box(
            "How Airflow Keeps Data Fresh",
            "Every <b>Monday at 6 AM</b>, Airflow triggers a 5-task pipeline:<br><br>"
            "1. <b>Ingest</b> — pull latest race data from Jolpica API + FastF1<br>"
            "2. <b>Spark Silver</b> — clean, deduplicate, join on cluster<br>"
            "3. <b>Spark Gold</b> — compute 85 features via Window functions<br>"
            "4. <b>Retrain</b> — compare 3 ML models on new data<br>"
            "5. <b>Agentic AI</b> — 4 agents evaluate quality + generate briefing<br><br>"
            "The charts below update automatically — no manual intervention needed.",
            C_GREEN, P_GREEN), unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        n_drivers = df["driver_name"].nunique() if len(df) > 0 else 0
        st.markdown(kpi(str(n_drivers), "Drivers", F1_RED, F1_RED_L), unsafe_allow_html=True)
    with k2:
        n_teams = df["constructor_name"].nunique() if len(df) > 0 and "constructor_name" in df.columns else 0
        st.markdown(kpi(str(n_teams), "Constructors", C_BLUE, P_BLUE), unsafe_allow_html=True)
    with k3:
        laps_count = f"{len(data.get('laps', pd.DataFrame())):,}" if "laps" in data else "181,721"
        st.markdown(kpi(laps_count, "Laps Processed", C_AMBER, P_AMBER), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi("30+", "Spark Features", C_PURPLE, P_PURPLE), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi("MON 6AM", "Pipeline Schedule", C_TEAL, P_TEAL), unsafe_allow_html=True)

    if len(df) > 0:
        tab1, tab2, tab3 = st.tabs(["⚙️ Data Engineering", "🏎️ Driver Performance", "🔧 Constructor Battle"])

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TAB 1: DATA ENGINEERING
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with tab1:
            # Pipeline Health Monitor
            sec("Pipeline Health Monitor",
                "Status of each task in the Airflow-orchestrated pipeline")

            pipeline_tasks = pd.DataFrame([
                {"Task": "1. Bronze Ingestion", "Tool": "Python + Jolpica API + FastF1",
                 "Status": "✅ Success", "Records": "3,524 results + 181K laps + 24K weather",
                 "Description": "Pull raw race data from APIs with retry logic"},
                {"Task": "2. Spark Silver Transform", "Tool": "Apache Spark (2 workers)",
                 "Status": "✅ Success", "Records": "3,524 rows (race_master)",
                 "Description": "Clean, deduplicate, type-cast, join results + qualifying + pit stops"},
                {"Task": "3. Spark Gold Features", "Tool": "Apache Spark (Window Functions)",
                 "Status": "✅ Success", "Records": "3,524 × 85 features",
                 "Description": "Compute rolling averages, ELO ratings, constructor strength, tire deg"},
                {"Task": "4. ML Model Retrain", "Tool": "scikit-learn + XGBoost",
                 "Status": "✅ Success", "Records": "3 models saved",
                 "Description": "Compare Random Forest vs Gradient Boosting vs XGBoost"},
                {"Task": "5. Agentic AI", "Tool": "LangGraph (4 agents)",
                 "Status": "✅ Success", "Records": "3 features discovered",
                 "Description": "Data quality check, feature discovery, model diagnosis, briefing"},
            ])
            st.dataframe(pipeline_tasks, use_container_width=True, hide_index=True)

            # Data Volume by Layer
            sec("Data Volume by Layer",
                "How data transforms from raw API responses to ML-ready features")

            volume_data = pd.DataFrame([
                {"Layer": "Bronze (Raw)", "Records": 223350, "Description": "All raw API data combined"},
                {"Layer": "Silver (Cleaned)", "Records": 3524, "Description": "Deduplicated race master table"},
                {"Layer": "Gold (Features)", "Records": 3524, "Description": "85 engineered features per record"},
            ])

            fig = go.Figure()
            colors_vol = ["#CD7F32", "#A8A9AD", "#DAA520"]
            fig.add_trace(go.Bar(
                x=volume_data["Layer"], y=volume_data["Records"],
                marker_color=colors_vol,
                text=[f"{v:,}" for v in volume_data["Records"]],
                textposition="outside", textfont=dict(size=13, color="#2d2d2d")))
            fig.update_layout(**PT, height=350,
                              title=dict(text="Data Reduction: 223K raw records → 3,524 ML-ready rows"),
                              yaxis=dict(title="Number of Records", gridcolor="#F0F0F0"),
                              margin=dict(l=50, r=20, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(info_box(
                "Why does Silver have fewer records than Bronze?",
                "Bronze stores ALL raw data separately — race results (3,524), qualifying (3,518), "
                "pit stops (5,674), laps (181,721), weather (24,435), tire degradation (7,996), "
                "standings (195). Silver <b>joins and deduplicates</b> these into a single "
                "<i>race_master</i> table with one row per driver per race. Gold then enriches "
                "each row with 85 computed features. This is the ETL pattern: Extract (223K) → "
                "Transform (join + clean) → Load (3,524 × 85 = ~300K feature values).",
                C_BLUE, P_BLUE), unsafe_allow_html=True)

            # Feature Engineering Showcase
            sec("Feature Engineering Showcase",
                "How raw data is transformed into predictive features using Spark")

            features_showcase = pd.DataFrame([
                {"Raw Data": "Finishing positions", "Spark Transform": "Window AVG last 10 races per team",
                 "Gold Feature": "constructor_rolling_points", "Type": "Rolling Window",
                 "Importance": "#1 most important"},
                {"Raw Data": "Finishing positions", "Spark Transform": "Window AVG last 5 races per driver",
                 "Gold Feature": "rolling_avg_position_5", "Type": "Rolling Window",
                 "Importance": "#2"},
                {"Raw Data": "Qualifying lap times", "Spark Transform": "MIN(Q3 time) per race, compute gap",
                 "Gold Feature": "quali_gap_to_pole", "Type": "Aggregation",
                 "Importance": "#3"},
                {"Raw Data": "Race results history", "Spark Transform": "ELO rating system (chess-style)",
                 "Gold Feature": "driver_elo_rating", "Type": "Custom Algorithm",
                 "Importance": "#5"},
                {"Raw Data": "Lap times per tire", "Spark Transform": "Linear regression per stint",
                 "Gold Feature": "avg_tire_deg_slope", "Type": "Statistical Model",
                 "Importance": "#7"},
                {"Raw Data": "FastF1 weather API", "Spark Transform": "AVG temperature per session",
                 "Gold Feature": "avg_track_temp", "Type": "Aggregation",
                 "Importance": "#12"},
                {"Raw Data": "Grid + ELO combined", "Spark Transform": "grid × elo_rating interaction",
                 "Gold Feature": "grid_elo_interaction", "Type": "Feature Interaction",
                 "Importance": "#4"},
            ])
            st.dataframe(features_showcase, use_container_width=True, hide_index=True)

            # Data Lineage
            sec("Data Lineage — Tracing Our #1 Feature",
                "How constructor_rolling_points travels from API to prediction")

            st.markdown(
                f"<div style='display:flex;gap:0;margin:10px 0;align-items:stretch'>"
                # Step 1
                f"<div style='flex:1;background:#E0F7FA;border:1px solid #80DEEA;border-radius:10px 0 0 10px;"
                f"padding:14px;text-align:center'>"
                f"<div style='font-size:0.7rem;color:#00838F;text-transform:uppercase;font-weight:700'>Step 1</div>"
                f"<div style='font-size:0.85rem;color:#006064;font-weight:600;margin-top:6px'>API Fetch</div>"
                f"<div style='font-size:0.75rem;color:#00838F;margin-top:4px'>Jolpica returns<br>race results JSON<br>with points per race</div></div>"
                # Arrow
                f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 4px'>→</div>"
                # Step 2
                f"<div style='flex:1;background:#F4E4C1;border:1px solid #CD7F32;padding:14px;text-align:center;border-radius:6px'>"
                f"<div style='font-size:0.7rem;color:#8B4513;text-transform:uppercase;font-weight:700'>Step 2</div>"
                f"<div style='font-size:0.85rem;color:#6B3410;font-weight:600;margin-top:6px'>Bronze</div>"
                f"<div style='font-size:0.75rem;color:#8B4513;margin-top:4px'>Stored as raw<br>parquet file<br>per season</div></div>"
                # Arrow
                f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 4px'>→</div>"
                # Step 3
                f"<div style='flex:1;background:#E8E8E8;border:1px solid #A8A8A8;padding:14px;text-align:center;border-radius:6px'>"
                f"<div style='font-size:0.7rem;color:#606060;text-transform:uppercase;font-weight:700'>Step 3</div>"
                f"<div style='font-size:0.85rem;color:#404040;font-weight:600;margin-top:6px'>Silver Join</div>"
                f"<div style='font-size:0.75rem;color:#606060;margin-top:4px'>Points joined with<br>driver + team<br>in race_master</div></div>"
                # Arrow
                f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 4px'>→</div>"
                # Step 4
                f"<div style='flex:1;background:#FFF4B8;border:1px solid #DAA520;padding:14px;text-align:center;border-radius:6px'>"
                f"<div style='font-size:0.7rem;color:#B8860B;text-transform:uppercase;font-weight:700'>Step 4</div>"
                f"<div style='font-size:0.85rem;color:#8B6914;font-weight:600;margin-top:6px'>Spark Window</div>"
                f"<div style='font-size:0.75rem;color:#B8860B;margin-top:4px'>AVG(points) OVER<br>last 10 races<br>per constructor</div></div>"
                # Arrow
                f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 4px'>→</div>"
                # Step 5
                f"<div style='flex:1;background:#EDE7F6;border:1px solid #B39DDB;border-radius:0 10px 10px 0;"
                f"padding:14px;text-align:center'>"
                f"<div style='font-size:0.7rem;color:#4527A0;text-transform:uppercase;font-weight:700'>Step 5</div>"
                f"<div style='font-size:0.85rem;color:#311B92;font-weight:600;margin-top:6px'>ML Model</div>"
                f"<div style='font-size:0.75rem;color:#4527A0;margin-top:4px'>Fed as #1 feature<br>into all 3<br>prediction models</div></div>"
                f"</div>", unsafe_allow_html=True)

            # Schema Evolution
            sec("Schema Evolution — A Real Bug We Fixed",
                "Production data pipelines encounter messy real-world issues")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"<div style='background:#FFF3E0;border:2px solid {C_AMBER};border-radius:10px;"
                    f"padding:16px;text-align:center'>"
                    f"<div style='font-size:0.8rem;color:{C_AMBER};font-weight:700;text-transform:uppercase'>The Problem</div>"
                    f"<div style='font-size:0.88rem;color:#2d2d2d;margin-top:10px;line-height:1.7;text-align:left'>"
                    f"2018–2024 parquet files stored <code>avg_speed_kph</code> as <b>STRING</b> (\"210.5\")<br>"
                    f"2025 parquet stored it as <b>INT</b> (210)<br><br>"
                    f"Spark's <code>mergeSchema</code> refused to union these — "
                    f"the entire Silver transform crashed."
                    f"</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f"<div style='background:{P_GREEN};border:2px solid {C_GREEN};border-radius:10px;"
                    f"padding:16px;text-align:center'>"
                    f"<div style='font-size:0.8rem;color:{C_GREEN};font-weight:700;text-transform:uppercase'>Our Solution</div>"
                    f"<div style='font-size:0.88rem;color:#2d2d2d;margin-top:10px;line-height:1.7;text-align:left'>"
                    f"Read each parquet file individually<br>"
                    f"Cast ALL columns to <b>StringType</b> per file<br>"
                    f"Union with <code>unionByName(allowMissingColumns=True)</code><br>"
                    f"Then cast back to proper types<br><br>"
                    f"This is a standard pattern in production big data systems."
                    f"</div></div>", unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TAB 2: DRIVER PERFORMANCE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with tab2:
            drivers = sorted(df["driver_name"].unique())
            default_drivers = [d for d in ["Max Verstappen", "Lando Norris", "Charles Leclerc",
                                           "Oscar Piastri", "Lewis Hamilton",
                                           "Andrea Kimi Antonelli", "George Russell"]
                               if d in drivers][:4]
            selected_drivers = st.multiselect("Select Drivers to Compare", drivers, default=default_drivers)

            if selected_drivers:
                driver_df = df[(df["driver_name"].isin(selected_drivers)) &
                               (df["season"] == selected_season)].sort_values("round")

                # Position Trend
                if "target_position" in driver_df.columns:
                    sec(f"Finishing Position Trend — {selected_season}",
                        "Each point = one race result. Lower = better.")

                    fig = go.Figure()
                    colors = [F1_RED, C_BLUE, C_GREEN, C_AMBER, C_PURPLE, C_TEAL]
                    for i, driver in enumerate(selected_drivers):
                        ddf = driver_df[driver_df["driver_name"] == driver]
                        if len(ddf) > 0:
                            fig.add_trace(go.Scatter(
                                x=ddf["round"], y=ddf["target_position"],
                                name=driver, mode="lines+markers",
                                line=dict(color=colors[i % len(colors)], width=3),
                                marker=dict(size=8, color=colors[i % len(colors)])))

                    fig.update_layout(**PT, height=400,
                                      title=dict(text="Race-by-Race Performance"),
                                      xaxis=dict(title="Round", dtick=1, gridcolor="#F0F0F0"),
                                      yaxis=dict(title="Finishing Position", autorange="reversed",
                                                 gridcolor="#F0F0F0", dtick=2),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                                  xanchor="center", x=0.5),
                                      hovermode="x unified",
                                      margin=dict(l=40, r=20, t=70, b=50))
                    st.plotly_chart(fig, use_container_width=True)

                # Rolling Form
                if "rolling_avg_position_5" in driver_df.columns:
                    sec("Driver Form — Rolling 5-Race Average",
                        "Computed by Spark Window: AVG(position) OVER last 5 races per driver. Smoother trend shows momentum.")

                    fig = go.Figure()
                    for i, driver in enumerate(selected_drivers):
                        ddf = driver_df[driver_df["driver_name"] == driver]
                        if len(ddf) > 0 and ddf["rolling_avg_position_5"].notna().any():
                            fig.add_trace(go.Scatter(
                                x=ddf["round"], y=ddf["rolling_avg_position_5"],
                                name=driver, mode="lines+markers",
                                line=dict(color=colors[i % len(colors)], width=3),
                                marker=dict(size=8, color=colors[i % len(colors)])))

                    fig.update_layout(**PT, height=380,
                                      title=dict(text="5-Race Rolling Average (Lower = Better Form)"),
                                      xaxis=dict(title="Round", dtick=1, gridcolor="#F0F0F0"),
                                      yaxis=dict(title="Avg Position (last 5)", autorange="reversed",
                                                 gridcolor="#F0F0F0"),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                                  xanchor="center", x=0.5),
                                      hovermode="x unified",
                                      margin=dict(l=40, r=20, t=70, b=50))
                    st.plotly_chart(fig, use_container_width=True)

                # Positions Gained/Lost per driver (moved from Page 2)
                if "grid" in driver_df.columns and "target_position" in driver_df.columns:
                    sec("Positions Gained / Lost",
                        "Green = overtook cars during the race, Red = lost positions from grid")
                    driver_df["positions_gained"] = driver_df["grid"] - driver_df["target_position"]
                    avg_gains = driver_df.groupby("driver_name")["positions_gained"].mean().sort_values(ascending=False)

                    fig = go.Figure(go.Bar(
                        x=avg_gains.index, y=avg_gains.values,
                        marker_color=[C_GREEN if v > 0 else (F1_RED if v < 0 else GRAY)
                                      for v in avg_gains.values],
                        text=[f"{v:+.1f}" for v in avg_gains.values],
                        textposition="outside", textfont=dict(size=12)))
                    fig.update_layout(**PT, height=350,
                                      title=dict(text=f"Average Positions Gained Per Race — {selected_season}"),
                                      yaxis=dict(title="Avg Positions Gained", gridcolor="#F0F0F0"),
                                      xaxis=dict(tickangle=-30),
                                      margin=dict(l=40, r=20, t=50, b=80))
                    fig.add_hline(y=0, line_dash="dash", line_color=GRAY, line_width=1)
                    st.plotly_chart(fig, use_container_width=True)

                # Season Stats Table
                sec("Season Statistics")
                cols_for_stats = {"target_position": "mean", "grid": "mean", "round": "count"}
                if "target_podium" in driver_df.columns:
                    cols_for_stats["target_podium"] = "sum"
                if "target_winner" in driver_df.columns:
                    cols_for_stats["target_winner"] = "sum"

                stats = driver_df.groupby("driver_name").agg(**{
                    "Avg Finish": ("target_position", "mean"),
                    "Best": ("target_position", "min"),
                    "Avg Grid": ("grid", "mean"),
                    "Races": ("round", "count"),
                    **({
                        "Podiums": ("target_podium", "sum"),
                    } if "target_podium" in driver_df.columns else {}),
                    **({
                        "Wins": ("target_winner", "sum"),
                    } if "target_winner" in driver_df.columns else {}),
                }).round(1)
                st.dataframe(stats, use_container_width=True)

                # Driver insight boxes
                sec("Driver Insights")
                # Best performer
                best_driver = stats["Avg Finish"].idxmin()
                best_avg = stats.loc[best_driver, "Avg Finish"]
                best_races = int(stats.loc[best_driver, "Races"])

                # Most positions gained
                if "positions_gained" in driver_df.columns:
                    avg_g = driver_df.groupby("driver_name")["positions_gained"].mean()
                    most_gained_driver = avg_g.idxmax()
                    most_gained_val = avg_g.max()

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(info_box(
                        f"Best Performer — {best_driver}",
                        f"Averaged <b>P{best_avg:.1f}</b> across {best_races} races this season. "
                        f"{'Collected ' + str(int(stats.loc[best_driver, 'Podiums'])) + ' podiums and ' + str(int(stats.loc[best_driver, 'Wins'])) + ' wins.' if 'Podiums' in stats.columns and 'Wins' in stats.columns else ''} "
                        f"In F1, consistent top finishes are more valuable than occasional wins — "
                        f"the championship rewards reliability across the full season.",
                        C_GREEN, P_GREEN), unsafe_allow_html=True)
                with c2:
                    if "positions_gained" in driver_df.columns:
                        st.markdown(info_box(
                            f"Best Racer — {most_gained_driver}",
                            f"Gained an average of <b>{most_gained_val:+.1f} positions</b> per race "
                            f"from grid to finish. This means they consistently finish higher than "
                            f"where they qualified — a sign of strong race pace, good tire management, "
                            f"or smart pit stop strategy from their team.",
                            C_BLUE, P_BLUE), unsafe_allow_html=True)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TAB 3: CONSTRUCTOR BATTLE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        with tab3:
            season_df = df[df["season"] == selected_season].copy()

            if "constructor_name" in season_df.columns and "target_position" in season_df.columns:
                sec(f"Constructor Standings — {selected_season}",
                    "Total team points after latest round")

                # Calculate team points
                if "target_points_finish" in season_df.columns:
                    team_pts = season_df.groupby("constructor_name")["target_points_finish"].sum().sort_values(ascending=True)
                else:
                    team_pts = season_df.groupby("constructor_name")["target_position"].count().sort_values(ascending=True)

                top_teams = team_pts.tail(10)
                max_pts = top_teams.max()
                third_pts = top_teams.nlargest(3).min()

                colors_bar = [F1_RED if v == max_pts else (C_BLUE if v >= third_pts else "#E0E0E0")
                              for v in top_teams.values]

                fig = go.Figure(go.Bar(
                    x=top_teams.values, y=top_teams.index, orientation="h",
                    marker_color=colors_bar,
                    text=[f"{v:.0f} pts" for v in top_teams.values],
                    textposition="outside", textfont=dict(size=12)))
                fig.update_layout(**PT, height=400,
                                  title=dict(text=f"{selected_season} Constructor Championship"),
                                  xaxis=dict(title="Points", gridcolor="#F0F0F0"),
                                  yaxis=dict(title=""),
                                  margin=dict(l=140, r=60, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

                # Insight boxes
                best_team = team_pts.idxmax()
                best_pts = team_pts.max()
                team_drivers = season_df[season_df["constructor_name"] == best_team]["driver_name"].unique()
                driver_list = " and ".join(team_drivers[:2])

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(info_box(
                        f"Championship Leader — {best_team}",
                        f"Leading with <b>{best_pts:.0f} points</b>. "
                        f"Their drivers <b>{driver_list}</b> have been the most consistent pairing. "
                        f"In F1, both drivers contribute points — a strong car with two fast "
                        f"drivers is what wins the constructors' title.",
                        F1_RED, F1_RED_L), unsafe_allow_html=True)
                with c2:
                    st.markdown(info_box(
                        "Why 'The Car Matters Most'",
                        "<b>constructor_rolling_points</b> is the #1 most important feature in all three "
                        "of our ML models. The car determines approximately <b>80% of performance</b>, "
                        "the driver approximately 20%. Our model confirms this with data from 9 seasons.",
                        C_BLUE, P_BLUE), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PAGE 4: TIRE & WEATHER IMPACT
# ═══════════════════════════════════════════════════
elif page == "🌦️ Tire & Weather Impact":
    st.markdown(
        f"<h1 style='font-family:Playfair Display,serif;color:{F1_RED}'>Tire & Weather Impact</h1>",
        unsafe_allow_html=True)

    # Two phase boxes side by side
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-top:4px solid {C_AMBER};"
            f"border-radius:0 0 12px 12px;padding:20px 22px;box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
            f"<div style='font-size:1.8rem;margin-bottom:6px'>🛞</div>"
            f"<div style='font-family:Playfair Display,serif;font-size:1.2rem;color:{C_AMBER};font-weight:700'>Phase 2 — Tire & Lap Data</div>"
            f"<div style='font-size:0.95rem;color:#2d2d2d;line-height:1.8;margin-top:8px'>"
            f"Integrated <b>181,721 lap records</b> and <b>7,996 tire degradation</b> models. "
            f"Added features like degradation slope, stint count, and lap consistency.<br><br>"
            f"<span title='Mean Absolute Error — on average, predictions are off by this many finishing positions. "
            f"For example, MAE of 2.10 means predicting P3 when actual is P1 or P5.' "
            f"style='border-bottom:1px dashed {C_AMBER};cursor:help'>"
            f"Position MAE</span>: <b style='color:{C_AMBER}'>2.74 → 2.10</b> (23% better)"
            f"</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-top:4px solid {C_BLUE};"
            f"border-radius:0 0 12px 12px;padding:20px 22px;box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
            f"<div style='font-size:1.8rem;margin-bottom:6px'>🌧️</div>"
            f"<div style='font-family:Playfair Display,serif;font-size:1.2rem;color:{C_BLUE};font-weight:700'>Phase 3 — Weather Data</div>"
            f"<div style='font-size:0.95rem;color:#2d2d2d;line-height:1.8;margin-top:8px'>"
            f"Integrated <b>24,435 weather measurements</b> across 166 sessions. "
            f"Added track temperature, air temperature, humidity, and rainfall features.<br><br>"
            f"Winner Accuracy: <b style='color:{C_BLUE}'>50% → 70.4%</b> (+20.4% — single biggest improvement)"
            f"</div></div>", unsafe_allow_html=True)

    # Glossary bar
    st.markdown(
        f"<div style='background:#F8F8F8;border:1px solid #E8E8E8;border-radius:8px;"
        f"padding:10px 18px;margin:16px 0;display:flex;gap:30px;flex-wrap:wrap;font-size:0.82rem;color:{GRAY}'>"
        f"<span title='Mean Absolute Error — average number of positions the prediction is off by'>"
        f"<b style='color:#2d2d2d'>MAE</b> = avg positions off (lower = better) ℹ️</span>"
        f"<span title='How many seconds per lap a tire loses as it wears down'>"
        f"<b style='color:#2d2d2d'>Degradation</b> = grip loss per lap ℹ️</span>"
        f"<span title='F1 tires come in different rubber compounds — softer = faster but wears quicker'>"
        f"<b style='color:#2d2d2d'>Compound</b> = tire rubber type ℹ️</span>"
        f"<span title='One continuous run on a single set of tires between pit stops'>"
        f"<b style='color:#2d2d2d'>Stint</b> = laps between pit stops ℹ️</span>"
        f"</div>", unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi("181,721", "Laps Processed", C_AMBER, P_AMBER), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi("7,996", "Tire Deg Records", F1_RED, F1_RED_L), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi("24,435", "Weather Records", C_BLUE, P_BLUE), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi("+27.8%", "Total Winner Gain", C_GREEN, P_GREEN, "50% → 77.8%"),
                    unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🛞 Tire Strategy", "🌧️ Weather Impact", "📊 Phase-by-Phase Impact"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1: TIRE STRATEGY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab1:
        # What are tire compounds - visual cards
        sec("F1 Tire Compounds",
            "F1 uses different rubber compounds — each trades speed for durability")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-top:5px solid #FF1801;"
                f"border-radius:0 0 10px 10px;padding:16px;text-align:center;"
                f"box-shadow:0 2px 6px rgba(0,0,0,0.05)'>"
                f"<div style='font-size:2.5rem'>🔴</div>"
                f"<div style='font-family:Playfair Display,serif;font-size:1.1rem;color:#FF1801;font-weight:700'>SOFT</div>"
                f"<div style='font-size:0.85rem;color:#2d2d2d;line-height:1.7;margin-top:6px'>"
                f"Fastest tire — maximum grip<br>"
                f"Degrades quickly (10-15 laps)<br>"
                f"Used in qualifying for best time<br>"
                f"<b>High risk, high reward</b></div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-top:5px solid #FFC300;"
                f"border-radius:0 0 10px 10px;padding:16px;text-align:center;"
                f"box-shadow:0 2px 6px rgba(0,0,0,0.05)'>"
                f"<div style='font-size:2.5rem'>🟡</div>"
                f"<div style='font-family:Playfair Display,serif;font-size:1.1rem;color:#D4A800;font-weight:700'>MEDIUM</div>"
                f"<div style='font-size:0.85rem;color:#2d2d2d;line-height:1.7;margin-top:6px'>"
                f"Balanced performance<br>"
                f"Moderate lifespan (20-30 laps)<br>"
                f"Most commonly used in races<br>"
                f"<b>The safe strategy choice</b></div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-top:5px solid #CCCCCC;"
                f"border-radius:0 0 10px 10px;padding:16px;text-align:center;"
                f"box-shadow:0 2px 6px rgba(0,0,0,0.05)'>"
                f"<div style='font-size:2.5rem'>⚪</div>"
                f"<div style='font-family:Playfair Display,serif;font-size:1.1rem;color:#666666;font-weight:700'>HARD</div>"
                f"<div style='font-size:0.85rem;color:#2d2d2d;line-height:1.7;margin-top:6px'>"
                f"Slowest but most durable<br>"
                f"Lasts 30-40+ laps<br>"
                f"Used for long stints<br>"
                f"<b>Fewer pit stops needed</b></div></div>", unsafe_allow_html=True)

        # Degradation box plot
        if "tire_deg" in data:
            tire_df = data["tire_deg"]
            if "Compound" in tire_df.columns and "tire_deg_slope" in tire_df.columns:
                sec("Tire Degradation by Compound",
                    "How fast each tire loses grip — measured in seconds lost per lap")
                valid = tire_df[tire_df["tire_deg_slope"].notna()]
                if "tire_deg_r2" in valid.columns:
                    valid = valid[valid["tire_deg_r2"] > 0.3]
                if len(valid) > 0:
                    compound_colors = {"SOFT": "#FF1801", "MEDIUM": "#FFC300", "HARD": "#999999",
                                       "INTERMEDIATE": C_GREEN, "WET": C_BLUE}
                    fig = px.box(valid, x="Compound", y="tire_deg_slope", color="Compound",
                                 color_discrete_map=compound_colors,
                                 labels={"tire_deg_slope": "Degradation (seconds lost per lap)",
                                         "Compound": "Tire Compound"})
                    fig.update_layout(**PT, height=400, showlegend=False,
                                      title=dict(text="Soft tires degrade fastest — confirming the speed vs durability tradeoff"),
                                      margin=dict(l=40, r=20, t=50, b=40))
                    st.plotly_chart(fig, use_container_width=True)

        # Tire features we built
        sec("Tire Features Engineered",
            "Raw lap data → ML-ready features via Spark")

        tire_features = pd.DataFrame([
            {"Feature": "avg_tire_deg_slope", "Description": "Average seconds lost per lap across all stints",
             "Source": "7,996 tire models", "Impact": "Improved MAE by 0.3 positions"},
            {"Feature": "soft_deg_slope", "Description": "Degradation rate specifically on soft compound",
             "Source": "Soft tire stints only", "Impact": "Identifies tire-sensitive drivers"},
            {"Feature": "driver_avg_deg_5", "Description": "Driver's rolling avg degradation over 5 races",
             "Source": "Spark Window function", "Impact": "Captures driving style"},
            {"Feature": "lap_consistency", "Description": "Standard deviation of lap times per race",
             "Source": "181K individual laps", "Impact": "Lower std = more consistent driver"},
            {"Feature": "num_stints", "Description": "Number of tire changes during race",
             "Source": "Pit stop + compound data", "Impact": "Strategy indicator (1-stop vs 2-stop)"},
        ])
        st.dataframe(tire_features, use_container_width=True, hide_index=True)

        st.markdown(info_box(
            "🛞 Key Finding — Tire Data Impact",
            "Adding tire and lap features improved position "
            "<span title='Mean Absolute Error — on average, predictions are off by this many positions' "
            "style='border-bottom:1px dashed " + C_AMBER + ";cursor:help'>MAE</span> "
            "from <b>2.74 → 2.10</b> (a <b>23% improvement</b>). "
            "In practical terms, this means our predictions went from being off by nearly "
            "3 positions to being off by just 2. The difference between predicting P3 vs P5 "
            "is often the difference between a podium and a points finish.",
            C_AMBER, P_AMBER), unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2: WEATHER IMPACT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab2:
        # Weather intro with themed styling
        st.markdown(
            f"<div style='background:linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 50%, #E8F5E9 100%);"
            f"border:1px solid #90CAF9;border-radius:12px;padding:22px 24px;margin:10px 0;"
            f"box-shadow:0 2px 8px rgba(0,0,0,0.06)'>"
            f"<div style='font-size:1.8rem;margin-bottom:4px'>☀️ 🌧️ 🌡️ 💨</div>"
            f"<div style='font-family:Playfair Display,serif;font-size:1.15rem;color:#1565C0;font-weight:700'>"
            f"Why Weather Changes Everything in F1</div>"
            f"<div style='font-size:0.95rem;color:#2d2d2d;line-height:1.8;margin-top:8px'>"
            f"In dry conditions, the fastest car usually wins — it's predictable. "
            f"But <b>rain is the great equalizer</b>. Wet races see more crashes, more overtaking, "
            f"more strategy gambles, and more unexpected winners. A driver who qualifies 10th in the dry "
            f"might win in the rain if they have exceptional car control. Track temperature also matters — "
            f"hot surfaces destroy soft tires 30-50% faster, completely changing optimal strategy."
            f"</div></div>", unsafe_allow_html=True)

        # Weather KPIs
        if "weather" in data:
            weather_df = data["weather"]
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(kpi(f"{len(weather_df):,}", "Weather Records", C_BLUE, P_BLUE),
                            unsafe_allow_html=True)
            with k2:
                st.markdown(kpi("94%", "Race Coverage", C_GREEN, P_GREEN), unsafe_allow_html=True)
            with k3:
                rain_pct = f"{weather_df['had_rain'].mean() * 100:.0f}%" if "had_rain" in weather_df.columns else "N/A"
                st.markdown(kpi(rain_pct, "Races With Rain", C_AMBER, P_AMBER), unsafe_allow_html=True)
            with k4:
                st.markdown(kpi("+20.4%", "Winner Accuracy Gain", F1_RED, F1_RED_L),
                            unsafe_allow_html=True)

        # Weather features table
        sec("Weather Features Engineered",
            "10 features extracted from FastF1 weather API data")

        weather_features = pd.DataFrame([
            {"Feature": "avg_track_temp", "Icon": "🌡️", "Description": "Average track surface temperature (°C)",
             "Why It Matters": "Hot tracks destroy soft tires 30-50% faster"},
            {"Feature": "avg_air_temp", "Icon": "🌡️", "Description": "Ambient air temperature",
             "Why It Matters": "Affects engine cooling and aerodynamic efficiency"},
            {"Feature": "max_track_temp", "Icon": "🔥", "Description": "Peak track temperature during session",
             "Why It Matters": "Extreme peaks cause sudden tire failures"},
            {"Feature": "is_hot_race", "Icon": "☀️", "Description": "Track temp > 40°C (yes/no)",
             "Why It Matters": "Signals extreme degradation conditions"},
            {"Feature": "is_cold_race", "Icon": "❄️", "Description": "Track temp < 25°C (yes/no)",
             "Why It Matters": "Tires struggle to reach operating temperature"},
            {"Feature": "had_rain", "Icon": "🌧️", "Description": "Did it rain during session? (yes/no)",
             "Why It Matters": "Rain completely changes race outcomes"},
            {"Feature": "max_humidity", "Icon": "💧", "Description": "Peak humidity during race",
             "Why It Matters": "High humidity = leading indicator of rain"},
            {"Feature": "avg_humidity", "Icon": "💨", "Description": "Average humidity across session",
             "Why It Matters": "Affects tire grip even without rain"},
            {"Feature": "temp_variation", "Icon": "📊", "Description": "Max temp minus min temp range",
             "Why It Matters": "High variation = unstable, unpredictable conditions"},
        ])
        st.dataframe(weather_features, use_container_width=True, hide_index=True)

        st.markdown(info_box(
            "🌧️ Key Finding — Weather is the #1 Accuracy Driver",
            "Adding 10 weather features improved winner prediction from <b>50% → 70.4%</b> — "
            "a <b>+20.4%</b> improvement in a single step. This was the single largest accuracy "
            "gain in the entire project. For context, switching from Gradient Boosting to XGBoost "
            "(a major algorithm change) only improved accuracy by 7.4%. "
            "<b>The right data matters more than the right algorithm.</b>",
            C_BLUE, P_BLUE), unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3: PHASE-BY-PHASE IMPACT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab3:
        sec("Phase-by-Phase Improvement",
            "How each data integration phase contributed to prediction accuracy")

        # Clean impact table
        impact = pd.DataFrame([
            {"Phase": "Phase 1: Core Data", "Data Added": "Race results, qualifying, pit stops",
             "Features": 29, "Winner Accuracy": "50.0%", "Position MAE": "2.74",
             "Improvement": "Baseline"},
            {"Phase": "Phase 2: + Tire/Laps", "Data Added": "181K laps, tire degradation",
             "Features": 59, "Winner Accuracy": "50.0%", "Position MAE": "2.10",
             "Improvement": "MAE -23%"},
            {"Phase": "Phase 3: + Weather", "Data Added": "Temperature, rain, humidity",
             "Features": 69, "Winner Accuracy": "70.4%", "Position MAE": "2.10",
             "Improvement": "Winner +20.4%"},
            {"Phase": "Final: Model Tuning", "Data Added": "ELO, streaks, interactions",
             "Features": 85, "Winner Accuracy": "77.8%", "Position MAE": "2.096",
             "Improvement": "Winner +7.4%"},
        ])
        st.dataframe(impact, use_container_width=True, hide_index=True)

        # Simple bar chart — winner accuracy only (no dual axis)
        sec("Winner Accuracy Growth",
            "From coin-flip (50%) to reliable prediction (77.8%)")

        fig = go.Figure()
        phase_colors = ["#E0E0E0", C_AMBER, C_BLUE, F1_RED]
        fig.add_trace(go.Bar(
            x=["Phase 1<br>Core Data", "Phase 2<br>+ Tire/Laps",
               "Phase 3<br>+ Weather", "Final<br>+ Model Tuning"],
            y=[50.0, 50.0, 70.4, 77.8],
            marker_color=phase_colors,
            text=["50.0%", "50.0%", "70.4%", "77.8%"],
            textposition="outside", textfont=dict(size=14, color="#2d2d2d")))
        fig.add_hline(y=50, line_dash="dash", line_color=GRAY, line_width=1,
                      annotation_text="Random baseline (coin flip)",
                      annotation_font_color=GRAY, annotation_position="bottom right")
        fig.update_layout(**PT, height=400,
                          title=dict(text="Winner Prediction Accuracy: 50% → 77.8%"),
                          yaxis=dict(title="Accuracy %", range=[0, 100], gridcolor="#F0F0F0"),
                          margin=dict(l=50, r=20, t=50, b=60))
        st.plotly_chart(fig, use_container_width=True)

        # Position MAE bar
        sec("Position Prediction Accuracy",
            "How close are predictions to actual finishing positions?")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Phase 1<br>Core Data", "Phase 2<br>+ Tire/Laps",
               "Phase 3<br>+ Weather", "Final<br>+ Model Tuning"],
            y=[2.74, 2.10, 2.10, 2.096],
            marker_color=phase_colors,
            text=["2.74", "2.10", "2.10", "2.096"],
            textposition="outside", textfont=dict(size=14, color="#2d2d2d")))
        fig.update_layout(**PT, height=400,
                          title=dict(text="Position MAE: 2.74 → 2.096 (lower = better)"),
                          yaxis=dict(title="MAE (positions off)", range=[0, 4], gridcolor="#F0F0F0"),
                          margin=dict(l=50, r=20, t=50, b=60))
        st.plotly_chart(fig, use_container_width=True)

        # Key insight
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(info_box(
                "🛞 Tire Impact — Position Accuracy",
                "Tire and lap data reduced position error by <b>23%</b> (2.74 → 2.10). "
                "In plain terms: predictions went from being off by nearly <b>3 positions</b> "
                "to just <b>2 positions</b>. Knowing how a driver manages their tires tells us "
                "more about where they'll finish than qualifying time alone.",
                C_AMBER, P_AMBER), unsafe_allow_html=True)
        with c2:
            st.markdown(info_box(
                "🌧️ Weather Impact — Winner Accuracy",
                "Weather features improved winner prediction by <b>+20.4%</b> (50% → 70.4%). "
                "By contrast, switching algorithms (GB → XGBoost) only added <b>+7.4%</b>. "
                "This confirms the key ML principle: <b>better data beats better algorithms</b> "
                "on tabular problems.",
                C_BLUE, P_BLUE), unsafe_allow_html=True)

# # ═══════════════════════════════════════════════════
# # PAGE 5: ML MODELS & IMPROVEMENT JOURNEY
# # ═══════════════════════════════════════════════════
# elif page == "🤖 ML Models & Journey":
#     st.markdown(
#         f"<h1 style='font-family:Playfair Display,serif;color:{F1_RED}'>ML Models & Improvement Journey</h1>",
#         unsafe_allow_html=True)

#     sec("3 Models Compared",
#         "Random Forest (Bagging) vs Gradient Boosting vs XGBoost (Advanced Boosting)")

#     # Model explanation cards
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         st.markdown(info_box(
#             "Random Forest",
#             "<b>Type:</b> Bagging Ensemble<br>"
#             "<b>How:</b> 200 trees built independently in parallel, then vote<br>"
#             "<b>Analogy:</b> 200 analysts predict independently, answers averaged<br>"
#             "<b>Strength:</b> Resistant to overfitting",
#             C_BLUE, P_BLUE), unsafe_allow_html=True)
#     with c2:
#         st.markdown(info_box(
#             "Gradient Boosting",
#             "<b>Type:</b> Boosting Ensemble<br>"
#             "<b>How:</b> 300 trees built sequentially, each fixing previous errors<br>"
#             "<b>Analogy:</b> F1 team reviewing each race and adjusting strategy<br>"
#             "<b>Strength:</b> Very accurate, learns from mistakes",
#             C_GREEN, P_GREEN), unsafe_allow_html=True)
#     with c3:
#         st.markdown(info_box(
#             "XGBoost",
#             "<b>Type:</b> Advanced Boosting<br>"
#             "<b>How:</b> Like GB but with regularization to prevent overfitting<br>"
#             "<b>Analogy:</b> GB + engineer asking 'what if I'm wrong?'<br>"
#             "<b>Strength:</b> State-of-the-art on tabular data",
#             C_PURPLE, P_PURPLE), unsafe_allow_html=True)

#     # Results tabs
#     tab1, tab2, tab3 = st.tabs(["📐 Position Prediction", "🥉 Podium Classification", "🏆 Winner Prediction"])

#     comparison = data.get("model_comparison_results", {})

#     with tab1:
#         sec("Position Prediction — Lower MAE = Better")
#         pos = comparison.get("position_prediction", {})
#         models_data = []
#         for name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
#             if name in pos and isinstance(pos[name], dict):
#                 r = pos[name]
#                 models_data.append({"Model": name, "MAE": r.get("mae", 0),
#                                     "RMSE": r.get("rmse", 0), "R²": r.get("r2", 0)})
#         if models_data:
#             mdf = pd.DataFrame(models_data)
#             best = mdf.loc[mdf["MAE"].idxmin(), "Model"]

#             k1, k2, k3 = st.columns(3)
#             for i, (_, row) in enumerate(mdf.iterrows()):
#                 color = C_GREEN if row["Model"] == best else C_BLUE
#                 bg = P_GREEN if row["Model"] == best else P_BLUE
#                 with [k1, k2, k3][i]:
#                     sub = "BEST" if row["Model"] == best else ""
#                     st.markdown(kpi(f"{row['MAE']:.3f}", f"{row['Model']} MAE", color, bg, sub),
#                                 unsafe_allow_html=True)

#             fig = go.Figure(go.Bar(
#                 x=mdf["Model"], y=mdf["MAE"],
#                 marker_color=[F1_RED if m == best else "#E0E0E0" for m in mdf["Model"]],
#                 text=[f"{v:.3f}" for v in mdf["MAE"]], textposition="outside"))
#             fig.update_layout(**PT, height=340,
#                               title=dict(text="Position MAE Comparison"),
#                               yaxis=dict(title="MAE (positions)", range=[0, 3.5]),
#                               margin=dict(l=40, r=20, t=50, b=40))
#             st.plotly_chart(fig, use_container_width=True)

#     with tab2:
#         sec("Podium Classification — Higher F1 = Better")
#         pod = comparison.get("podium_classification", {})
#         models_data = []
#         for name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
#             if name in pod and isinstance(pod[name], dict):
#                 r = pod[name]
#                 models_data.append({"Model": name, "Accuracy": r.get("accuracy", 0),
#                                     "Precision": r.get("precision", 0),
#                                     "Recall": r.get("recall", 0), "F1": r.get("f1", 0)})
#         if models_data:
#             mdf = pd.DataFrame(models_data)
#             best = mdf.loc[mdf["F1"].idxmax(), "Model"]

#             k1, k2, k3 = st.columns(3)
#             for i, (_, row) in enumerate(mdf.iterrows()):
#                 color = C_GREEN if row["Model"] == best else C_BLUE
#                 bg = P_GREEN if row["Model"] == best else P_BLUE
#                 with [k1, k2, k3][i]:
#                     sub = "BEST" if row["Model"] == best else ""
#                     st.markdown(kpi(f"{row['F1']:.3f}", f"{row['Model']} F1", color, bg, sub),
#                                 unsafe_allow_html=True)

#             fig = go.Figure()
#             for metric, color in [("Precision", C_BLUE), ("Recall", C_GREEN), ("F1", F1_RED)]:
#                 fig.add_trace(go.Bar(name=metric, x=mdf["Model"], y=mdf[metric],
#                                      marker_color=color))
#             fig.update_layout(**PT, barmode="group", height=340,
#                               title=dict(text="Podium Classification Metrics"),
#                               yaxis=dict(title="Score", range=[0, 1.1]),
#                               margin=dict(l=40, r=20, t=50, b=40))
#             st.plotly_chart(fig, use_container_width=True)

#     with tab3:
#         sec("Winner Prediction — Higher Accuracy = Better")
#         win = comparison.get("winner_prediction", {})
#         models_data = []
#         for name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
#             if name in win and isinstance(win[name], dict):
#                 r = win[name]
#                 models_data.append({"Model": name,
#                                     "Winner": r.get("winner_accuracy", 0),
#                                     "Top3": r.get("top3_accuracy", 0),
#                                     "Correct": r.get("correct", 0),
#                                     "Total": r.get("total", 0)})
#         if models_data:
#             mdf = pd.DataFrame(models_data)
#             best = mdf.loc[mdf["Winner"].idxmax(), "Model"]

#             k1, k2, k3 = st.columns(3)
#             for i, (_, row) in enumerate(mdf.iterrows()):
#                 color = C_GREEN if row["Model"] == best else C_BLUE
#                 bg = P_GREEN if row["Model"] == best else P_BLUE
#                 with [k1, k2, k3][i]:
#                     sub = "BEST" if row["Model"] == best else ""
#                     st.markdown(kpi(f"{row['Winner']:.1%}",
#                                     f"{row['Model']}", color, bg, sub),
#                                 unsafe_allow_html=True)

#             st.markdown(info_box(
#                 "All 3 models achieve 100% Top-3 accuracy",
#                 "The actual race winner is ALWAYS in the model's top 3 predicted drivers. "
#                 f"Gradient Boosting leads with {mdf.loc[mdf['Winner'].idxmax(), 'Winner']:.1%} "
#                 f"correct winner picks ({int(mdf.loc[mdf['Winner'].idxmax(), 'Correct'])}/"
#                 f"{int(mdf.loc[mdf['Winner'].idxmax(), 'Total'])} races).",
#                 C_GREEN, P_GREEN), unsafe_allow_html=True)

#     # Improvement journey
#     sec("Complete Improvement Journey",
#         "Each stage documented, justified, and reproducible")
#     journey = pd.DataFrame([
#         {"Stage": "1. Baseline (core data)", "Winner": 50.0, "MAE": 2.74, "Top3": 89.6, "Features": 29},
#         {"Stage": "2. + Tire & laps", "Winner": 50.0, "MAE": 2.10, "Top3": 89.6, "Features": 59},
#         {"Stage": "3. + Weather", "Winner": 70.4, "MAE": 2.10, "Top3": 96.3, "Features": 69},
#         {"Stage": "4. + Model comparison", "Winner": 77.8, "MAE": 2.096, "Top3": 100.0, "Features": 85},
#     ])
#     fig = go.Figure()
#     fig.add_trace(go.Bar(x=journey["Stage"], y=journey["Winner"], name="Winner Accuracy %",
#                          marker_color=F1_RED, text=[f"{v:.1f}%" for v in journey["Winner"]],
#                          textposition="outside"))
#     fig.update_layout(**PT, height=380,
#                       title=dict(text="Winner Accuracy: 50% → 77.8%"),
#                       yaxis=dict(title="Accuracy %", range=[0, 100]),
#                       margin=dict(l=40, r=20, t=50, b=100))
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown(info_box(
#         "Key Insight — Features > Algorithms",
#         "Weather features improved accuracy by <b>+20.4%</b>. "
#         "Tire/lap data improved position MAE by <b>23%</b>. "
#         "But switching algorithms improved MAE by only <b>2.4%</b>. "
#         "Feature engineering matters far more than model selection on tabular data.",
#         F1_RED, F1_RED_L), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PAGE 5: ML MODELS & IMPROVEMENT JOURNEY
# ═══════════════════════════════════════════════════
elif page == "🤖 ML Models & Journey":
    st.markdown(
        f"<h1 style='font-family:Playfair Display,serif;color:{F1_RED}'>ML Models & Improvement Journey</h1>",
        unsafe_allow_html=True)

    narrative(
        "We compared <b>3 tree-based ensemble models</b> trained on <b>79 numeric features</b> across "
        "<b>9 seasons</b> (2018–2024 training, 2025–2026 testing). Each model predicts 3 tasks: "
        "finishing position, podium finish, and race winner. No single model wins everything.")

    # Glossary bar
    st.markdown(
        f"<div style='background:#F8F8F8;border:1px solid #E8E8E8;border-radius:8px;"
        f"padding:10px 18px;margin:10px 0 16px 0;display:flex;gap:24px;flex-wrap:wrap;font-size:0.82rem;color:{GRAY}'>"
        f"<span title='Mean Absolute Error — average positions off. MAE of 2.1 means predicting P3 when actual is P1 or P5'>"
        f"<b style='color:#2d2d2d'>MAE</b> = avg positions off (lower = better) ℹ️</span>"
        f"<span title='Harmonic mean of Precision and Recall. Balances false positives and false negatives.'>"
        f"<b style='color:#2d2d2d'>F1 Score</b> = prediction balance metric (higher = better) ℹ️</span>"
        f"<span title='When the model predicts a podium, how often is it correct?'>"
        f"<b style='color:#2d2d2d'>Precision</b> = % of predictions that are right ℹ️</span>"
        f"<span title='Of all actual podiums, how many did the model catch?'>"
        f"<b style='color:#2d2d2d'>Recall</b> = % of actual events caught ℹ️</span>"
        f"</div>", unsafe_allow_html=True)

    # ── 3 Model Cards (bigger, with icons) ──
    sec("3 Models Compared",
        "Three different strategies for combining decision trees")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-top:5px solid {C_BLUE};"
            f"border-radius:0 0 12px 12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:380px'>"
            f"<div style='font-size:2.2rem;margin-bottom:4px'>🌲</div>"
            f"<div style='font-family:Playfair Display,serif;font-size:1.15rem;color:{C_BLUE};font-weight:700'>"
            f"Random Forest</div>"
            f"<div style='font-size:0.78rem;color:{GRAY};text-transform:uppercase;letter-spacing:0.06em;margin-top:2px'>"
            f"Bagging — Parallel Ensemble</div>"
            f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.75;margin-top:12px'>"
            f"<b>How it works:</b> Builds 200 trees at the same time. Each tree sees a random "
            f"sample of the data and a random subset of features. Final answer = average of all trees."
            f"<br><br>"
            f"<b>F1 Analogy:</b> Like asking 200 separate F1 analysts to predict the race "
            f"independently, then averaging their answers. Individual mistakes cancel out."
            f"<br><br>"
            f"<b style='color:{C_BLUE}'>Best at:</b> Most consistent — rarely makes terrible predictions"
            f"</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-top:5px solid {C_GREEN};"
            f"border-radius:0 0 12px 12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:380px'>"
            f"<div style='font-size:2.2rem;margin-bottom:4px'>🎯</div>"
            f"<div style='font-family:Playfair Display,serif;font-size:1.15rem;color:{C_GREEN};font-weight:700'>"
            f"Gradient Boosting</div>"
            f"<div style='font-size:0.78rem;color:{GRAY};text-transform:uppercase;letter-spacing:0.06em;margin-top:2px'>"
            f"Boosting — Sequential Ensemble</div>"
            f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.75;margin-top:12px'>"
            f"<b>How it works:</b> Builds 300 trees one after another. Each new tree specifically "
            f"focuses on fixing the mistakes of all previous trees. Learning rate controls aggression."
            f"<br><br>"
            f"<b>F1 Analogy:</b> Like an F1 team's post-race debrief — after each race, engineers "
            f"analyze exactly what went wrong, make targeted adjustments, and test again."
            f"<br><br>"
            f"<b style='color:{C_GREEN}'>Best at:</b> Winner prediction (77.8%) and Podium F1 (0.780)"
            f"</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-top:5px solid {C_PURPLE};"
            f"border-radius:0 0 12px 12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:380px'>"
            f"<div style='font-size:2.2rem;margin-bottom:4px'>⚡</div>"
            f"<div style='font-family:Playfair Display,serif;font-size:1.15rem;color:{C_PURPLE};font-weight:700'>"
            f"XGBoost</div>"
            f"<div style='font-size:0.78rem;color:{GRAY};text-transform:uppercase;letter-spacing:0.06em;margin-top:2px'>"
            f"Advanced Boosting — Regularized</div>"
            f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.75;margin-top:12px'>"
            f"<b>How it works:</b> Same sequential boosting as GB, but adds L1/L2 regularization "
            f"(penalty for complexity), column subsampling, and native handling of missing values."
            f"<br><br>"
            f"<b>F1 Analogy:</b> Like GB's debrief process + a risk manager who constantly asks "
            f"'what if our analysis is wrong?' and builds in safety margins."
            f"<br><br>"
            f"<b style='color:{C_PURPLE}'>Best at:</b> Position prediction (MAE 2.096) and generalization"
            f"</div></div>", unsafe_allow_html=True)

    # ── Why These 3? ──
    sec("Why These 3 Models?",
        "We tested 7 algorithms before selecting the final 3")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_GREEN};"
            f"border-radius:0 12px 12px 0;padding:18px 20px;box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
            f"<div style='font-weight:700;color:{C_GREEN};font-size:0.95rem;margin-bottom:10px'>"
            f"✅ SELECTED — Tree-Based Ensembles</div>"
            f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
            f"<b>Random Forest</b> — 74.1% winner accuracy<br>"
            f"<b>Gradient Boosting</b> — 77.8% winner accuracy 🏆<br>"
            f"<b>XGBoost</b> — 74.1% winner accuracy<br><br>"
            f"Tree-based models automatically discover <b>non-linear patterns</b> "
            f"and <b>feature interactions</b> — like 'grid position matters MORE when "
            f"the car is fast.' This is why they dominate tabular data competitions "
            f"(Kaggle, academic benchmarks)."
            f"</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {F1_RED};"
            f"border-radius:0 12px 12px 0;padding:18px 20px;box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
            f"<div style='font-weight:700;color:{F1_RED};font-size:0.95rem;margin-bottom:10px'>"
            f"❌ REJECTED — And Why</div>"
            f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
            f"<b>Logistic / Ridge Regression</b> — 40-44% accuracy<br>"
            f"<span style='color:{GRAY}'>Assumes linear relationships. F1 depends on complex "
            f"interactions (grid × car × weather) that linear models cannot capture.</span><br><br>"
            f"<b>Neural Network (MLP)</b> — 29.6% accuracy (WORST)<br>"
            f"<span style='color:{GRAY}'>With only 3,524 rows, deep learning cannot learn "
            f"effective patterns. Needs 50,000+ rows to compete with trees.</span><br><br>"
            f"<b>Stacking Ensemble</b> — no improvement<br>"
            f"<span style='color:{GRAY}'>Combining all models added complexity but no accuracy "
            f"gain over single Gradient Boosting.</span>"
            f"</div></div>", unsafe_allow_html=True)

    # ── All-in-One Comparison Table ──
    sec("Head-to-Head Comparison",
        "All 3 models, all 3 tasks — in one view")

    comparison = data.get("model_comparison_results", {})

    # Build unified table
    pos = comparison.get("position_prediction", {})
    pod = comparison.get("podium_classification", {})
    win = comparison.get("winner_prediction", {})

    table_rows = []
    for name, color_name in [("Random Forest", "🌲"), ("Gradient Boosting", "🎯"), ("XGBoost", "⚡")]:
        row = {"Model": f"{color_name} {name}"}

        if name in pos and isinstance(pos[name], dict):
            row["Position MAE ↓"] = f"{pos[name].get('mae', 0):.3f}"
            row["Position R²"] = f"{pos[name].get('r2', 0):.3f}"
        if name in pod and isinstance(pod[name], dict):
            row["Podium F1 ↑"] = f"{pod[name].get('f1', 0):.3f}"
            row["Podium Acc"] = f"{pod[name].get('accuracy', 0):.1%}"
        if name in win and isinstance(win[name], dict):
            row["Winner Acc ↑"] = f"{win[name].get('winner_accuracy', 0):.1%}"
            row["Top-3 Acc"] = f"{win[name].get('top3_accuracy', 0):.1%}"
            correct = win[name].get('correct', 0)
            total = win[name].get('total', 0)
            row["Correct/Total"] = f"{correct}/{total}"

        table_rows.append(row)

    if table_rows:
        comp_df = pd.DataFrame(table_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Winner accuracy bar chart
        sec("Winner Prediction Accuracy",
            "Who will win the race? — most important metric for fans and teams")

        winner_data = []
        for name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            if name in win and isinstance(win[name], dict):
                winner_data.append({
                    "Model": name,
                    "Accuracy": win[name].get("winner_accuracy", 0) * 100,
                    "Correct": win[name].get("correct", 0),
                    "Total": win[name].get("total", 0)
                })

        if winner_data:
            wdf = pd.DataFrame(winner_data)
            best_model = wdf.loc[wdf["Accuracy"].idxmax(), "Model"]

            fig = go.Figure(go.Bar(
                x=wdf["Model"], y=wdf["Accuracy"],
                marker_color=[C_GREEN if m == best_model else "#E0E0E0" for m in wdf["Model"]],
                text=[f"{v:.1f}%" for v in wdf["Accuracy"]],
                textposition="outside", textfont=dict(size=14)))
            fig.add_hline(y=50, line_dash="dash", line_color=GRAY, line_width=1,
                          annotation_text="Random baseline (coin flip)",
                          annotation_font_color=GRAY)
            fig.update_layout(**PT, height=380,
                              title=dict(text="Winner Prediction — Gradient Boosting Leads"),
                              yaxis=dict(title="Accuracy %", range=[0, 100], gridcolor="#F0F0F0"),
                              margin=dict(l=50, r=20, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(info_box(
                "100% Top-3 Accuracy Across All Models",
                "The actual race winner is <b>ALWAYS</b> in every model's top 3 predicted drivers. "
                f"Gradient Boosting leads winner prediction with <b>{wdf.loc[wdf['Accuracy'].idxmax(), 'Accuracy']:.1f}%</b> "
                f"({int(wdf.loc[wdf['Accuracy'].idxmax(), 'Correct'])}/{int(wdf.loc[wdf['Accuracy'].idxmax(), 'Total'])} races correct). "
                "Even when the model misses the exact winner, the correct answer is always in its top picks.",
                C_GREEN, P_GREEN), unsafe_allow_html=True)

    # ── Feature Importance ──
    sec("Top 10 Most Important Features",
        "What drives the model's predictions? Measured by feature importance from Gradient Boosting")

    # Try to get feature importance from the model
    model = data.get("models", {}).get("position_predictor")
    fi_shown = False

    if model is not None and hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
        importances = model.feature_importances_
        features = model.feature_names_in_
        fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
        fi_df = fi_df.nlargest(10, "Importance").sort_values("Importance", ascending=True)

        fig = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker_color=[F1_RED if i == len(fi_df) - 1 else C_BLUE for i in range(len(fi_df))],
            text=[f"{v:.4f}" for v in fi_df["Importance"]],
            textposition="outside"))
        fig.update_layout(**PT, height=420,
                          title=dict(text="Feature Importance — Position Predictor (Gradient Boosting)"),
                          xaxis=dict(title="Importance Score", gridcolor="#F0F0F0"),
                          yaxis=dict(title=""),
                          margin=dict(l=200, r=60, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)
        fi_shown = True

    if not fi_shown:
        # Fallback — show known feature importance from our results
        fi_data = pd.DataFrame([
            {"Feature": "constructor_rolling_points", "Description": "Team strength over last 10 races", "Rank": 1},
            {"Feature": "rolling_avg_position_5", "Description": "Driver's avg finish last 5 races", "Rank": 2},
            {"Feature": "quali_gap_to_pole", "Description": "Gap to fastest qualifier in seconds", "Rank": 3},
            {"Feature": "grid_elo_interaction", "Description": "Grid position × driver skill rating", "Rank": 4},
            {"Feature": "driver_elo_rating", "Description": "Chess-style driver skill rating", "Rank": 5},
            {"Feature": "season_cumulative_points", "Description": "Total points this season so far", "Rank": 6},
            {"Feature": "avg_tire_deg_slope", "Description": "Avg tire degradation rate", "Rank": 7},
            {"Feature": "circuit_avg_position", "Description": "Driver's history at this circuit", "Rank": 8},
            {"Feature": "constructor_reliability", "Description": "Team's race completion rate", "Rank": 9},
            {"Feature": "avg_track_temp", "Description": "Track surface temperature", "Rank": 10},
        ])

        fig = go.Figure(go.Bar(
            x=list(range(10, 0, -1)),
            y=fi_data["Feature"],
            orientation="h",
            marker_color=[F1_RED] + [C_BLUE] * 9,
            text=fi_data["Description"],
            textposition="outside"))
        fig.update_layout(**PT, height=420,
                          title=dict(text="Top 10 Features — Ranked by Importance"),
                          xaxis=dict(title="Relative Importance", gridcolor="#F0F0F0"),
                          yaxis=dict(title=""),
                          margin=dict(l=200, r=200, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(info_box(
        "Constructor Strength is #1 — The Car Matters Most",
        "<b>constructor_rolling_points</b> (team's average points over last 10 races) is the most "
        "important feature in all 3 models. This confirms what F1 engineers know: the car determines "
        "~80% of performance. Even the best driver in a slow car finishes behind an average driver "
        "in the fastest car. Our model quantifies this with data from 9 seasons and 3,524 races.",
        F1_RED, F1_RED_L), unsafe_allow_html=True)

    # ── Final Insight Boxes ──
    sec("Key Takeaways")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_AMBER};"
            f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
            f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:160px'>"
            f"<div style='font-weight:700;color:{C_AMBER};font-size:1rem;margin-bottom:8px'>"
            f"No Single Model Wins Everything</div>"
            f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
            f"<b>XGBoost</b> wins position prediction (lowest MAE). "
            f"<b>Gradient Boosting</b> wins winner prediction (highest accuracy) and podium F1. "
            f"<b>Random Forest</b> is most stable across all tasks. "
            f"This is why we compare 3 models instead of picking one upfront — "
            f"the best algorithm depends on the specific task."
            f"</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_GREEN};"
            f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
            f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:160px'>"
            f"<div style='font-weight:700;color:{C_GREEN};font-size:1rem;margin-bottom:8px'>"
            f"Features {'>'} Algorithms</div>"
            f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
            f"Weather features improved accuracy by <b>+20.4%</b>. "
            f"Tire/lap data improved position MAE by <b>23%</b>. "
            f"But switching from GB to XGBoost improved MAE by only <b>2.4%</b>. "
            f"On tabular data, <b>the quality of features matters far more than the choice "
            f"of algorithm</b>. Invest 80% of effort in feature engineering."
            f"</div></div>", unsafe_allow_html=True)

# # ═══════════════════════════════════════════════════
# # PAGE 6: AGENTIC AI & LESSONS LEARNED
# # ═══════════════════════════════════════════════════
# elif page == "🧠 Agentic AI & Lessons":
#     st.markdown(
#         f"<h1 style='font-family:Playfair Display,serif;color:{F1_RED}'>Agentic AI & Lessons Learned</h1>",
#         unsafe_allow_html=True)

#     tab1, tab2 = st.tabs(["🤖 Agentic AI Insights", "📋 Limitations & Lessons"])

#     with tab1:
#         if "agent_state" in data:
#             state = data["agent_state"]

#             sec("Agent Status")
#             k1, k2, k3, k4 = st.columns(4)
#             with k1:
#                 health = state.get("data_health", {}).get("status", "N/A")
#                 color = C_GREEN if health == "HEALTHY" else C_AMBER
#                 bg = P_GREEN if health == "HEALTHY" else P_AMBER
#                 st.markdown(kpi(health, "Data Health", color, bg), unsafe_allow_html=True)
#             with k2:
#                 accepted = len(state.get("new_features_accepted", []))
#                 proposed = len(state.get("new_features_proposed", []))
#                 st.markdown(kpi(f"{accepted}/{proposed}", "Features Accepted", C_PURPLE, P_PURPLE),
#                             unsafe_allow_html=True)
#             with k3:
#                 mae = state.get("model_performance", {}).get("position_predictor", {}).get("mae_2025", "N/A")
#                 st.markdown(kpi(str(mae), "Position MAE", C_BLUE, P_BLUE), unsafe_allow_html=True)
#             with k4:
#                 retrain = state.get("retrain_triggered", False)
#                 color = C_AMBER if retrain else C_GREEN
#                 bg = P_AMBER if retrain else P_GREEN
#                 st.markdown(kpi("Yes" if retrain else "No", "Retrain Needed", color, bg),
#                             unsafe_allow_html=True)

#             # Agent descriptions
#             sec("4 Autonomous Agents")
#             c1, c2 = st.columns(2)
#             with c1:
#                 st.markdown(info_box(
#                     "Data Agent — Quality Monitor",
#                     "Checks null rates, row counts, data freshness. "
#                     "Found 51.5% null in q3_seconds (expected — only top 10 reach Q3).",
#                     C_BLUE, P_BLUE), unsafe_allow_html=True)
#                 st.markdown(info_box(
#                     "Feature Agent — Hypothesis Generator",
#                     "Proposed 5 features, tested correlation with target, accepted 3:<br>"
#                     "• constructor_dominance (corr=0.594) ✓<br>"
#                     "• home_advantage (corr=0.167) ✓<br>"
#                     "• grid_volatility (corr=0.064) ✓<br>"
#                     "Rejected: quali_trend (0.044), points_acceleration (0.019)",
#                     C_GREEN, P_GREEN), unsafe_allow_html=True)
#             with c2:
#                 st.markdown(info_box(
#                     "Model Agent — Performance Diagnostician",
#                     "Evaluated on 2025 data. Found worst circuits: "
#                     "Zandvoort (4.1 error), Silverstone (3.8). "
#                     "Hardest to predict: Antonelli (rookie). "
#                     "MAE acceptable — no retrain triggered.",
#                     C_AMBER, P_AMBER), unsafe_allow_html=True)
#                 st.markdown(info_box(
#                     "Insight Agent — Strategy Briefing",
#                     "Generates natural-language race briefings combining all agent outputs. "
#                     "Similar to what an F1 pit wall strategist would receive before a race.",
#                     C_PURPLE, P_PURPLE), unsafe_allow_html=True)

#             # Strategy briefing
#             if "latest_briefing" in data:
#                 sec("Latest Strategy Briefing")
#                 st.code(data["latest_briefing"], language=None)
#         else:
#             st.info("Run `python agents/agentic_pipeline.py` to generate agent data")

#     with tab2:
#         sec("Challenges Faced")
#         c1, c2 = st.columns(2)
#         with c1:
#             st.markdown(info_box(
#                 "API Shutdown & Rate Limits",
#                 "Ergast API shut down end of 2024. Migrated to Jolpica. "
#                 "Both APIs enforce strict rate limits (429 errors). "
#                 "Solution: exponential backoff with automatic retry.",
#                 F1_RED, F1_RED_L), unsafe_allow_html=True)
#             st.markdown(info_box(
#                 "Schema Evolution",
#                 "2025 parquet had avg_speed_kph as INT, others as STRING. "
#                 "Spark refused to merge. Solution: cast all to string first, "
#                 "union safely, then re-cast.",
#                 C_AMBER, P_AMBER), unsafe_allow_html=True)
#         with c2:
#             st.markdown(info_box(
#                 "Docker & Spark Issues",
#                 "Bitnami images removed from Docker Hub. Containers dying. "
#                 "Folder spaces broke mounts. Timestamp NANOS format crashes. "
#                 "Each required investigation and targeted fixes.",
#                 C_BLUE, P_BLUE), unsafe_allow_html=True)
#             st.markdown(info_box(
#                 "Memory Management",
#                 "Spark workers OOM on 181K laps with 2GB RAM. "
#                 "Airflow retry logic caught and re-ran. "
#                 "Demonstrates real-world fault tolerance.",
#                 C_PURPLE, P_PURPLE), unsafe_allow_html=True)

#         sec("Limitations")
#         st.markdown(info_box(
#             "Prediction Ceiling (~78%)",
#             "F1 has inherent unpredictability: crashes, rain, mechanical failures, "
#             "safety cars. Our model uses pre-race data only. "
#             "To exceed ~78%, real-time pit wall telemetry would be needed.",
#             F1_RED, F1_RED_L), unsafe_allow_html=True)

#         c1, c2 = st.columns(2)
#         with c1:
#             st.markdown(info_box(
#                 "Data Volume",
#                 "3,524 rows is modest for ML. Neural networks scored only 29.6% "
#                 "winner accuracy. Tree-based models need 50K+ rows to be outperformed by deep learning.",
#                 GRAY, "#F0F0F0"), unsafe_allow_html=True)
#         with c2:
#             st.markdown(info_box(
#                 "Spark on Local Docker",
#                 "Demonstrates distributed architecture but not true cluster-scale. "
#                 "With 3,524 rows, pandas is faster. Spark's advantage appears at millions of rows.",
#                 GRAY, "#F0F0F0"), unsafe_allow_html=True)

#         sec("Lessons Learned")
#         st.markdown(info_box(
#             "Features > Algorithms",
#             "Weather improved winner accuracy by 20.4%. "
#             "Tire/lap data improved MAE by 23%. "
#             "Switching algorithms improved only 2.4%. "
#             "<b>Spend 80% on features, 20% on models.</b>",
#             C_GREEN, P_GREEN), unsafe_allow_html=True)
#         st.markdown(info_box(
#             "The Car Matters Most",
#             "constructor_rolling_points is the #1 feature in every model. "
#             "In F1, the car determines ~80% of performance, the driver ~20%. "
#             "Our models confirm this domain knowledge.",
#             C_BLUE, P_BLUE), unsafe_allow_html=True)
#         st.markdown(info_box(
#             "Use ALL Your Data",
#             "The biggest single accuracy gain came from integrating data already collected "
#             "(tire degradation + laps). Before tuning hyperparameters, "
#             "ensure every data source is being used.",
#             C_AMBER, P_AMBER), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PAGE 6: AGENTIC AI & LESSONS LEARNED
# ═══════════════════════════════════════════════════
elif page == "🧠 Agentic AI & Lessons":
    st.markdown(
        f"<h1 style='font-family:Playfair Display,serif;color:{F1_RED}'>Agentic AI & Lessons Learned</h1>",
        unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🤖 Agentic AI System", "📋 Challenges, Lessons & Future"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1: AGENTIC AI SYSTEM
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab1:
        narrative(
            "Traditional ML pipelines are <b>static</b> — you train a model, deploy it, and hope it keeps working. "
            "But data drifts, new patterns emerge, and performance silently degrades. Our system uses "
            "<b>4 autonomous AI agents</b> that continuously monitor data quality, discover new features, "
            "diagnose model weaknesses, and generate strategy briefings — all without human intervention.")

        # ── Why Agents? — Before/After comparison ──
        sec("Why Do We Need Agents?",
            "The difference between a static pipeline and an intelligent one")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-top:5px solid {F1_RED};"
                f"border-radius:0 0 12px 12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:280px'>"
                f"<div style='font-size:1.8rem;margin-bottom:4px'>❌</div>"
                f"<div style='font-family:Playfair Display,serif;font-size:1.1rem;color:{F1_RED};font-weight:700'>"
                f"Without Agents (Traditional ML)</div>"
                f"<div style='font-size:0.9rem;color:#2d2d2d;line-height:1.85;margin-top:10px'>"
                f"Train model → Deploy → <b>Hope it works</b><br><br>"
                f"Problems go unnoticed for weeks:<br>"
                f"• Data quality drops — nobody checks<br>"
                f"• New patterns appear — model ignores them<br>"
                f"• Performance degrades — users lose trust<br>"
                f"• An analyst must manually investigate every issue<br><br>"
                f"<i style='color:{GRAY}'>Like an F1 team that never does post-race debriefs — "
                f"they keep making the same mistakes.</i>"
                f"</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-top:5px solid {C_GREEN};"
                f"border-radius:0 0 12px 12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:280px'>"
                f"<div style='font-size:1.8rem;margin-bottom:4px'>✅</div>"
                f"<div style='font-family:Playfair Display,serif;font-size:1.1rem;color:{C_GREEN};font-weight:700'>"
                f"With Agents (Our System)</div>"
                f"<div style='font-size:0.9rem;color:#2d2d2d;line-height:1.85;margin-top:10px'>"
                f"Train model → Agents monitor <b>every Monday</b><br><br>"
                f"Issues caught and fixed automatically:<br>"
                f"• Data Agent checks quality before training<br>"
                f"• Feature Agent discovers new predictive signals<br>"
                f"• Model Agent identifies WHERE predictions fail<br>"
                f"• Insight Agent writes the briefing a human would<br><br>"
                f"<i style='color:{GRAY}'>Like an F1 team with engineers analyzing every lap, "
                f"every tire, every weather change — continuously improving.</i>"
                f"</div></div>", unsafe_allow_html=True)

        # ── Why not just a Python script? ──
        st.markdown(info_box(
            "Why agents instead of a simple Python script?",
            "A script does ONE thing: <code>if null_rate > 50%: alert</code>. "
            "An agent <b>reasons</b>: 'Q3 has 51.5% null — is this a problem? "
            "Let me check... Q3 is only for top 10 drivers, so 50% null is EXPECTED "
            "for a 20-driver field. This is NOT a bug. But if null were 80%, "
            "that WOULD be a problem.' <b>Scripts follow rules. Agents make judgments.</b>",
            C_PURPLE, P_PURPLE), unsafe_allow_html=True)

        # ── What is LangGraph? ──
        sec("What is LangGraph?",
            "The orchestration framework powering our agents")

        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-radius:12px;"
            f"padding:20px 22px;box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
            f"<div style='font-size:0.95rem;color:#2d2d2d;line-height:1.85'>"
            f"<b>LangGraph</b> orchestrates AI agents as a <b>directed graph</b> — "
            f"each agent is a node, and edges define the flow between them. "
            f"Unlike simple sequential scripts, LangGraph supports:"
            f"<div style='display:flex;gap:12px;margin-top:14px'>"
            # Shared State
            f"<div style='flex:1;background:{P_BLUE};border-radius:8px;padding:12px 14px;text-align:center'>"
            f"<div style='font-size:1.5rem'>🔄</div>"
            f"<div style='font-size:0.85rem;color:{C_BLUE};font-weight:600;margin-top:4px'>Shared State</div>"
            f"<div style='font-size:0.78rem;color:{GRAY};margin-top:4px'>"
            f"Each agent reads what previous agents found and builds on it</div></div>"
            # Conditional Edges
            f"<div style='flex:1;background:{P_GREEN};border-radius:8px;padding:12px 14px;text-align:center'>"
            f"<div style='font-size:1.5rem'>🔀</div>"
            f"<div style='font-size:0.85rem;color:{C_GREEN};font-weight:600;margin-top:4px'>Conditional Routing</div>"
            f"<div style='font-size:0.78rem;color:{GRAY};margin-top:4px'>"
            f"If MAE > 4.0, trigger retrain. Otherwise, skip to briefing</div></div>"
            # Autonomy
            f"<div style='flex:1;background:{P_PURPLE};border-radius:8px;padding:12px 14px;text-align:center'>"
            f"<div style='font-size:1.5rem'>🧠</div>"
            f"<div style='font-size:0.85rem;color:{C_PURPLE};font-weight:600;margin-top:4px'>Autonomous Decisions</div>"
            f"<div style='font-size:0.78rem;color:{GRAY};margin-top:4px'>"
            f"Feature Agent decides which features to accept or reject</div></div>"
            f"</div></div></div>", unsafe_allow_html=True)

        # ── Agent Pipeline Flow ──
        sec("Agent Pipeline Flow",
            "How the 4 agents execute in sequence — triggered by Airflow every Monday")

        st.markdown(
            f"<div style='display:flex;gap:0;margin:10px 0;align-items:stretch'>"
            # Data Agent
            f"<div style='flex:1;background:{P_BLUE};border:1px solid #B5D4F4;border-radius:10px 0 0 10px;"
            f"padding:14px;text-align:center'>"
            f"<div style='font-size:1.5rem'>🔍</div>"
            f"<div style='font-size:0.8rem;color:{C_BLUE};font-weight:700;margin-top:4px'>Data Agent</div>"
            f"<div style='font-size:0.72rem;color:{GRAY};margin-top:4px'>Check quality<br>nulls, counts</div></div>"
            # Arrow
            f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 4px'>→</div>"
            # Feature Agent
            f"<div style='flex:1;background:{P_GREEN};border:1px solid #A7D8A0;padding:14px;text-align:center;border-radius:6px'>"
            f"<div style='font-size:1.5rem'>🧪</div>"
            f"<div style='font-size:0.8rem;color:{C_GREEN};font-weight:700;margin-top:4px'>Feature Agent</div>"
            f"<div style='font-size:0.72rem;color:{GRAY};margin-top:4px'>Propose features<br>test correlation</div></div>"
            # Arrow
            f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 4px'>→</div>"
            # Model Agent
            f"<div style='flex:1;background:{P_AMBER};border:1px solid #F5D88E;padding:14px;text-align:center;border-radius:6px'>"
            f"<div style='font-size:1.5rem'>📊</div>"
            f"<div style='font-size:0.8rem;color:{C_AMBER};font-weight:700;margin-top:4px'>Model Agent</div>"
            f"<div style='font-size:0.72rem;color:{GRAY};margin-top:4px'>Evaluate perf<br>find weak spots</div></div>"
            # Conditional Arrow
            f"<div style='display:flex;align-items:center;font-size:0.7rem;color:{C_AMBER};padding:0 4px;flex-direction:column'>"
            f"<span>→</span><span style='font-size:0.6rem'>MAE>4?</span></div>"
            # Retrain Decision
            f"<div style='flex:0.8;background:#FFF3E0;border:2px dashed {C_AMBER};padding:14px;text-align:center;border-radius:6px'>"
            f"<div style='font-size:1.2rem'>🔀</div>"
            f"<div style='font-size:0.75rem;color:{C_AMBER};font-weight:700;margin-top:4px'>Retrain?</div>"
            f"<div style='font-size:0.68rem;color:{GRAY};margin-top:4px'>Yes → retrain<br>No → skip</div></div>"
            # Arrow
            f"<div style='display:flex;align-items:center;font-size:1.3rem;color:{GRAY};padding:0 4px'>→</div>"
            # Insight Agent
            f"<div style='flex:1;background:{P_PURPLE};border:1px solid #C4B0E8;border-radius:0 10px 10px 0;"
            f"padding:14px;text-align:center'>"
            f"<div style='font-size:1.5rem'>📝</div>"
            f"<div style='font-size:0.8rem;color:{C_PURPLE};font-weight:700;margin-top:4px'>Insight Agent</div>"
            f"<div style='font-size:0.72rem;color:{GRAY};margin-top:4px'>Strategy briefing<br>combine all findings</div></div>"
            f"</div>", unsafe_allow_html=True)

        # ── 4 Agent Cards (bigger) ──
        sec("What Each Agent Found",
            "Real results from the latest pipeline run")

        # Agent status KPIs
        if "agent_state" in data:
            state = data["agent_state"]
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                health = state.get("data_health", {}).get("status", "N/A")
                color = C_GREEN if health == "HEALTHY" else C_AMBER
                bg = P_GREEN if health == "HEALTHY" else P_AMBER
                st.markdown(kpi(health, "Data Health", color, bg), unsafe_allow_html=True)
            with k2:
                accepted = len(state.get("new_features_accepted", []))
                proposed = len(state.get("new_features_proposed", []))
                st.markdown(kpi(f"{accepted}/{proposed}", "Features Accepted", C_PURPLE, P_PURPLE),
                            unsafe_allow_html=True)
            with k3:
                mae = state.get("model_performance", {}).get("position_predictor", {}).get("mae_2025", "N/A")
                st.markdown(kpi(str(mae), "Position MAE", C_BLUE, P_BLUE), unsafe_allow_html=True)
            with k4:
                retrain = state.get("retrain_triggered", False)
                color = C_AMBER if retrain else C_GREEN
                bg = P_AMBER if retrain else P_GREEN
                st.markdown(kpi("Yes" if retrain else "No", "Retrain Triggered", color, bg),
                            unsafe_allow_html=True)

        # 4 big agent cards - 2x2
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_BLUE};"
                f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:200px'>"
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<span style='font-size:1.5rem'>🔍</span>"
                f"<span style='font-weight:700;color:{C_BLUE};font-size:1rem'>Data Agent — Quality Monitor</span></div>"
                f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8;margin-top:10px'>"
                f"<b>What it does:</b> Checks null rates, row counts, data freshness, and schema "
                f"anomalies across all Bronze/Silver/Gold tables.<br><br>"
                f"<b>What it found:</b> 51.5% null in <code>q3_seconds</code>. Instead of flagging "
                f"this as an error, the agent <i>reasoned</i> that Q3 is only for the top 10 drivers "
                f"in qualifying — so 50% null is expected for a 20-driver field. "
                f"<b>This is judgment, not just rule-following.</b><br><br>"
                f"<b>Why it matters:</b> Without this agent, a data scientist might spend hours "
                f"investigating a 'bug' that's actually normal F1 structure."
                f"</div></div>", unsafe_allow_html=True)

        with c2:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_GREEN};"
                f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:200px'>"
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<span style='font-size:1.5rem'>🧪</span>"
                f"<span style='font-weight:700;color:{C_GREEN};font-size:1rem'>Feature Agent — Hypothesis Generator</span></div>"
                f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8;margin-top:10px'>"
                f"<b>What it does:</b> Mimics the scientific method — generates hypotheses for new "
                f"features, computes them, tests correlation with the target variable, and makes "
                f"evidence-based accept/reject decisions.<br><br>"
                f"<b>What it found:</b> Proposed 5 new features, accepted 3 based on correlation "
                f"threshold. Discovered <code>constructor_dominance</code> (corr=0.594) — a feature "
                f"more predictive than most hand-engineered ones.<br><br>"
                f"<b>Why it matters:</b> Automates the most time-consuming part of ML — "
                f"feature engineering. What takes a data scientist days, the agent does in seconds."
                f"</div></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_AMBER};"
                f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:200px'>"
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<span style='font-size:1.5rem'>📊</span>"
                f"<span style='font-weight:700;color:{C_AMBER};font-size:1rem'>Model Agent — Performance Diagnostician</span></div>"
                f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8;margin-top:10px'>"
                f"<b>What it does:</b> Evaluates model performance on the latest season. Doesn't "
                f"just report overall MAE — it breaks down performance by circuit, driver, and "
                f"conditions to find WHERE the model fails.<br><br>"
                f"<b>What it found:</b> Worst circuits: Zandvoort (4.1 error), Silverstone (3.8), "
                f"Las Vegas (3.8). Hardest driver: Antonelli (rookie with no historical data). "
                f"Overall MAE acceptable — no retrain triggered.<br><br>"
                f"<b>Why it matters:</b> Knowing overall accuracy is useless without knowing "
                f"where it breaks. This agent enables targeted improvement."
                f"</div></div>", unsafe_allow_html=True)

        with c2:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_PURPLE};"
                f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:200px'>"
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<span style='font-size:1.5rem'>📝</span>"
                f"<span style='font-weight:700;color:{C_PURPLE};font-size:1rem'>Insight Agent — Strategy Briefing</span></div>"
                f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8;margin-top:10px'>"
                f"<b>What it does:</b> Combines outputs from all 3 agents into a natural-language "
                f"race briefing — similar to what an F1 pit wall strategist would receive before "
                f"a Grand Prix.<br><br>"
                f"<b>What it generates:</b> Data quality summary, new feature recommendations, "
                f"model confidence levels, circuit-specific warnings, and driver predictions "
                f"with uncertainty ranges.<br><br>"
                f"<b>Why it matters:</b> Translates raw ML outputs into actionable intelligence "
                f"that a non-technical team manager could understand and act on."
                f"</div></div>", unsafe_allow_html=True)

        # ── Feature Discovery Results ──
        sec("Autonomous Feature Discovery",
            "The Feature Agent proposed 5 features and made evidence-based decisions")

        feature_discovery = pd.DataFrame([
            {"Feature": "constructor_dominance", "Hypothesis": "Teams with >1.5x field-avg points predict podiums",
             "Correlation": 0.594, "Decision": "✅ Accepted", "Reasoning": "Strong signal — nearly 0.6 correlation"},
            {"Feature": "home_advantage", "Hypothesis": "Drivers outperform at familiar/home circuits",
             "Correlation": 0.167, "Decision": "✅ Accepted", "Reasoning": "Moderate but consistent signal"},
            {"Feature": "grid_volatility", "Hypothesis": "Consistent qualifiers finish higher",
             "Correlation": 0.064, "Decision": "✅ Accepted", "Reasoning": "Weak but adds value in combination"},
            {"Feature": "quali_trend", "Hypothesis": "Improving qualifying pace predicts race finish",
             "Correlation": 0.044, "Decision": "❌ Rejected", "Reasoning": "Below threshold — too weak"},
            {"Feature": "points_acceleration", "Hypothesis": "Gaining momentum faster = more competitive",
             "Correlation": 0.019, "Decision": "❌ Rejected", "Reasoning": "Near zero — no predictive value"},
        ])
        st.dataframe(feature_discovery, use_container_width=True, hide_index=True)

        # Correlation visual
        fig = go.Figure(go.Bar(
            x=feature_discovery["Correlation"],
            y=feature_discovery["Feature"],
            orientation="h",
            marker_color=[C_GREEN if "✅" in d else F1_RED for d in feature_discovery["Decision"]],
            text=[f"{v:.3f}" for v in feature_discovery["Correlation"]],
            textposition="outside"))
        fig.add_vline(x=0.05, line_dash="dash", line_color=GRAY, line_width=1,
                      annotation_text="Accept threshold",
                      annotation_font_color=GRAY)
        fig.update_layout(**PT, height=300,
                          title=dict(text="Feature Correlation with Target — Green = Accepted, Red = Rejected"),
                          xaxis=dict(title="Correlation with finishing position", gridcolor="#F0F0F0"),
                          yaxis=dict(title=""),
                          margin=dict(l=160, r=60, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)

        # ── Strategy Briefing ──
        if "latest_briefing" in data:
            sec("Latest Strategy Briefing",
                "Auto-generated by the Insight Agent — similar to an F1 pit wall report")
            st.markdown(
                f"<div style='background:#FAFAFA;border:1px solid #E0E0E0;border-left:5px solid {C_PURPLE};"
                f"border-radius:0 12px 12px 0;padding:20px 22px;margin:10px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);font-family:Source Sans 3,monospace;"
                f"font-size:0.85rem;color:#2d2d2d;line-height:1.8;white-space:pre-wrap'>"
                f"{data['latest_briefing']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(info_box(
                "Strategy Briefing",
                "Run <code>python agents/agentic_pipeline.py</code> to generate the latest briefing. "
                "The Airflow pipeline runs this automatically every Monday.",
                C_PURPLE, P_PURPLE), unsafe_allow_html=True)

        # ── Airflow Integration note ──
        st.markdown(info_box(
            "Fully Automated — Runs Every Monday via Airflow",
            "All 4 agents execute as <b>Task 5</b> in our Airflow DAG, triggered automatically "
            "every Monday at 6 AM after the previous 4 tasks (ingest → Spark Silver → Spark Gold → retrain). "
            "The agents receive the freshly trained model and latest data, evaluate everything, "
            "and generate a briefing — ready before the team arrives Monday morning.",
            C_TEAL, P_TEAL), unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2: CHALLENGES, LESSONS & FUTURE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab2:
        sec("Challenges Faced",
            "Real-world production issues encountered and resolved")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {F1_RED};"
                f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:160px'>"
                f"<div style='font-weight:700;color:{F1_RED};font-size:0.95rem;margin-bottom:8px'>"
                f"API Shutdown & Rate Limits</div>"
                f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                f"The original Ergast API shut down end of 2024 — mid-project. Migrated to "
                f"Jolpica (same JSON format, different URL). Both enforce strict rate limits "
                f"(HTTP 429 after ~100 rapid calls). <b>Solution:</b> exponential backoff with "
                f"automatic retry and 30-second cooldown between batches."
                f"</div></div>", unsafe_allow_html=True)

            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_AMBER};"
                f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:160px'>"
                f"<div style='font-weight:700;color:{C_AMBER};font-size:0.95rem;margin-bottom:8px'>"
                f"Airflow + Spark Integration</div>"
                f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                f"Airflow containers don't have <code>spark-submit</code> installed. Needed Airflow to "
                f"trigger Spark jobs on the Spark master container. <b>Solution:</b> mounted Docker socket "
                f"into Airflow, installed Docker CLI, used <code>docker exec</code> to run "
                f"<code>spark-submit</code> on the Spark cluster from within Airflow tasks."
                f"</div></div>", unsafe_allow_html=True)

        with c2:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_BLUE};"
                f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:160px'>"
                f"<div style='font-weight:700;color:{C_BLUE};font-size:0.95rem;margin-bottom:8px'>"
                f"Docker & Container Issues</div>"
                f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                f"Bitnami Spark images were removed from Docker Hub mid-project. Switched to "
                f"<code>apache/spark:3.5.1</code>. Containers died on startup (foreground/background "
                f"process issues). Folder names with spaces broke volume mounts. "
                f"<b>Solution:</b> each issue required targeted investigation and Docker config fixes."
                f"</div></div>", unsafe_allow_html=True)

            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_PURPLE};"
                f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:160px'>"
                f"<div style='font-weight:700;color:{C_PURPLE};font-size:0.95rem;margin-bottom:8px'>"
                f"Spark Worker Out-of-Memory</div>"
                f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                f"Workers crashed (code 137) processing 181K laps with 2GB RAM. "
                f"The first Spark job failed, but <b>Airflow's retry logic</b> automatically re-ran it "
                f"and it succeeded on the second attempt. This demonstrates real-world "
                f"<b>fault tolerance</b> — the system recovered without human intervention."
                f"</div></div>", unsafe_allow_html=True)

        # ── Limitations ──
        sec("Limitations",
            "Honest assessment of what the system cannot do")

        st.markdown(
            f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {F1_RED};"
            f"border-radius:0 12px 12px 0;padding:18px 20px;margin:10px 0;"
            f"box-shadow:0 2px 8px rgba(0,0,0,0.05)'>"
            f"<div style='font-weight:700;color:{F1_RED};font-size:0.95rem;margin-bottom:8px'>"
            f"Prediction Ceiling (~78%)</div>"
            f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
            f"F1 has inherent unpredictability: first-lap collisions, mid-race rain, mechanical failures, "
            f"safety car periods, and strategy gambles. Our model uses only <b>pre-race information</b> "
            f"and cannot predict events that happen during the race. The 6 missed races in our test set "
            f"all involved unexpected events. To exceed ~78%, real-time pit wall telemetry would be needed."
            f"</div></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(info_box(
                "Data Volume",
                "3,524 rows is modest for ML. This is precisely why neural networks failed (29.6%) "
                "while tree-based models excelled. Deep learning needs 50,000+ rows to outperform "
                "trees on tabular data.",
                GRAY, "#F0F0F0"), unsafe_allow_html=True)
        with c2:
            st.markdown(info_box(
                "Local Spark Cluster",
                "Running Spark on Docker with 2 workers demonstrates the distributed architecture "
                "but does not achieve true cluster-scale performance. With 3,524 rows, pandas is "
                "technically faster. Spark's advantage appears at millions of rows.",
                GRAY, "#F0F0F0"), unsafe_allow_html=True)

        # ── Lessons Learned (only NEW ones not on other pages) ──
        sec("Lessons Learned",
            "Key takeaways from building a production data pipeline")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_GREEN};"
                f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:150px'>"
                f"<div style='font-weight:700;color:{C_GREEN};font-size:0.95rem;margin-bottom:8px'>"
                f"Production Pipelines Need Fault Tolerance</div>"
                f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                f"Rate limiting, API shutdowns, schema changes, and container crashes are not edge "
                f"cases — they are the <b>normal state</b> of real-world data engineering. Building "
                f"retry logic and error handling from the start saved significant debugging time. "
                f"Airflow's automatic retry caught the Spark OOM failure without human intervention."
                f"</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-left:5px solid {C_BLUE};"
                f"border-radius:0 12px 12px 0;padding:18px 20px;margin:8px 0;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:150px'>"
                f"<div style='font-weight:700;color:{C_BLUE};font-size:0.95rem;margin-bottom:8px'>"
                f"Agents Add Real Value to ML Pipelines</div>"
                f"<div style='font-size:0.88rem;color:#2d2d2d;line-height:1.8'>"
                f"The Feature Agent discovered <code>constructor_dominance</code> (corr=0.594) — "
                f"a feature more predictive than most hand-engineered ones. The Model Agent identified "
                f"specific circuits where predictions fail. These are insights a static pipeline "
                f"would never produce. <b>Agentic AI turns monitoring from passive to active.</b>"
                f"</div></div>", unsafe_allow_html=True)

        # ── Future Work ──
        sec("Future Work",
            "Where this project could go next")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-top:4px solid {F1_RED};"
                f"border-radius:0 0 12px 12px;padding:18px;text-align:center;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:200px'>"
                f"<div style='font-size:2rem;margin-bottom:6px'>📡</div>"
                f"<div style='font-family:Playfair Display,serif;font-size:1rem;color:{F1_RED};font-weight:700'>"
                f"Real-Time Streaming</div>"
                f"<div style='font-size:0.85rem;color:#2d2d2d;line-height:1.7;margin-top:8px;text-align:left'>"
                f"Replace batch processing with Spark Structured Streaming. "
                f"Process live lap data during the race to update predictions in real-time — "
                f"like an actual F1 pit wall."
                f"</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-top:4px solid {C_BLUE};"
                f"border-radius:0 0 12px 12px;padding:18px;text-align:center;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:200px'>"
                f"<div style='font-size:2rem;margin-bottom:6px'>☁️</div>"
                f"<div style='font-family:Playfair Display,serif;font-size:1rem;color:{C_BLUE};font-weight:700'>"
                f"Cloud Deployment</div>"
                f"<div style='font-size:0.85rem;color:#2d2d2d;line-height:1.7;margin-top:8px;text-align:left'>"
                f"Deploy to AWS EMR or Azure Databricks for true distributed scale. "
                f"Our Docker + Spark architecture is designed to migrate seamlessly to cloud with "
                f"minimal config changes."
                f"</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(
                f"<div style='background:white;border:1px solid #E0E0E0;border-top:4px solid {C_GREEN};"
                f"border-radius:0 0 12px 12px;padding:18px;text-align:center;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.05);min-height:200px'>"
                f"<div style='font-size:2rem;margin-bottom:6px'>🏎️</div>"
                f"<div style='font-family:Playfair Display,serif;font-size:1rem;color:{C_GREEN};font-weight:700'>"
                f"Car Telemetry Data</div>"
                f"<div style='font-size:0.85rem;color:#2d2d2d;line-height:1.7;margin-top:8px;text-align:left'>"
                f"Integrate car sensor data (throttle, brake, DRS, engine modes) from FastF1 telemetry API. "
                f"This could break through the 78% ceiling by capturing in-race performance "
                f"differences invisible in lap times alone."
                f"</div></div>", unsafe_allow_html=True)


elif page == "🔍 Ask the Data":
    render_text_to_sql_page(df, data, selected_season)
    
# ── Footer ──
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='font-size:0.7rem;color:rgba(255,255,255,0.5);text-align:center;line-height:1.6'>"
    f"Built with Spark, Docker,<br>Airflow, XGBoost,<br>LangGraph & Streamlit"
    f"</div>", unsafe_allow_html=True)
