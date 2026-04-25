# 🏎️ F1 Big Data Analytics Platform

**Race Outcome Prediction Using Apache Spark, Docker, Airflow & Machine Learning**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.5.1-orange?logo=apachespark)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![Airflow](https://img.shields.io/badge/Airflow-2.8.1-017CEE?logo=apacheairflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)

> **77.8% Winner Accuracy** • **93.4% Podium Accuracy** • **100% Top-3 Accuracy** • **MAE 2.096**

---

## 📊 Overview

A complete end-to-end big data analytics platform that predicts Formula 1 race outcomes — who will win, who will podium, and where each driver will finish. Built with a **Medallion Lakehouse Architecture**, processed on **Apache Spark**, scheduled via **Apache Airflow**, and monitored by **4 autonomous AI agents**.

| Metric | Result |
|--------|--------|
| Seasons Analyzed | 9 (2018–2026) |
| Training Records | 3,524 race-driver records |
| Laps Processed | 181,721 individual laps |
| ML Features | 85 engineered features |
| Models Compared | 3 (Random Forest, Gradient Boosting, XGBoost) |
| Winner Prediction | 77.8% correct (21/27 races) |
| Top-3 Accuracy | 100% — winner ALWAYS in top 3 |

---

## 🏗️ Architecture

```
Data Sources (Jolpica API, FastF1, Weather)
       │
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   BRONZE    │───▶│   SILVER    │───▶│    GOLD     │───▶│ CONSUMPTION │
│  Raw Data   │    │  Cleaned    │    │ 85 Features │    │ ML + AI +   │
│  Parquet    │    │  Spark Join │    │ Window Fns  │    │ Dashboard   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
   223K records      3,524 rows       3,524 × 85         3 Models
                                                          4 Agents
                                                          6-Page UI
```

---

## 🐳 Docker Infrastructure (7 Containers)

| Container | Technology | Port | Role |
|-----------|-----------|------|------|
| Spark Master | Apache Spark 3.5.1 | 8080 | Distributes computation |
| Spark Worker 1 | Apache Spark 3.5.1 | — | Worker (2 cores, 2GB) |
| Spark Worker 2 | Apache Spark 3.5.1 | — | Second worker |
| PostgreSQL | PostgreSQL 15 | 5433 | Airflow metadata |
| Airflow Web | Airflow 2.8.1 | 8081 | Pipeline monitoring UI |
| Airflow Scheduler | Airflow 2.8.1 | — | Executes scheduled DAGs |
| Jupyter | PySpark | 8888 | Interactive exploration |

---

## 📁 Project Structure

```
f1_bigdata/
├── config/
│   └── settings.py              # API URLs, paths, configuration
├── ingestion/
│   ├── bronze_ingestion.py      # API data fetching (Jolpica + FastF1)
│   ├── silver_transform.py      # Clean, deduplicate, join
│   └── gold_features.py         # 85 ML features via Spark Windows
├── spark_jobs/
│   ├── spark_silver.py          # Spark Silver transform (cluster)
│   └── spark_gold.py            # Spark Gold features (cluster)
├── ml/
│   ├── model_comparison.py      # 3-model comparison (RF, GB, XGBoost)
│   └── integrate_idle_data.py   # Merge tire, laps, standings into Gold
├── agents/
│   └── agentic_pipeline.py      # 4 LangGraph agents
├── dashboard/
│   └── app.py                   # Streamlit 6-page dashboard
├── airflow/
│   └── dags/
│       └── f1_pipeline_dag.py   # 5-task Airflow DAG
├── enhanced_pipeline.py         # Weather feature integration
├── docker-compose.yml           # 7-container orchestration
├── requirements.txt             # Python dependencies
├── data/
│   ├── bronze/                  # Raw parquet files
│   ├── silver/                  # Cleaned race_master
│   └── gold/                    # ML-ready features
├── models/                      # Saved ML models (.pkl)
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop (8GB+ RAM allocated)
- Git

### Step 1: Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/f1-bigdata-analytics.git
cd f1-bigdata-analytics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Ingest Data
```bash
python run_all.py
```
This fetches race results, qualifying, pit stops, standings, laps, and weather from APIs with retry logic.

### Step 3: Build Silver + Gold Layers
```bash
python main.py --layer silver
python main.py --layer gold
```

### Step 4: Integrate All Data Sources
```bash
python ml/integrate_idle_data.py
```

### Step 5: Train & Compare 3 ML Models
```bash
python ml/model_comparison.py
```

### Step 6: Start Docker Stack
```bash
docker-compose up -d
```

### Step 7: Run Agentic AI
```bash
python agents/agentic_pipeline.py
```

### Step 8: Launch Dashboard
```bash
streamlit run dashboard/app.py
```
Open http://localhost:8501
Cloud https://f1-bigdata-analytics.streamlit.app/

---

## 📈 3-Phase Methodology

| Phase | Data Added | Key Improvement |
|-------|-----------|-----------------|
| **Phase 1** | Race results, qualifying, pit stops | Baseline: 50% winner accuracy |
| **Phase 2** | 181K laps, tire degradation | Position MAE: 2.74 → 2.10 (−23%) |
| **Phase 3** | Weather (temperature, rain, humidity) | Winner accuracy: 50% → 77.8% (+27.8%) |

**Key Insight:** Weather features (+20.4% accuracy) contributed 3× more than switching algorithms (+7.4%).

---

## 🤖 3 ML Models Compared

| Model | Position MAE ↓ | Winner Accuracy ↑ | Top-3 |
|-------|---------------|-------------------|-------|
| Random Forest | 2.187 | 74.1% | 100% |
| **Gradient Boosting** | 2.148 | **77.8%** | 100% |
| **XGBoost** | **2.096** | 74.1% | 100% |

No single model wins everything — XGBoost for position, Gradient Boosting for winner prediction.

---

## 🧠 Agentic AI (LangGraph)

4 autonomous agents run every Monday via Airflow:

| Agent | Role | Key Finding |
|-------|------|-------------|
| Data Agent | Quality Monitor | 51.5% null in Q3 is expected (only top 10 reach Q3) |
| Feature Agent | Hypothesis Generator | Discovered constructor_dominance (corr=0.594) |
| Model Agent | Performance Diagnostician | Worst circuit: Zandvoort (4.1 error) |
| Insight Agent | Strategy Briefing | Auto-generates race briefings |

---

## 🌐 Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| Streamlit Dashboard | http://localhost:8501 | — |
| Airflow UI | http://localhost:8081 | admin / admin |
| Spark Master UI | http://localhost:8080 | — |
| Jupyter Lab | http://localhost:8888 | — |

---

## 📊 Dashboard Pages

1. **Project Overview** — KPIs, architecture, championship standings
2. **Drivers & Constructors** — Data engineering, driver trends, team battle
3. **Tire & Weather Impact** — Compound analysis, weather features, phase impact
4. **ML Models & Journey** — 3-model comparison, feature importance
5. **Race Predictions** — Model vs reality, prediction errors, insight boxes
6. **Agentic AI & Lessons** — Agent results, challenges, future work

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| Distributed Processing | Apache Spark 3.5.1 (PySpark) |
| Containerization | Docker Compose (7 services) |
| Job Scheduling | Apache Airflow 2.8.1 |
| ML Models | scikit-learn, XGBoost |
| Agentic AI | LangGraph, LangChain |
| Dashboard | Streamlit, Plotly |
| Data Storage | Apache Parquet |
| Database | PostgreSQL 15 |
| Data Sources | Jolpica API, FastF1 |

---

## 📄 Data Sources

All data is free and open-source:

- **Jolpica API** — Community replacement for Ergast API (discontinued Dec 2024). Race results, qualifying, pit stops, standings.
- **FastF1** — Python library for F1 telemetry. Lap times, tire compounds, weather data.

---

## 📝 License

This project is for educational purposes — 3rd Semester Big Data Analytics coursework, 2026.
