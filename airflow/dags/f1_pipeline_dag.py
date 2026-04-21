"""
Airflow DAG — F1 Big Data Pipeline
===================================
Runs every Monday after race weekends:
  1. Ingest new race data (Bronze)
  2. Spark Silver transform (on Spark cluster)
  3. Spark Gold features (on Spark cluster)
  4. Integrate all data + retrain ML models
  5. Run Agentic AI + generate briefing

Access Airflow UI: http://localhost:8081 (admin/admin)
Spark Master UI: http://localhost:8080
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "f1_bigdata",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "f1_bigdata_pipeline",
    default_args=default_args,
    description="F1 Big Data: Ingest -> Spark Transform -> ML Train -> Predict",
    schedule_interval="0 6 * * MON",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["f1", "bigdata", "spark", "ml"],
)

# ── Task 1: Bronze Ingestion ──
ingest_bronze = BashOperator(
    task_id="ingest_bronze_latest",
    bash_command="""
        cd /opt/airflow &&
        python -c "
import sys
sys.path.insert(0, '/opt/airflow')
from ingestion.bronze_ingestion import ErgastIngestion, FastF1Ingestion
import fastf1, time

season = {{ execution_date.year }}
ergast = ErgastIngestion()
ergast.ingest_race_results(season)
time.sleep(2)
ergast.ingest_qualifying(season)
time.sleep(2)
ergast.ingest_pit_stops(season)
time.sleep(2)
ergast.ingest_driver_standings(season)

ff1 = FastF1Ingestion()
schedule = fastf1.get_event_schedule(season)
latest_round = schedule[schedule['EventDate'] < '{{ ds }}']['RoundNumber'].max()
if latest_round > 0:
    ff1.ingest_session_laps(season, int(latest_round), 'R')
    ff1.ingest_weather(season, int(latest_round), 'R')
print(f'Ingested up to {season} R{latest_round}')
"
    """,
    dag=dag,
)

# ── Task 2: Spark Silver Transform (runs on Spark cluster!) ──
spark_silver = BashOperator(
    task_id="spark_silver_transform",
    bash_command="""
        docker exec f1_spark_master /opt/spark/bin/spark-submit \
            --master spark://spark-master:7077 \
            --driver-memory 1g \
            --executor-memory 1g \
            /opt/spark-jobs/spark_silver.py
    """,
    dag=dag,
)

# ── Task 3: Spark Gold Features (runs on Spark cluster!) ──
spark_gold = BashOperator(
    task_id="spark_gold_features",
    bash_command="""
        docker exec f1_spark_master /opt/spark/bin/spark-submit \
            --master spark://spark-master:7077 \
            --driver-memory 1g \
            --executor-memory 1g \
            /opt/spark-jobs/spark_gold.py
    """,
    dag=dag,
)

# ── Task 4: Integrate All Data + Retrain Models ──
retrain_models = BashOperator(
    task_id="retrain_models",
    bash_command="""
        cd /opt/airflow &&
        python -c "
import sys
sys.path.insert(0, '/opt/airflow')

from enhanced_pipeline import build_weather_features, merge_weather_into_gold
build_weather_features()
merge_weather_into_gold()

from ml.integrate_idle_data import merge_all_into_gold
merge_all_into_gold()

from ml.model_comparison import main as compare_main
compare_main()
print('Models retrained')
"
    """,
    dag=dag,
)

# ── Task 5: Run Agentic AI ──
agentic_ai = BashOperator(
    task_id="agentic_ai_briefing",
    bash_command="""
        cd /opt/airflow &&
        python -c "
import sys
sys.path.insert(0, '/opt/airflow')
from agents.agentic_pipeline import build_agent_graph, AgentState
from datetime import datetime

pipeline = build_agent_graph()
state = {
    'data_health': {}, 'ingestion_status': '', 'data_issues': [],
    'new_features_proposed': [], 'new_features_accepted': [],
    'feature_test_results': {}, 'model_performance': {},
    'model_recommendations': [], 'retrain_triggered': False,
    'race_briefing': '', 'predictions': {},
    'phase': 'airflow_scheduled', 'errors': [],
    'timestamp': datetime.now().isoformat(),
}
final = pipeline.invoke(state)
print(f'Agentic AI complete')
print(final.get('race_briefing', 'No briefing'))
"
    """,
    dag=dag,
)

# ── DAG Flow ──
ingest_bronze >> spark_silver >> spark_gold >> retrain_models >> agentic_ai
