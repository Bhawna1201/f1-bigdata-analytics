#!/bin/bash
# ============================================================
# F1 Big Data Stack — Startup Script
# ============================================================
# Prerequisites: Docker Desktop installed and running
# ============================================================

echo "🏎️  F1 Big Data Stack — Starting..."
echo "============================================================"

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Start all services
echo "Starting Spark cluster + Airflow + Jupyter..."
docker-compose up -d

echo ""
echo "============================================================"
echo "  Services starting up (wait ~60 seconds)..."
echo "============================================================"
echo ""
echo "  📊 Spark Master UI:    http://localhost:8080"
echo "  📋 Airflow UI:         http://localhost:8081  (admin/admin)"
echo "  📓 Jupyter Lab:        http://localhost:8888"
echo "  🗄️  PostgreSQL:         localhost:5432"
echo ""
echo "============================================================"
echo "  Quick Start Commands:"
echo "============================================================"
echo ""
echo "  # Submit Spark Silver job:"
echo "  docker exec f1_spark_master spark-submit \\"
echo "    --master spark://spark-master:7077 \\"
echo "    --packages io.delta:delta-core_2.12:2.4.0 \\"
echo "    /opt/spark-jobs/spark_silver.py"
echo ""
echo "  # Submit Spark Gold job:"
echo "  docker exec f1_spark_master spark-submit \\"
echo "    --master spark://spark-master:7077 \\"
echo "    --packages io.delta:delta-core_2.12:2.4.0 \\"
echo "    /opt/spark-jobs/spark_gold.py"
echo ""
echo "  # View Spark workers:"
echo "  docker ps --filter name=f1_spark"
echo ""
echo "  # Stop everything:"
echo "  docker-compose down"
echo ""
echo "  # Stop and remove data:"
echo "  docker-compose down -v"
echo "============================================================"
