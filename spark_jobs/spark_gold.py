"""
Gold Layer — PySpark Feature Engineering
Distributed feature computation on Spark cluster.

Submit: spark-submit --master spark://spark-master:7077 spark_jobs/spark_gold.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType


def create_spark_session():
    return (SparkSession.builder
            .appName("F1_BigData_Gold")
            .getOrCreate())


def build_features(spark, silver_path, gold_path):
    """Build ML-ready feature table with Spark window functions."""
    print("🏎️  F1 Gold Layer — Feature Engineering (PySpark)")
    print("=" * 60)

    master = spark.read.parquet(f"{silver_path}/race_master")
    print(f"Loaded race_master: {master.count()} rows")

    # ── Window definitions ──
    driver_window = (Window
                     .partitionBy("driver_id")
                     .orderBy("season", "round")
                     .rowsBetween(-5, -1))

    constructor_window = (Window
                          .partitionBy("constructor_id")
                          .orderBy("season", "round")
                          .rowsBetween(-10, -1))

    circuit_window = (Window
                      .partitionBy("driver_id", "circuit_id")
                      .orderBy("season", "round")
                      .rowsBetween(Window.unboundedPreceding, -1))

    season_window = (Window
                     .partitionBy("driver_id", "season")
                     .orderBy("round")
                     .rowsBetween(Window.unboundedPreceding, -1))

    race_window = Window.partitionBy("season", "round")

    # ── Rolling Driver Form (last 5 races) ──
    master = (master
              .withColumn("rolling_avg_position_5",
                          F.avg("position").over(driver_window))
              .withColumn("rolling_avg_points_5",
                          F.avg("points").over(driver_window))
              .withColumn("rolling_dnf_rate_5",
                          F.avg(F.col("dnf").cast("double")).over(driver_window))
              .withColumn("rolling_positions_gained_5",
                          F.avg("position_gained").over(driver_window)))

    # ── Constructor Strength ──
    master = master.withColumn(
        "constructor_rolling_points",
        F.avg("points").over(constructor_window)
    )

    # ── Grid vs Finish Consistency ──
    master = master.withColumn(
        "grid_finish_delta",
        F.col("grid") - F.coalesce(F.col("position"), F.lit(20))
    )
    master = master.withColumn(
        "rolling_consistency",
        F.stddev("grid_finish_delta").over(driver_window)
    )

    # ── Qualifying Pace Relative to Field ──
    master = master.withColumn(
        "race_best_quali",
        F.min("best_quali_seconds").over(race_window)
    )
    master = (master
              .withColumn("quali_gap_to_pole",
                          F.col("best_quali_seconds") - F.col("race_best_quali"))
              .withColumn("quali_gap_pct",
                          (F.col("quali_gap_to_pole") / F.col("race_best_quali")) * 100))

    # ── Pit Stop Strategy Features ──
    master = master.withColumn(
        "race_avg_pits",
        F.avg("num_pit_stops").over(race_window)
    )
    master = (master
              .withColumn("pit_strategy_aggressive",
                          (F.col("num_pit_stops") >= 3).cast("int"))
              .withColumn("pit_stops_vs_field",
                          F.col("num_pit_stops") - F.col("race_avg_pits")))

    # ── Circuit Historical Performance ──
    master = (master
              .withColumn("circuit_avg_position",
                          F.avg("position").over(circuit_window))
              .withColumn("circuit_races_count",
                          F.count("position").over(circuit_window)))

    # ── Season Momentum ──
    master = master.withColumn(
        "season_cumulative_points",
        F.sum("points").over(season_window)
    )

    # ── Target Variables ──
    master = (master
              .withColumn("target_position", F.col("position"))
              .withColumn("target_podium",
                          (F.col("position") <= 3).cast("int"))
              .withColumn("target_points_finish",
                          (F.col("position") <= 10).cast("int"))
              .withColumn("target_winner",
                          (F.col("position") == 1).cast("int")))

    # ── Drop leaky and temp columns ──
    drop_cols = ["points", "status", "laps_completed", "time_millis",
                 "fastest_lap_rank", "fastest_lap_time", "fastest_lap_seconds",
                 "avg_speed_kph", "finished", "dnf", "position_gained",
                 "grid_finish_delta", "had_slow_stop", "total_pit_time",
                 "avg_pit_duration", "race_best_quali", "race_avg_pits"]

    feature_df = master.drop(*[c for c in drop_cols if c in master.columns])

    # ── Write to Gold ──
    feature_df.write.mode("overwrite").parquet(f"{gold_path}/race_prediction_features")

    count = feature_df.count()
    cols = len(feature_df.columns)
    print(f"✓ Gold race_prediction_features: {count} rows, {cols} columns")
    print(f"\nFeatures: {feature_df.columns}")

    return feature_df


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")


    silver_path = "/opt/spark-data/silver_spark"
    gold_path = "/opt/spark-data/gold_spark"

    build_features(spark, silver_path, gold_path)

    print("\n✅ Spark Gold layer complete!")
    spark.stop()


if __name__ == "__main__":
    main()
