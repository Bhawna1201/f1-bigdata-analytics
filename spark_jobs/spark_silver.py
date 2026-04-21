"""
Silver Layer — PySpark
Runs on Spark cluster via docker-compose.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, StringType
import glob



def create_spark_session():
    return (SparkSession.builder
            .appName("F1_BigData_Silver")
            .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
            .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
            .config("spark.sql.legacy.parquet.nanosAsLong", "true")
            .getOrCreate())



def read_parquets_safe(spark, pattern):
    """Read multiple parquet files with mismatched schemas safely."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            d = spark.read.parquet(f)
            for col_name in d.columns:
                d = d.withColumn(col_name, F.col(col_name).cast(StringType()))
            dfs.append(d)
        except Exception as e:
            print(f"  WARN: Skipping {f}: {e}")
    if not dfs:
        return None
    df = dfs[0]
    for d in dfs[1:]:
        df = df.unionByName(d, allowMissingColumns=True)
    return df


def transform_race_results(spark, bronze_path, silver_path):
    print("=" * 60)
    print("SILVER: Race Results")
    print("=" * 60)

    df = read_parquets_safe(spark, f"{bronze_path}/ergast/race_results_*.parquet")
    print(f"Read {df.count()} raw rows")

    df = df.dropDuplicates(["season", "round", "driver_id"])

    df = (df
          .withColumn("season", F.col("season").cast("int"))
          .withColumn("round", F.col("round").cast("int"))
          .withColumn("race_date", F.to_date("race_date"))
          .withColumn("points", F.col("points").cast(DoubleType()))
          .withColumn("grid", F.col("grid").cast("int"))
          .withColumn("position", F.col("position").cast("int"))
          .withColumn("laps_completed", F.col("laps_completed").cast("int"))
          .withColumn("time_millis", F.col("time_millis").cast(DoubleType()))
          .withColumn("avg_speed_kph", F.col("avg_speed_kph").cast(DoubleType())))

    df = df.withColumn(
        "fastest_lap_seconds",
        F.when(
            F.col("fastest_lap_time").contains(":"),
            F.split(F.col("fastest_lap_time"), ":")[0].cast(DoubleType()) * 60 +
            F.split(F.col("fastest_lap_time"), ":")[1].cast(DoubleType())
        ).otherwise(F.lit(None))
    )

    df = (df
          .withColumn("position_gained",
                      F.col("grid") - F.coalesce(F.col("position"), F.col("grid")))
          .withColumn("finished",
                      F.lower(F.col("status")).rlike("finished|\\+"))
          .withColumn("dnf", ~F.col("finished")))

    df = df.drop("_ingested_at")
    df.write.mode("overwrite").parquet(f"{silver_path}/race_results")
    print(f"✓ Silver race_results: {df.count()} rows")
    return df


def transform_qualifying(spark, bronze_path, silver_path):
    print("=" * 60)
    print("SILVER: Qualifying")
    print("=" * 60)

    df = read_parquets_safe(spark, f"{bronze_path}/ergast/qualifying_*.parquet")
    df = (df
          .withColumn("season", F.col("season").cast("int"))
          .withColumn("round", F.col("round").cast("int"))
          .withColumn("position", F.col("position").cast("int")))
    df = df.dropDuplicates(["season", "round", "driver_id"])

    for col_name in ["q1", "q2", "q3"]:
        df = df.withColumn(
            f"{col_name}_seconds",
            F.when(
                F.col(col_name).contains(":"),
                F.split(F.col(col_name), ":")[0].cast(DoubleType()) * 60 +
                F.split(F.col(col_name), ":")[1].cast(DoubleType())
            ).otherwise(F.lit(None))
        )

    df = df.withColumn("best_quali_seconds",
                       F.least("q1_seconds", "q2_seconds", "q3_seconds"))

    df = df.drop("_ingested_at")
    df.write.mode("overwrite").parquet(f"{silver_path}/qualifying")
    print(f"✓ Silver qualifying: {df.count()} rows")
    return df


def transform_pit_stops(spark, bronze_path, silver_path):
    print("=" * 60)
    print("SILVER: Pit Stops")
    print("=" * 60)

    df = read_parquets_safe(spark, f"{bronze_path}/ergast/pit_stops_*.parquet")
    df = (df
          .withColumn("season", F.col("season").cast("int"))
          .withColumn("round", F.col("round").cast("int"))
          .withColumn("stop_number", F.col("stop_number").cast("int"))
          .withColumn("lap", F.col("lap").cast("int")))
    df = df.dropDuplicates(["season", "round", "driver_id", "stop_number"])

    df = (df
          .withColumn("duration_seconds", F.col("duration").cast(DoubleType()))
          .withColumn("is_slow_stop", F.col("duration_seconds") > 5.0)
          .withColumn("is_problem_stop", F.col("duration_seconds") > 15.0))

    df = df.drop("_ingested_at")
    df.write.mode("overwrite").parquet(f"{silver_path}/pit_stops")
    print(f"✓ Silver pit_stops: {df.count()} rows")
    return df


def transform_laps(spark, bronze_path, silver_path):
    print("=" * 60)
    print("SILVER: Laps")
    print("=" * 60)

    df = read_parquets_safe(spark, f"{bronze_path}/fastf1/laps_*.parquet")
    df = (df
          .withColumn("season", F.col("season").cast("int"))
          .withColumn("round", F.col("round").cast("int")))
    df = df.dropDuplicates(["season", "round", "Driver", "LapNumber"])

    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2,
                    "INTERMEDIATE": 3, "WET": 4}
    mapping_expr = F.create_map([F.lit(x) for pair in compound_map.items()
                                 for x in pair])
    if "Compound" in df.columns:
        df = df.withColumn("compound_encoded",
                           F.coalesce(mapping_expr[F.col("Compound")], F.lit(-1)))

    if "LapTime_seconds" in df.columns:
        df = df.withColumn("LapTime_seconds", F.col("LapTime_seconds").cast(DoubleType()))
        w = Window.partitionBy("season", "round", "LapNumber")
        df = df.withColumn("delta_to_leader_seconds",
                           F.col("LapTime_seconds") - F.min("LapTime_seconds").over(w))

    df = df.drop("_ingested_at")
    df.write.mode("overwrite").parquet(f"{silver_path}/laps")
    print(f"✓ Silver laps: {df.count()} rows")
    return df


def build_race_master(spark, silver_path):
    print("=" * 60)
    print("SILVER: Race Master (Joined)")
    print("=" * 60)

    results = spark.read.parquet(f"{silver_path}/race_results")
    quali = spark.read.parquet(f"{silver_path}/qualifying")
    pits = spark.read.parquet(f"{silver_path}/pit_stops")

    pit_agg = (pits.groupBy("season", "round", "driver_id").agg(
        F.max("stop_number").alias("num_pit_stops"),
        F.avg("duration_seconds").alias("avg_pit_duration"),
        F.sum("duration_seconds").alias("total_pit_time"),
        F.max(F.col("is_slow_stop").cast("int")).alias("had_slow_stop"),
    ))

    quali_cols = ["season", "round", "driver_id", "best_quali_seconds",
                  "q1_seconds", "q2_seconds", "q3_seconds"]
    master = results.join(quali.select(*quali_cols),
                          on=["season", "round", "driver_id"], how="left")
    master = master.join(pit_agg, on=["season", "round", "driver_id"], how="left")
    master = master.fillna({"num_pit_stops": 0})

    master.write.mode("overwrite").parquet(f"{silver_path}/race_master")
    print(f"✓ Silver race_master: {master.count()} rows, {len(master.columns)} columns")
    return master


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    bronze_path = "/opt/spark-data/bronze"
    silver_path = "/opt/spark-data/silver_spark"

    print("F1 Big Data — Spark Silver Transform")
    print("=" * 60)

    transform_race_results(spark, bronze_path, silver_path)
    transform_qualifying(spark, bronze_path, silver_path)
    transform_pit_stops(spark, bronze_path, silver_path)
    transform_laps(spark, bronze_path, silver_path)
    build_race_master(spark, silver_path)

    print("\nSpark Silver layer complete!")
    spark.stop()


if __name__ == "__main__":
    main()