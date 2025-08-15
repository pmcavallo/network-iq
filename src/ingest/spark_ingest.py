import argparse, os
from pyspark.sql import SparkSession, functions as F
from utils.schema import telemetry_schema

def main(input_path: str, output_path: str):
    spark = SparkSession.builder.appName("network-iq-ingest").getOrCreate()
    df = (spark.read
          .option("header", True)
          .schema(telemetry_schema())
          .csv(input_path))

    # Basic cleansing
    df = df.filter(F.col("latency_ms") > 0).filter(F.col("throughput_mbps") >= 0)
    df = df.withColumn("date", F.to_date("timestamp"))
    df = df.withColumn("hour", F.hour("timestamp"))

    # Write Parquet partitioned by date & cell
    (df.repartition("date","cell_id")
       .write.mode("overwrite")
       .partitionBy("date","cell_id")
       .parquet(output_path))
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    main(args.input, args.output)