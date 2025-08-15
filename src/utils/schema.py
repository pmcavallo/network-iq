from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType

def telemetry_schema():
    return StructType([
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("cell_id", StringType(), nullable=False),
        StructField("lat", DoubleType(), nullable=True),
        StructField("lon", DoubleType(), nullable=True),
        StructField("rsrp_dbm", DoubleType(), nullable=True),
        StructField("rsrq_db", DoubleType(), nullable=True),
        StructField("sinr_db", DoubleType(), nullable=True),
        StructField("throughput_mbps", DoubleType(), nullable=True),
        StructField("latency_ms", DoubleType(), nullable=True),
        StructField("jitter_ms", DoubleType(), nullable=True),
        StructField("drop_rate", DoubleType(), nullable=True),
        StructField("tech", StringType(), nullable=True),
        StructField("band", StringType(), nullable=True),
    ])