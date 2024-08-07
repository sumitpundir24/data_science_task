from pyspark.sql import SparkSession

def create_spark_session(worker_count):
    spark = SparkSession.builder \
        .appName("Book Impact Prediction") \
        .master(f"local[{worker_count}]") \
        .getOrCreate()
    return spark
