import pandas as pd
from pyspark.sql import SparkSession

class SparkDataLoader:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load(self, file_path: str):
        data = pd.read_csv(file_path).head(500)
        return self.spark.createDataFrame(data)
