import mlflow
import mlflow.spark
import os
from src.spark_session import create_spark_session
from src.data_loader import SparkDataLoader
from src.preprocessor import SparkPreprocessor
from src.model_trainer import LinearRegressionModelTrainer, GradientBoostingModelTrainer, NeuralNetworkModelTrainer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def runner(file_path: str):
    worker_configs = [1, 2, 4]
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Book Impact Prediction")

    os.environ['PYSPARK_PYTHON'] = '/Users/sumitpundir/anaconda3/bin/python3'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/sumitpundir/anaconda3/bin/python3'

    for worker_count in worker_configs:
        try:
            spark = create_spark_session(worker_count)
            data_loader = SparkDataLoader(spark)
            preprocessor = SparkPreprocessor()
            
            df = data_loader.load(file_path)
            processed_df = preprocessor.preprocess(df)
            train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)

            # Choose the model trainer
            model_trainer = LinearRegressionModelTrainer()
            # model_trainer = GradientBoostingModelTrainer()  # Uncomment for Gradient Boosting
            # model_trainer = NeuralNetworkModelTrainer()  # Uncomment for Neural Network

            with mlflow.start_run() as run:
                mlflow.log_param("worker_count", worker_count)
                mlflow.log_param("num_features", 10000)
                mlflow.log_param("cross_validator_folds", 3)

                mae, training_time, model = model_trainer.train_and_evaluate(train_df, test_df)

                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("Training Time", training_time)

                mlflow.spark.log_model(model.bestModel, "model")

                logger.info(f"Worker Count: {worker_count}, MAE: {mae}, Training Time: {training_time}s")

        except Exception as e:
            logger.error(f"Error in processing worker count {worker_count}: {e}")
        finally:
            spark.stop()

if __name__ == "__main__":
    file_path = "books_task.csv"
    runner(file_path)

    
