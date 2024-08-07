import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Configurations
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Book Impact Prediction"

class MLFlowDataRetriever:
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = self._setup_mlflow_client()

    def _setup_mlflow_client(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        return MlflowClient()

    def get_runs(self):
        experiment_id = self.client.get_experiment_by_name(self.experiment_name).experiment_id
        return self.client.search_runs(experiment_ids=[experiment_id])

class DataProcessor:
    @staticmethod
    def extract_data(runs):
        worker_counts = []
        mapes = []
        training_times = []

        for run in runs:
            worker_count = run.data.params.get("worker_count")
            if worker_count:
                worker_counts.append(int(worker_count))
                mapes.append(run.data.metrics.get("MAPE"))
                training_times.append(run.data.metrics.get("Training Time"))

        return worker_counts, mapes, training_times

    @staticmethod
    def create_analysis_df(worker_counts, mapes, training_times):
        return pd.DataFrame({
            'Worker Count': worker_counts,
            'MAPE': mapes,
            'Training Time': training_times
        })

class Analysis:
    @staticmethod
    def generate_summary_stats(analysis_df):
        return analysis_df.groupby('Worker Count').agg(
            MAPE_Mean=('MAPE', 'mean'),
            MAPE_StdDev=('MAPE', 'std'),
            Training_Time_Mean=('Training Time', 'mean'),
            Training_Time_StdDev=('Training Time', 'std')
        ).reset_index()

def main():
    data_retriever = MLFlowDataRetriever(MLFLOW_TRACKING_URI, EXPERIMENT_NAME)
    runs = data_retriever.get_runs()

    worker_counts, mapes, training_times = DataProcessor.extract_data(runs)
    analysis_df = DataProcessor.create_analysis_df(worker_counts, mapes, training_times)
    summary_stats = Analysis.generate_summary_stats(analysis_df)

    print("\nPerformance Analysis Summary:")
    print(summary_stats)

if __name__ == "__main__":
    main()
