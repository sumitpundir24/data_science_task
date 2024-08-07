from abc import ABC, abstractmethod
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
import time

class ModelTrainer(ABC):
    @abstractmethod
    def train_and_evaluate(self, train_df: DataFrame, test_df: DataFrame):
        pass

class LinearRegressionModelTrainer(ModelTrainer):
    def train_and_evaluate(self, train_df: DataFrame, test_df: DataFrame):
        lr = LinearRegression(featuresCol="features", labelCol="Impact")
        
        param_grid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.1, 0.01]) \
            .build()
        
        evaluator = RegressionEvaluator(labelCol="Impact", predictionCol="prediction", metricName="mae")
        
        crossval = CrossValidator(estimator=lr,
                                  estimatorParamMaps=param_grid,
                                  evaluator=evaluator,
                                  numFolds=3)
        
        start_time = time.time()
        cv_model = crossval.fit(train_df)
        training_time = time.time() - start_time
        
        predictions = cv_model.transform(test_df)
        mae = evaluator.evaluate(predictions)
        
        return mae, training_time, cv_model

class GradientBoostingModelTrainer(ModelTrainer):
    def train_and_evaluate(self, train_df: DataFrame, test_df: DataFrame):
        gbt = GBTRegressor(featuresCol="features", labelCol="Impact")
        
        param_grid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [5, 10]) \
            .addGrid(gbt.maxIter, [20, 50]) \
            .build()
        
        evaluator = RegressionEvaluator(labelCol="Impact", predictionCol="prediction", metricName="mae")
        
        crossval = CrossValidator(estimator=gbt,
                                  estimatorParamMaps=param_grid,
                                  evaluator=evaluator,
                                  numFolds=3)
        
        start_time = time.time()
        cv_model = crossval.fit(train_df)
        training_time = time.time() - start_time
        
        predictions = cv_model.transform(test_df)
        mae = evaluator.evaluate(predictions)
        
        return mae, training_time, cv_model

class NeuralNetworkModelTrainer(ModelTrainer):
    def train_and_evaluate(self, train_df: DataFrame, test_df: DataFrame):
        layers = [len(train_df.columns) - 1, 10, 5, 1]  # Input layer size, two hidden layers, and output layer
        mlp = MultilayerPerceptronClassifier(featuresCol="features", labelCol="Impact", maxIter=100, layers=layers)
        
        param_grid = ParamGridBuilder() \
            .addGrid(mlp.maxIter, [50, 100]) \
            .addGrid(mlp.blockSize, [128, 256]) \
            .build()
        
        evaluator = RegressionEvaluator(labelCol="Impact", predictionCol="prediction", metricName="mae")
        
        crossval = CrossValidator(estimator=mlp,
                                  estimatorParamMaps=param_grid,
                                  evaluator=evaluator,
                                  numFolds=3)
        
        start_time = time.time()
        cv_model = crossval.fit(train_df)
        training_time = time.time() - start_time
        
        predictions = cv_model.transform(test_df)
        mae = evaluator.evaluate(predictions)
        
        return mae, training_time, cv_model
