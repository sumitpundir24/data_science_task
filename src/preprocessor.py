from pyspark.ml.feature import StringIndexer, OneHotEncoder, Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame

class SparkPreprocessor:
    def preprocess(self, df: DataFrame) -> DataFrame:
        df = df.na.fill({"description": "", "authors": "", "categories": ""})
        
        indexer = StringIndexer(inputCols=["publisher", "categories"], outputCols=["publisher_indexed", "categories_indexed"])
        encoder = OneHotEncoder(inputCols=["publisher_indexed", "categories_indexed"], outputCols=["publisher_vec", "categories_vec"])
        tokenizer = Tokenizer(inputCol="description", outputCol="words")
        hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=10000)
        idf = IDF(inputCol="raw_features", outputCol="description_tfidf")
        assembler = VectorAssembler(inputCols=["description_tfidf", "publisher_vec", "categories_vec"], outputCol="features")
        
        pipeline = Pipeline(stages=[indexer, encoder, tokenizer, hashing_tf, idf, assembler])
        return pipeline.fit(df).transform(df)
