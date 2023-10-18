import math
from multipledispatch import dispatch
from logging import Logger
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import DataFrame, Bucketizer

from wave_ml.ml.data.SparkDataObject import SparkDataObject


class FDSModel:

    pipeline: Pipeline
    pipeline_array: list
    model_result: PipelineModel
    prediction: DataFrame
    prediction_bucketizer_size: float

    model_title = "model_title"
    media_group_name: str
    media_group_code_array: list

    def __init__(self, log, data):
        self.log: Logger = log
        self.data: SparkDataObject = data
        self.media_group_name = data.get_media_group_name()
        self.media_group_code_array = data.get_media_group_code_array()
        self.pipeline_array = self.data.get_pipeline_array()
        self.prediction_bucketizer_size = 0.0

    @dispatch()
    def training(self):
        self.training(self.data.get_train_data())

    @dispatch(DataFrame)
    def training(self, data):
        self.log.info("Model Training")
        self.model_result = self.pipeline.fit(data)

    @dispatch(int)
    def training(self, data_index: int):
        self.training(self.data.get_split_data(data_index))

    @dispatch()
    def predicting(self):
        self.predicting(self.data.get_test_data())

    @dispatch(int)
    def predicting(self, data_index):
        self.predicting(self.data.get_split_data(data_index))

    @dispatch(DataFrame)
    def predicting(self, data):
        self.log.info("Model Predicting")
        self.prediction = self.model_result.transform(data)

    def prediction_result(self):
        return self.prediction.select("prediction", ML_Field.get_label())

    def save_model(self, file_path):
        try:
            self.model_result.write().overwrite().save(file_path)
        except Exception as e:
            print(f"{self.__class__} ERROR : {e}")
            self.log.error(f"{self.__class__} ERROR : {e}")

    def save_pipeline(self, file_path):
        try:
            self.pipeline.write().overwrite().save(file_path)
        except Exception as e:
            print(f"{self.__class__} ERROR : {e}")
            self.log.error(f"{self.__class__} ERROR : {e}")

    def load_pipeline(self, file_path, model_title):
        try:
            self.pipeline = Pipeline.read().load(file_path)
            self.model_title = model_title
        except Exception as e:
            print(f"{self.__class__} ERROR : {e}")
            self.log.error(f"{self.__class__} ERROR : {e}")

    # Prediction Bucketizer
    @dispatch()
    def prediction_transform_bucketizer(self):
        if self.prediction_bucketizer_size != 0.0:
            self.prediction_transform_bucketizer(self.prediction_bucketizer_size)
        else:
            self.prediction_transform_bucketizer(0.5)

    @dispatch(float)
    def prediction_transform_bucketizer(self, threshold):
        splits = [-math.inf, threshold, math.inf]
        self.prediction_transform_bucketizer(splits)

    @dispatch(list)
    def prediction_transform_bucketizer(self, splits):
        column = "prediction"
        bucketizer = Bucketizer(inputCol=column, outputCol=f"{column}Bucketed", splits=splits)
        self.prediction = bucketizer.transform(self.prediction)
        self.prediction = self.prediction.drop(column).withColumnRenamed(f"{column}Bucketed", column)

    def data_show(self):
        return self.data.get_data().show()

    def prediction_show(self):
        return self.prediction.show()

    # Getter / Setter
    def get_model_result(self):
        return self.model_result

    def set_model_result(self, model_result):
        self.model_result = model_result

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def get_prediction(self):
        return self.prediction

    def set_prediction(self, prediction):
        self.prediction = prediction
