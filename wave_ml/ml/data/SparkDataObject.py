from multipledispatch import dispatch
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.feature import VectorAssembler, StringIndexer, Bucketizer
from pyspark.ml.feature import Normalizer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from wave_ml.ml.common.SparkCommon import SparkCommon as scommon
from wave_ml.ml.data.DataObject import DataObject

import warnings

warnings.filterwarnings("ignore")


class SparkDataObject:
    __data: DataFrame
    __columnList: list
    split_data: list
    pipeline_array: list
    features: VectorAssembler
    label_column: str

    def __init__(self, data: DataObject):
        spark = scommon.getInstance().get_spark_session()

        self.__data = spark.createDataFrame(data.get_data())
        self.__columnList = self.__data.columns
        self.pipeline_array = []

    # Getter / Setter
    def get_data(self):
        return self.__data

    def set_data(self, data):
        self.__data = data
        self.__columnList = self.__data.columns

    def get_pipeline_array(self):
        return self.pipeline_array

    def set_pipeline_array(self, pipeline_array):
        self.pipeline_array = pipeline_array

    def get_features(self):
        return self.features

    def get_label_column(self):
        return self.label_column

    def set_label_column(self, label_column):
        self.label_column = label_column

    def get_columns_size(self):
        return len(self.__columnList)

    # Data Split
    @dispatch(float)
    def set_split_data(self, train_size):
        splits: list = [train_size, 1-train_size]
        self.split_data = self.__data.randomSplit(splits)

    @dispatch(float, int)
    def set_split_data(self, train_size, seed):
        splits: list = [train_size, 1 - train_size]
        self.split_data = self.__data.randomSplit(weights=splits, seed=seed)

    def get_split_data(self, data_index):
        if self.split_data is None:
            return self.__data
        elif len(self.split_data) <= data_index:
            return self.split_data[len(self.split_data)-1]
        else:
            return self.split_data[data_index]

    def get_train_data(self):
        if self.split_data is None:
            return self.__data
        return self.split_data[0]

    def get_test_data(self):
        if self.split_data is None:
            return self.__data
        return self.split_data[1]

    def set_train_data(self, data):
        self.split_data[0] = data

    def set_test_data(self, data):
        self.split_data[1] = data

    # Features Settings
    @dispatch(str, list)
    def append_features(self, output_column, columns):
        self.pipeline_array.append(VectorAssembler(inputCols=columns, outputCol=output_column, handleInvalid="skip"))

    @dispatch(list)
    def append_features(self, columns):
        self.pipeline_array.append(VectorAssembler(inputCols=columns, outputCol="features", handleInvalid="skip"))

    def set_features(self, columns):
        self.features = VectorAssembler(inputCols=columns, outputCol="features", handleInvalid="skip")