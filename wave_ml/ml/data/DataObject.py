import os
import glob
import pandas as pd
import numpy as np
from pandas import DataFrame
from multipledispatch import dispatch
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from pyspark.sql.types import StringType, DoubleType, LongType, IntegerType, FloatType, TimestampType, StructField, StructType
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score


class DataObject:

    __data: DataFrame
    __train_data: DataFrame
    __test_data: DataFrame

    @dispatch(str, str)
    def __init__(self, file_path: str, file_name: str):
        if file_path.__ne__("") and file_name.__ne__(""):
            files = os.path.join(file_path, file_name)
            list_files = glob.glob(files)
            data = []
            for file in list_files:
                try:
                    df = pd.read_csv(file, encoding="euc-kr", index_col=None, header=0)
                except Exception as e:
                    print(e)
                    df = pd.read_csv(file, encoding="utf-8", index_col=None, header=0)
                data.append(df)
            self.__data = pd.concat(data, axis=0, ignore_index=True)

    @dispatch(str)
    def __init__(self, file_name: str):
        if file_name is not None:
            try:
                df = pd.read_csv(file_name, encoding="euc-kr", index_col=None, header=0)
            except Exception as e:
                print(e)
                df = pd.read_csv(file_name, encoding="utf-8", index_col=None, header=0)
            self.__data = pd.DataFrame(df)

    @dispatch(str, list)
    def __init__(self, file_path: str, files: list):
        data = []
        for file in files:
            try:
                df = pd.read_csv(f"{file_path}/{file}", encoding='euc-kr', index_col=None, header=0)
            except Exception as e:
                print(e)
                df = pd.read_csv(f"{file_path}/{file}", encoding='utf-8', index_col=None, header=0)
            data.append(df)
        self.__data = pd.concat(data, axis=0, ignore_index=True)

    def get_data(self):
        return self.__data

    def set_data(self, data: DataFrame):
        self.__data = data

    def get_training_data(self):
        return self.__train_data

    def set_training_data(self, data: DataFrame):
        self.__train_data = data

    def get_test_data(self):
        return self.__test_data

    def set_test_data(self, data: DataFrame):
        self.__test_data = data

    def get_data_schema(self):
        schema = []
        for column, type in self.__data.dtypes.items():
            spark_data_type = get_spark_data_type(type)
            schema.append(StructField(column, spark_data_type))
        return StructType(schema)


def get_spark_data_type(type_name):
    if type_name == "int64": return LongType()
    elif type_name == "int32": return IntegerType()
    elif type_name == "float64": return DoubleType()
    elif type_name == "float32": return FloatType()
    return StringType()
