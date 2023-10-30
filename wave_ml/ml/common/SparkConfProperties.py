import configparser
from configparser import SectionProxy
from multipledispatch import dispatch
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.config.settings.base import BASE_DIR


class SparkConfProperties:
    spark: SectionProxy = {}
    properties_path: str = f"{BASE_DIR}\\ml\\resource\\sparkConf.ini"
    properties = configparser.ConfigParser()

    def __init__(self):
        try:
            self.properties.read(self.properties_path, encoding='UTF-8')
            self.spark = self.properties['SPARK']
        except Exception as e:
            print(f"{self.__class__} init ERROR : {e}")

    @dispatch(str)
    def get_bool_properties(self, key: str):
        if self.spark.get(key).__eq__("1") or self.spark.get(key).lower() == "true":
            return True
        else:
            return False

    @dispatch(str, str)
    def get_bool_properties(self, key: str, value: str):
        if value.lower() == self.spark.get(key).lower():
            return True
        else:
            return False

    def get_integer_properties(self, key: str):
        try:
            value: str = self.spark.get(key)
            return int(value)
        except Exception as e:
            print(f"{self.__class__} get_integer_properties ERROR : {e}")
            return 0

    def get_float_properties(self, key: str):
        try:
            value: str = self.spark.get(key)
            return float(value)
        except Exception as e:
            print(f"{self.__class__} get_float_properties ERROR : {e}")
            return 0.0

    def get_split_properties(self, key: str, seperator: str):
        try:
            value: str = self.spark.get(key)
            return value.split(seperator)
        except Exception as e:
            print(f"{self.__class__} get_split_properties ERROR : {e}")
            return None
