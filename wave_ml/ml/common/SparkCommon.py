from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType
from wave_ml.ml.common.SparkConfProperties import SparkConfProperties as sprop
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from multipledispatch import dispatch
import findspark


class SparkCommon:
    __spark: SparkSession = None
    __instance = None

    def __init__(self):
        SparkCommon.__instance = self
        findspark.init()
        self.set_spark_session()

    @staticmethod
    def getInstance():
        if SparkCommon.__instance is None:
            SparkCommon()
        return SparkCommon.__instance

    def set_spark_session(self):
        # SPARK_JARS = ["org.apache.arrow:arrow-format:7.0.0"]

        self.__spark = SparkSession.builder\
            .appName(prop().get_spark_app_name())\
            .master(prop().get_spark_master())\
            .config("spark.sql.execution.arrow.pyspark.enabled", True)\
            .config("spark.sql.codegen.wholeStage", False) \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
            .getOrCreate()

        self.__spark.sparkContext.setLogLevel("DEBUG")

        # for it in sprop().spark.__iter__():
        #     self.__spark.conf.set(it, sprop().spark.get(it))
        # if prop().is_server():
        #     self.__spark.conf.set("spark.local.dir", prop().get_spark_user_dir())
        #     self.__spark.conf.set("spark.executor.cores", prop().get_spark_executor_cores())  # 실행기에서 사용할 코어 수
        # else:
        #     self.__spark.conf.set("spark.local.dir", prop().get_spark_user_dir())

    def get_spark_session(self):
        if self.__spark is None:
            self.set_spark_session()
        return self.__spark

    @dispatch(str)
    def loading_csv(self, path_csv):
        return self.get_spark_session().read.option("header", "True").option("encoding", "EUC-KR").csv(path_csv)

    @dispatch(str, StructType)
    def loading_csv(self, path_csv, struct_type):
        return self.get_spark_session().read.option("header", "True").option("encoding", "EUC-KR").schema(struct_type).csv(path_csv)
