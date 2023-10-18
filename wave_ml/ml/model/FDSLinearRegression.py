from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

from multipledispatch import dispatch

from wave_ml.ml.data.SparkDataObject import SparkDataObject
from wave_ml.ml.model.FDSModel import FDSModel
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}FDSLinearRegression", "a", "DEBUG")


class FDSLinearRegression(FDSModel):

    model: LinearRegression
    train_validation_split: TrainValidationSplit

    def __init__(self, data: SparkDataObject):
        super(FDSLinearRegression, self).__init__(logger.get_log(), data)
        self.model = LinearRegression()

    def create(self):
        self.model_title = self.model.__class__.__name__
        self.log = logger.get_log()
        return self

    @dispatch(float, float, str)
    def set_model(self, train_size, prediction_bucketizer, label):
        self.log.info(f"{self.model_title} set Model")
        self.model.setFeaturesCol("features").setLabelCol(label)

        max_iter: list = [10, 15, 20]
        reg_param: list = [0.01, 0.03, 0.05, 0.07, 0.1]
        elasticnet_param: list = [0.1, 0.25, 0.5, 0.75, 1.0]

        param_grid = ParamGridBuilder()\
            .addGrid(self.model.maxIter, max_iter)\
            .addGrid(self.model.regParam, reg_param)\
            .addGrid(self.model.fitIntercept, [True])\
            .addGrid(self.model.elasticNetParam, elasticnet_param)\
            .build()

        self.train_validation_split = TrainValidationSplit()\
            .setEstimator(self.model)\
            .setEvaluator(RegressionEvaluator(labelCol=label))\
            .setEstimatorParamMaps(param_grid)\
            .setTrainRatio(train_size)\
            .setParallelism(3)

        self.log.debug(f"prediction Transform Bucketizer : {prediction_bucketizer}")
        self.log.debug(f"maxIter : {str(max_iter)}")
        self.log.debug(f"regParam : {str(reg_param)}")
        self.log.debug(f"elasticNetParam : {str(elasticnet_param)}")

        pipeline_stage = self.pipeline_array
        pipeline_stage.append(self.train_validation_split)

        self.pipeline = Pipeline(stages=pipeline_stage)
        self.prediction_bucketizer_size = prediction_bucketizer
        return self

    @dispatch(float, int, float, float, bool, str)
    def set_model(self, prediction_bucketizer, max_iter, reg_param, elasticnet_param, fitintercept, label):
        self.log.info(f"{self.model_title} set Model")

        self.model.setFeaturesCol("features")\
            .setLabelCol(label)\
            .setMaxIter(max_iter)\
            .setRegParam(reg_param)\
            .setFitIntercept(fitintercept)\
            .setElasticNetParam(elasticnet_param)

        self.log.debug(f"prediction Transform Bucketizer : {prediction_bucketizer}")
        self.log.debug(f"maxIter : {max_iter}")
        self.log.debug(f"regParam : {reg_param}")
        self.log.debug(f"elasticNetParam : {elasticnet_param}")
        self.log.debug(f"fitIntercept : {fitintercept}")

        pipeline_stage = self.pipeline_array
        pipeline_stage.append(self.model)

        self.pipeline = Pipeline(stages=pipeline_stage)
        self.prediction_bucketizer_size = prediction_bucketizer
        return self
