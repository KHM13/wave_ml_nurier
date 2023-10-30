from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

from multipledispatch import dispatch

from wave_ml.ml.data.SparkDataObject import SparkDataObject
from wave_ml.ml.model.FDSModel import FDSModel
from wave_ml.ml.common.CommonProperties import CommonProperties as prop


class FDSLinearRegression(FDSModel):

    model: LinearRegression
    train_validation_split: TrainValidationSplit

    def __init__(self, data: SparkDataObject):
        super(FDSLinearRegression, self).__init__(data)
        self.model = LinearRegression()

    def create(self):
        self.model_title = self.model.__class__.__name__
        return self

    @dispatch(float, str, list, list, list)
    def set_model(self, train_size, label, max_iter, reg_param, elasticnet_param):
        self.model.setFeaturesCol("features").setLabelCol(label)

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

        pipeline_stage = self.pipeline_array
        pipeline_stage.append(self.train_validation_split)

        self.pipeline = Pipeline(stages=pipeline_stage)
        self.prediction_bucketizer_size = 0.5
        return self

    @dispatch(float, str, list, list, list, float)
    def set_model(self, train_size, label, max_iter, reg_param, elasticnet_param, prediction_bucketizer):
        self.model.setFeaturesCol("features").setLabelCol(label)

        param_grid = ParamGridBuilder() \
            .addGrid(self.model.maxIter, max_iter) \
            .addGrid(self.model.regParam, reg_param) \
            .addGrid(self.model.fitIntercept, [True]) \
            .addGrid(self.model.elasticNetParam, elasticnet_param) \
            .build()

        self.train_validation_split = TrainValidationSplit() \
            .setEstimator(self.model) \
            .setEvaluator(RegressionEvaluator(labelCol=label)) \
            .setEstimatorParamMaps(param_grid) \
            .setTrainRatio(train_size) \
            .setParallelism(2)

        pipeline_stage = self.pipeline_array
        pipeline_stage.append(self.train_validation_split)

        self.pipeline = Pipeline(stages=pipeline_stage)
        self.prediction_bucketizer_size = prediction_bucketizer
        return self

    @dispatch(float, int, float, float, bool, str)
    def set_model(self, prediction_bucketizer, max_iter, reg_param, elasticnet_param, fitintercept, label):
        self.model.setFeaturesCol("features")\
            .setLabelCol(label)\
            .setMaxIter(max_iter)\
            .setRegParam(reg_param)\
            .setFitIntercept(fitintercept)\
            .setElasticNetParam(elasticnet_param)

        pipeline_stage = self.pipeline_array
        pipeline_stage.append(self.model)

        self.pipeline = Pipeline(stages=pipeline_stage)
        self.prediction_bucketizer_size = prediction_bucketizer
        return self
