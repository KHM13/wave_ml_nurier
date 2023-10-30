from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

from wave_ml.ml.data.SparkDataObject import SparkDataObject
from wave_ml.ml.model.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop


class FDSLinearSVC(FDSModel):

    model: LinearSVC
    train_validation_split: TrainValidationSplit

    def __init__(self, data: SparkDataObject):
        super(FDSLinearSVC, self).__init__(data)
        self.model = LinearSVC()

    def create(self):
        self.model_title = self.model.__class__.__name__
        return self

    def set_model(self, train_size, label, max_iter, reg_param):
        self.model.setFeaturesCol("features").setLabelCol(label)

        param_grid = ParamGridBuilder()\
            .addGrid(self.model.maxIter, max_iter)\
            .addGrid(self.model.regParam, reg_param)\
            .addGrid(self.model.fitIntercept, [True])\
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
        return self
