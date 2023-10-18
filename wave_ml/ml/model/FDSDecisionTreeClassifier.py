from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

from wave_ml.ml.data.SparkDataObject import SparkDataObject
from wave_ml.ml.model.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}FDSDecisionTreeClassifier", "a", "DEBUG")


class FDSDecisionTreeClassifier(FDSModel):

    model: DecisionTreeClassifier
    train_validation_split: TrainValidationSplit

    def __init__(self, data: SparkDataObject):
        super(FDSDecisionTreeClassifier, self).__init__(logger.get_log(), data)
        self.model = DecisionTreeClassifier()

    def create(self):
        self.model_title = self.model.__class__.__name__
        self.log = logger.get_log()
        return self

    def set_model(self, train_size: float, label):
        self.log.info(f"{self.model_title} set Model")
        self.model.setMaxBins(prop().get_model_max_beans())\
            .setFeaturesCol("features")\
            .setLabelCol(label)

        max_depth: list = [5, 10, 15, 20]

        param_grid = ParamGridBuilder()\
            .addGrid(self.model.maxDepth, max_depth)\
            .build()

        self.train_validation_split = TrainValidationSplit()\
            .setEstimator(self.model)\
            .setEvaluator(RegressionEvaluator(labelCol=label))\
            .setEstimatorParamMaps(param_grid)\
            .setTrainRatio(train_size)\
            .setParallelism(3)

        self.log.debug(f"maxDepth : {str(max_depth)}")

        pipeline_stage = self.pipeline_array
        pipeline_stage.append(self.train_validation_split)

        self.pipeline = Pipeline(stages=pipeline_stage)
        return self
