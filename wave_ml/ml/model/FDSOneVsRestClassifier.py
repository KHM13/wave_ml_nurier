from pyspark.ml.classification import OneVsRest, LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import Pipeline

from wave_ml.ml.data.SparkDataObject import SparkDataObject
from wave_ml.ml.model.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}FDSOneVsRestClassifier", "a", "DEBUG")


class FDSOneVsRestClassifier(FDSModel):

    model: OneVsRest
    train_validation_split: TrainValidationSplit

    def __init__(self, data: SparkDataObject):
        super(FDSOneVsRestClassifier, self).__init__(logger.get_log(), data)
        self.model = OneVsRest()

    def create(self):
        self.model_title = self.model.__class__.__name__
        self.log = logger.get_log()
        return self

    def set_model(self, max_iter: int, label):
        self.log.info(f"{self.model_title} set Model")

        classifier = LogisticRegression().setMaxIter(max_iter)\
            .setTol(1E-6)\
            .setFitIntercept(True)

        self.model.setClassifier(classifier)\
            .setFeaturesCol("features")\
            .setLabelCol(label)

        self.log.debug(f"maxIter : {max_iter}")

        pipeline_stage = self.pipeline_array
        pipeline_stage.append(self.model)

        self.pipeline = Pipeline(stages=pipeline_stage)
        return self
