from pyspark.ml.classification import NaiveBayes
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import Pipeline

from wave_ml.ml.data.SparkDataObject import SparkDataObject
from wave_ml.ml.model.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}FDSNaiveBayes", "a", "DEBUG")


class FDSNaiveBayes(FDSModel):

    model: NaiveBayes
    train_validation_split: TrainValidationSplit

    def __init__(self, data: SparkDataObject):
        super(FDSNaiveBayes, self).__init__(logger.get_log(), data)
        self.model = NaiveBayes()

    def create(self):
        self.model_title = self.model.__class__.__name__
        self.log = logger.get_log()
        return self

    def set_model(self, label):
        self.log.info(f"{self.model_title} set Model")
        self.model.setFeaturesCol("features")\
            .setLabelCol(label)

        pipeline_stage = self.pipeline_array
        pipeline_stage.append(self.model)

        self.pipeline = Pipeline(stages=pipeline_stage)
        return self
