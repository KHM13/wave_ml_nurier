from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import Pipeline

from wave_ml.ml.data.SparkDataObject import SparkDataObject
from wave_ml.ml.model.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}FDSMultilayerPerceptronClassifier", "a", "DEBUG")


class FDSMultilayerPerceptronClassifier(FDSModel):

    model: MultilayerPerceptronClassifier
    train_validation_split: TrainValidationSplit

    def __init__(self, data: SparkDataObject):
        super(FDSMultilayerPerceptronClassifier, self).__init__(logger.get_log(), data)
        self.model = MultilayerPerceptronClassifier()

    def create(self):
        self.model_title = self.model.__class__.__name__
        self.log = logger.get_log()
        return self

    def set_model(self, block_size: int, max_iter: int, label):
        self.log.info(f"{self.model_title} set Model")
        layers_option: list
        size = self.data.get_columns_size()
        if size > 20:
            layers_option = [int(size-1), int(size/2), int(size/4), int(size/8), 2]
        else:
            layers_option = [9, 4, 2]

        self.model.setLayers(layers_option)\
            .setBlockSize(block_size)\
            .setSeed(10)\
            .setMaxIter(max_iter)\
            .setFeaturesCol("features")\
            .setLabelCol(label)

        self.log.debug(f"maxIter : {max_iter}")
        self.log.debug(f"blockSize : {block_size}")
        self.log.debug(f"Layers : {str(layers_option)}")

        pipeline_stage = self.pipeline_array
        pipeline_stage.append(self.model)

        self.pipeline = Pipeline(stages=pipeline_stage)
        return self
