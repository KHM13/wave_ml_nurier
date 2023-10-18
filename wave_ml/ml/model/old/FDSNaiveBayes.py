from sklearn.naive_bayes import BernoulliNB

from wave_ml.ml.model.old.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}{util().now_type3()}_FDSNaiveBayes", "a", "DEBUG")


class FDSNaiveBayes(FDSModel):

    def __init__(self, data):
        self.model = BernoulliNB()
        self.log = logger.get_log()
        super().__init__(self.log, self.model, data)

    def set_model(self):
        self.model = BernoulliNB()
