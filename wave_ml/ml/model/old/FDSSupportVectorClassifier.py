from multipledispatch import dispatch
from sklearn.svm import SVC

from wave_ml.ml.model.old.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}{util().now_type3()}_FDSSupportVectorClassifier", "a", "DEBUG")


class FDSSupportVectorClassifier(FDSModel):

    """
    C : float, optional (default=1.0)
            - 클수록 정확하게 (마진이 작아짐, 과대적합)
            - 작을수록 과대적합 방지 (과대적합 방지),  alpha (가중치 규제) 의 역수

    gamma : float, optional (default=’auto’)
            - 클수록 정확하게 (경사가 급해짐, 과대적합)
            - 작을수록 과대적합 방지 (과대적합 방지)
            - 비선형 (kernel=‘rbf’, kernel=‘poly’)에서만 이 옵션 사용

    kernel : str, optional
            - linear ( 선형 )
            - poly ( 다항 )
            - rbf, sigmoid ( 비선형 )
    """

    def __init__(self, data):
        self.model = SVC(random_state=42)
        self.log = logger.get_log()
        super().__init__(self.log, self.model, data)

    @dispatch(float)
    def set_model(self, c):
        self.model = SVC(C=c, random_state=42)

    @dispatch(str, float)
    def set_model(self, kernel, gamma):
        self.model = SVC(kernel=kernel, gamma=gamma, random_state=42)

    @dispatch(float, str, float)
    def set_model(self, c, kernel, gamma):
        self.model = SVC(C=c, kernel=kernel, gamma=gamma, random_state=42)
