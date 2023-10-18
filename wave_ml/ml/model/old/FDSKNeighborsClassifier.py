from multipledispatch import dispatch
from sklearn.neighbors import KNeighborsClassifier

from wave_ml.ml.model.old.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}{util().now_type3()}_FDSKNeighborsClassifier", "a", "DEBUG")


class FDSKNeighborsClassifier(FDSModel):
    """
    n_neighbors = K (default = 5)
            - k가 작을수록 모델이 복잡해져 과적합이 일어나고 너무 크면 단순해져 성능이 나빠진다

    metric : string or callable, default ‘minkowski’
            - 'manhattan': 맨하튼 거리 측정 방법 사용
            - 'euclidean': 유클리디안 거리 측정 방법 사용
            - 'minkowski': 민코프스키 거리 측정 방법 사용

    p : integer, optional (default = 2)
            - Power parameter for the Minkowski metric.
            When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    weights : str or callable, optional (default = ‘uniform’)
            - 'uniform': 거리에 가중치 부여하지 않음
            - 'distance': 거리에 가중치 부여함
    """

    def __init__(self, data):
        self.model = KNeighborsClassifier()
        self.log = logger.get_log()
        super().__init__(self.log, self.model, data)

    @dispatch(int)
    def set_model(self, n_neighbors):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    @dispatch(str)
    def set_model(self, weights):
        self.model = KNeighborsClassifier(weights=weights)

    @dispatch(int, int)
    def set_model(self, n_neighbors, p):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)

    @dispatch(int, int, str)
    def set_model(self, n_neighbors, p, weights):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, weights=weights)
