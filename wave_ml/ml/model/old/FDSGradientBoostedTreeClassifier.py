from multipledispatch import dispatch
from sklearn.ensemble import GradientBoostingRegressor

from wave_ml.ml.model.old.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}{util().now_type3()}_FDSGradientBoostingRegressor", "a", "DEBUG")


class FDSGradientBoostingRegressor(FDSModel):

    """
    loss      : 최적화할 손실함수를 선택하는 것이다.
                - 'ls'는 최소 제곱(Least Square) 회귀를 나타낸다.
                - 'lad'(최소 절대 편차)는 입력 변수의 순서 정보만을 기반으로하는 매우 강력한 손실 함수이다.
                - 'huber'는 이 둘의 조합이다.
                - 'quantile'은 분위수 회귀를 허용한다. (default=’ls’)

    learning_rate : 각 트리의 기여도를 나타낸다. (default=0.1)
    n_estimators : 부스팅 단계의 수를 나타낸다. (default=100)
    subsample : 샘플비율을 나타낸다. 1보다 작으면 확률적 그라데이션 부스팅이 발생한다. (default=1.0)
    criterion : 분할하는데의 기준을 선택하는 것이다. 'friedman_mse', 'mse', 'mae'가 존재하며 일반적으로 defalut값인 friedman_mse가 기능이 좋다.
    """

    def __init__(self, data):
        self.model = GradientBoostingRegressor(random_state=42)
        self.log = logger.get_log()
        super().__init__(self.log, self.model, data)

    @dispatch(int)
    def set_model(self, n_iter):
        self.model = GradientBoostingRegressor(n_estimators=n_iter, random_state=42)

    @dispatch(float)
    def set_model(self, learning_rate):
        self.model = GradientBoostingRegressor(learning_rate=learning_rate, random_state=42)

    @dispatch(str)
    def set_model(self, loss):
        self.model = GradientBoostingRegressor(loss=loss, random_state=42)

    @dispatch(int, float)
    def set_model(self, n_iter, learning_rate):
        self.model = GradientBoostingRegressor(n_estimators=n_iter, learning_rate=learning_rate, random_state=42)

    @dispatch(str, int, float)
    def set_model(self, loss, n_iter, learning_rate):
        self.model = GradientBoostingRegressor(n_estimators=n_iter, learning_rate=learning_rate, loss=loss, random_state=42)
