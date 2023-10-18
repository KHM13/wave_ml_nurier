from multipledispatch import dispatch
from sklearn.ensemble import RandomForestClassifier

from wave_ml.ml.model.old.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}{util().now_type3()}_FDSRandomForestClassifier", "a", "DEBUG")


class FDSRandomForestClassifier(FDSModel):

    """
    n_estimators ( default : 10 )
                - 사용되는 Decision Tree의 갯수를 지정
                - 무작정 트리 갯수를 늘리면 성능 좋아지는 것 대비 시간이 걸릴 수 있음

    min_samples_split ( default : 2 )
                - 노드를 분할하기 위한 최소한의 샘플 데이터수
                → 과적합을 제어하는데 사용합니다. 값이 작을수록 분할노드가 많아져 과적합 가능성 증가

    min_samples_leaf ( default : 1 )
                - 리프노드가 되기 위한 최소한의 샘플 데이터수
                → 과적합을 제어하는데 사용합니다. 값이 작을수록 과적합 가능성 증가
                - 불균형 데이터의 경우 특정 클래스 데이터가 극도로 적을 수 있으므로 작은 값으로 설정 필요

    max_features ( default : auto (Decision Tree에서는 default가 None인 것과 차이점) )
                - 최적의 분할을 위해 고려할 피처의 최대 갯수
                - int형으로 지정 → 피처 갯수
                - float형으로 지정 → 전체 갯수의 일정 비율만큼 사용
                - sqrt 또는 auto → 전체 피처 중 √(피처 개수) 만큼 선정
                - log2 : 전체 피처 중 log2(전체 피처 개수) 만큼 선정

    max_depth ( default : None → 완벽하게 클래스 값이 결정될 때까지 분할 또는 데이터 갯수가 min_samples_split보다 작아질 때까지 분할 )
                - 트리의 최대 깊이
                - 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요

    max_leaf_nodes ( default : None )
                - 리프노드의 최대 갯수
    """

    def __init__(self, data):
        self.model = RandomForestClassifier(random_state=42)
        self.log = logger.get_log()
        super().__init__(self.log, self.model, data)

    @dispatch(int)
    def set_model(self, n_estimators):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    @dispatch(int, int)
    def set_model(self, min_samples_split, min_samples_leaf):
        self.model = RandomForestClassifier(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)

    @dispatch(int, int, int)
    def set_model(self, n_estimators, min_samples_split, min_samples_leaf):
        self.model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)

    @dispatch(int, int, int, int)
    def set_model(self, n_estimators, min_samples_split, min_samples_leaf, max_depth):
        self.model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_depth=max_depth, random_state=42)
