from multipledispatch import dispatch
from sklearn.tree import DecisionTreeClassifier

from wave_ml.ml.model.old.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}{util().now_type3()}_FDSDecisionTreeClassifier", "a", "DEBUG")


class FDSDecisionTreeClassifier(FDSModel):
    def __init__(self, data):
        self.model = DecisionTreeClassifier(random_state=42)
        self.log = logger.get_log()
        super().__init__(self.log, self.model, data)

    """
    min_samples_split	- 노드를 분할하기 위한 최소한의 샘플 데이터수 → 과적합을 제어하는데 사용
                        - Default = 2 → 작게 설정할 수록 분할 노드가 많아져 과적합 가능성 증가
                        
    min_samples_leaf	- 리프노드가 되기 위해 필요한 최소한의 샘플 데이터수
                        - min_samples_split과 함께 과적합 제어 용도
                        - 불균형 데이터의 경우 특정 클래스의 데이터가 극도로 작을 수 있으므로 작게 설정 필요
                        
    max_features	    - 최적의 분할을 위해 고려할 최대 feature 개수
                        - Default = None → 데이터 세트의 모든 피처를 사용
                        - int형으로 지정 →피처 갯수 / float형으로 지정 →비중
                        - sqrt 또는 auto : 전체 피처 중 √(피처개수) 만큼 선정
                        - log : 전체 피처 중 log2(전체 피처 개수) 만큼 선정
                        
    max_depth	        - 트리의 최대 깊이
                        - default = None
                        → 완벽하게 클래스 값이 결정될 때 까지 분할
                        또는 데이터 개수가 min_samples_split보다 작아질 때까지 분할
                        - 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요
                        
    max_leaf_nodes	    - 리프노드의 최대 개수
    """

    @dispatch(int)
    def set_model(self, min_samples_split):
        self.model = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=42)

    @dispatch(float)
    def set_model(self, min_samples_leaf):
        self.model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=42)

    @dispatch(int, float)
    def set_model(self, min_samples_split, min_samples_leaf):
        self.model = DecisionTreeClassifier(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
