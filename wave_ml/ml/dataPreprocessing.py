import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
from pandas import DataFrame
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import train_test_split, StratifiedKFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Normalizer, MaxAbsScaler


# 데이터 타입 변경
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, CrossValidator


def data_type_control(df, column, data_type):
    print(f"column : {column}, data_type : {data_type}")

    if data_type not in df[column].dtypes.name:
        if data_type in "datetime":
            df[column] = pd.to_datetime(df[column])
        elif data_type in "category":
            df[column] = df[column].factorize()[0]
            # df = df.astype({column: data_type})
            df.loc[:, [column]] = df.loc[:, [column]].astype(data_type)
        else:
            # df = df.astype({column: data_type})
            df.loc[:, [column]] = df.loc[:, [column]].astype(data_type)

    print(f"{column} type name : {df[column].dtype.name}")
    return df


# 아웃라이어 탐지
def detect_outlier(df, column):
    if df[column].isnull().sum() > 0:
        df = df.dropna(axis=0)
    Q1 = np.percentile(df[column], 25)
    me = np.percentile(df[column], 50)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    outlier_indices = list(df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index)
    outlier_values = set(df.loc[outlier_indices, column].values)

    result = {'q1': Q1, 'q3': Q3, 'median': me, 'outlier': ",".join(map(str, outlier_values)), 'index': outlier_indices}
    return result


# 결측치 처리
def process_missing(df, column, process, input_value):
    df_data: pd.DataFrame = df.copy()
    if process == "remove":     # 결측치 제거
        df_data = df_data.dropna(subset=[column])
    elif process == "interpolation":    # 보간값으로 결측치 변환
        df_data[column] = df_data[column].interpolate(method='values')
    elif process == "mean":     # 평균값으로 결측치 변환
        df_data[column] = df_data[column].fillna(df_data[column].mean())
    elif process == "input":    # 변환값 직접 입력
        type_name = df_data[column].dtypes.name
        print(f" type name : {type_name}, input value : {input_value}")
        if input_value != "":  # 입력값을 str 타입으로 받기 때문에 형변환 필요
            if "int" in type_name:
                input_value = int(input_value)
            elif "float" in type_name:
                input_value = float(input_value)
            df_data[column] = df_data[column].fillna(input_value)
    return df_data


# 이상치 처리
def process_outlier(df, column, process, input_value):
    df_data = df.copy()
    result = detect_outlier(df_data, column)
    outlier_index_list = result.get("index")
    print(f"outlier index list : {outlier_index_list}")

    df = df.drop(outlier_index_list, axis=0)

    if process == "minmax":   # 최소/최대값으로 이상치 변환
        max = df[column].max()
        min = df[column].min()

        for index in outlier_index_list:
            value = df_data[column].iloc[int(index)]
            if value >= max:
                df_data[column].iloc[index] = max
            elif value <= min:
                df_data[column].iloc[index] = min
    elif process == "mean":     # 평균값으로 이상치 변환
        mean = df_data[column].mean()
        if "int" in df[column].dtype.name:
            mean = int(mean)
        df_data[column].iloc[outlier_index_list] = mean
    elif process == "input":    # 변환값 직접 입력
        type_name = df[column].dtype.name
        if input_value != "":
            if "int" in type_name:
                input_value = int(input_value)
            elif "float" in type_name:
                input_value = float(input_value)
            df_data[column].iloc[outlier_index_list] = input_value
    return df_data


# 문자열 변환
def fs_replace_value(df, column, work_input, replace_input):
    df_data = df.copy()
    df_data = df_data.replace({column: work_input}, replace_input)
    return df_data


# 가변수화
def process_dummy_data(df, column):
    df_data = pd.get_dummies(df, columns=[column])
    return df_data


# 데이터 정규화
def process_scaler(df, column, process):
    df_data = df.copy()

    if process == "standard":
        scaler = StandardScaler()
    elif process == "robust":
        scaler = RobustScaler()
    elif process == "minmax":
        scaler = MinMaxScaler()
    elif process == "normal":
        scaler = Normalizer()
    elif process == "maxabs":
        scaler = MaxAbsScaler()

    df_data[column] = scaler.fit_transform(df_data[[column]])
    return df_data


def process_apply(df, column, process, work, input_value, replace_value):
    df_apply = df.copy()

    if process == 'missing':
        df_apply = process_missing(df, column, work, replace_value)
    elif process == 'outlier':
        df_apply = process_outlier(df, column, work, replace_value)
    elif process == 'datatype':
        df_apply = data_type_control(df, column, work)
    elif process == 'dummy':
        df_apply = process_dummy_data(df, column)
    elif process == 'replace':
        df_apply = fs_replace_value(df, column, input_value, replace_value)
    elif process == 'scaler':
        df_apply = process_scaler(df, column, work)

    return df_apply


def execute_fs(df, target, fs):

    # const ( 상수 ) 추가
    df_data = sm.add_constant(df, has_constant='add')

    x_data = df_data.drop([target], axis=1)
    y_data = df_data[target]

    if fs == "Forward Selection":
        best_model = forward_model(x_data, y_data)
    elif fs == "Backward Elimination":
        best_model = backword_model(x_data, y_data)
    elif fs == "Stepwise Selection":
        best_model = stepwise_model(x_data, y_data)

    print(f'best model : \n{best_model.summary()}')
    print(f"best AIC : {best_model.aic}")
    best_columns = [column for column in list(best_model.params.index) if column != 'const']
    print(f"best columns : {best_columns}")
    best_columns.append(target)

    result = {'name': fs, 'aic': best_model.aic, 'columns': best_columns, 'size': len(best_columns)}
    return result


# AIC 구하기
def processSubset(x, y, feature_set):
    model = sm.OLS(y, x[list(feature_set)])  # modeling
    regr = model.fit()  # 모델학습
    AIC = regr.aic  # 모델의 AIC
    return {"model": regr, "AIC": AIC}


# 전진 선택법(Step=1)
def forward(x, y, predictors):
    # const ( 상수 ) 가 아닌 predictors 에 포함되어있지 않은 변수들 선택
    remaining_predictors = [p for p in x.columns.difference(['const']) if p not in predictors]
    tic = time.time()
    results = []
    for p in remaining_predictors:
        # aic 계산
        results.append(processSubset(x=x, y=y, feature_set=predictors + [p] + ['const']))
    # 데이터프레임으로 변환
    models = pd.DataFrame(results)

    # AIC가 가장 낮은 것을 선택
    best_model = models.loc[models['AIC'].argmin()]  # index
    toc = time.time()
    print(f"Processed {models.shape[0]}, models on {len(predictors) + 1}, predictors in {(toc - tic)}")
    print(f"Selected predictors : {best_model['model'].model.exog_names}, AIC : {best_model[0]}")
    return best_model


# 전진선택법 모델
def forward_model(x, y):
    f_models = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    # 미리 정의된 데이터 변수
    predictors = []
    # 변수1~10개 : 0~9 -> 1~10
    for i in range(1, len(x.columns.difference(['const'])) + 1):
        forward_result = forward(x, y, predictors)
        if i > 1:
            if forward_result['AIC'] > fmodel_before:   # 새 조합으로 구한 aic 보다 이전 조합의 aic가 더 낮으면 for문 종료
                break
        f_models.loc[i] = forward_result
        predictors = f_models.loc[i]["model"].model.exog_names
        fmodel_before = f_models.loc[i]["AIC"]
        predictors = [k for k in predictors if k != 'const']
    toc = time.time()
    print("Total elapesed time : ", (toc - tic), "seconds.")

    return (f_models['model'][len(f_models['model'])])


# 후진제거법
def backward(x, y, predictors):
    tic = time.time()
    results = []
    # 데이터 변수들이 미리정의된 predictors 조합확인
    for combo in itertools.combinations(predictors, len(predictors) - 1):
        results.append(processSubset(x, y, list(combo) + ['const']))
    models = pd.DataFrame(results)
    # 가장 낮은 AIC를 가진 모델을 선택
    best_model = models.loc[models['AIC'].argmin()]
    toc = time.time()
    print("Processed", models.shape[0], "models on", len(predictors) - 1, "predictors in", (toc - tic))
    print("Selected predictors :", best_model['model'].model.exog_names, ' AIC:', best_model[0])
    return best_model


# 후진 제거법 모델
def backword_model(x, y):
    b_models = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    # 미리 정의된 데이터 변수
    predictors = x.columns.difference(['const'])
    model_before = processSubset(x, y, predictors)['AIC']
    while len(predictors) > 1:
        backward_result = backward(x, y, predictors)
        if backward_result['AIC'] > model_before:       # 새 조합으로 구한 aic 보다 이전 조합의 aic가 더 낮으면 for문 종료
            break
        b_models.loc[len(predictors) - 1] = backward_result
        predictors = b_models.loc[len(predictors) - 1]["model"].model.exog_names
        model_before = backward_result["AIC"]
        predictors = [k for k in predictors if k != 'const']

    toc = time.time()
    print("Total elapsed time :", (toc - tic), "seconds.")

    return (b_models["model"].dropna().iloc[0])


# 단계적 선택법
def stepwise_model(x, y):
    step_models = pd.DataFrame(columns=["AIC", "model"])
    tic = time.time()
    predictors = []

    model_before = processSubset(x, y, predictors + ['const'])['AIC']
    for i in range(1, len(x.columns.difference(['const'])) + 1):
        forward_result = forward(x, y, predictors)
        step_models.loc[i] = forward_result
        predictors = step_models.loc[i]["model"].model.exog_names
        predictors = [k for k in predictors if k != 'const']
        backword_result = backward(x, y, predictors)

        if backword_result['AIC'] < forward_result['AIC']:
            step_models.loc[i] = backword_result
            predictors = step_models.loc[i]["model"].model.exog_names
            model_before = step_models.loc[i]["AIC"]
            predictors = [k for k in predictors if k != 'const']

        if step_models.loc[i]["AIC"] > model_before:
            break
        else:
            model_before = step_models.loc[i]["AIC"]
    toc = time.time()
    print("Total elapsed time : ", (toc - tic), "seconds")

    return (step_models['model'][len(step_models['model'])])


# 데이터 분할 (sklearn)
def train_test_data_division(df, target, columns, split_rate):
    x_data = df[columns]
    y_data = df[[target]]
    rate: float = split_rate / 100.0
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=rate, random_state=42)
    print(f"train : x - {x_train.shape}, y - {y_train.shape} | test : x - {x_test.shape}, y - {y_test.shape}")

    train_data: DataFrame = pd.concat([x_train, y_train], axis=1)
    test_data: DataFrame = pd.concat([x_test, y_test], axis=1)

    return train_data, test_data


# 데이터 분할 (sklearn)
def k_fold_cross_validation(df, columns, k_value, target):
    k_fold = StratifiedKFold(n_splits=k_value, shuffle=True, random_state=42)
    x_data = df[columns]
    y_data = df[[target]]

    train_list = []
    test_list = []
    for train_index, test_index in k_fold.split(x_data, y_data):
        train_data = pd.DataFrame(df, index=train_index).reset_index(drop=True)
        test_data = pd.DataFrame(df, index=test_index).reset_index(drop=True)
        print(f"train : {train_data.shape} | test : {test_data.shape}")
        train_list.append(train_data)
        test_list.append(test_data)

    return train_list, test_list


# 데이터 분할 (sklearn)
def shuffle_split(df, columns, split_rate, k_value, target):
    x_data = df[columns]
    y_data = df[[target]]
    rate: float = split_rate / 100.0

    train_list = []
    test_list = []
    ss = ShuffleSplit(n_splits=k_value, train_size=rate, random_state=42)
    for train_index, test_index in ss.split(x_data, y_data):
        train_data = pd.DataFrame(df, index=train_index).reset_index(drop=True)
        test_data = pd.DataFrame(df, index=test_index).reset_index(drop=True)
        print(f"train : {train_data.shape} | test : {test_data.shape}")
        train_list.append(train_data)
        test_list.append(test_data)

    return train_list, test_list


# 데이터 분할 (spark)
def train_test_data_split(df, split_rate, seed):
    rate: float = split_rate / 100.0
    splits: list = [rate, 1 - rate]
    split_data = df.randomSplit(weights=splits, seed=seed)
    train_data = split_data[0]
    test_data = split_data[1]

    return train_data, test_data


# 데이터 평가 분할 (spark)
def get_train_validation_split(model, param_grid, target, split_rate, k):
    rate: float = split_rate / 100.0
    train_validation_split = TrainValidationSplit().setEstimator(model)\
                                .setEvaluator(RegressionEvaluator(labelCol=target))\
                                .setEstimatorParamMaps(param_grid)\
                                .setTrainRatio(rate)\
                                .setParallelism(k)

    return train_validation_split


# 데이터 분할 (spark)
def get_cross_validator(model, param_grid, target, k):
    cross_validator = CrossValidator().setEstimator(model)\
                        .setEvaluator(RegressionEvaluator(labelCol=target))\
                        .setEstimatorParamMaps(param_grid)\
                        .setParallelism(k)

    return cross_validator


# 불균형 데이터 보정
def set_imbalanced_data(algorithm, train_data, target):
    train_X = train_data.drop([target], axis=1)
    train_y = train_data[target]

    print(f"train y : {train_y.value_counts()}")

    if algorithm == "RandomUnderSampling":
        imbalanced_data = RandomUnderSampler(random_state=42, sampling_strategy='majority')
    elif algorithm == "RandomOverSampling":
        imbalanced_data = RandomOverSampler(random_state=0, sampling_strategy='minority')
    elif algorithm == "SMOTE":
        imbalanced_data = SMOTE(random_state=0)
    elif algorithm == "SMOTEENN":
        imbalanced_data = SMOTEENN(random_state=0)
    elif algorithm == "SMOTETOMEK":
        imbalanced_data = SMOTETomek(random_state=42)

    ib_x, ib_y = imbalanced_data.fit_resample(train_X, train_y)
    df_balanced_data = ib_x.copy()
    df_balanced_data[target] = ib_y

    return df_balanced_data
