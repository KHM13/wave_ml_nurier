import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from wave_ml.ml import dataChart


def analyze_model(train_data, test_data, target, algorithm, hyper_parameters):
    max_iter = []
    elasticnet_param = []
    reg_param = []
    max_depth = []
    min_samples_split = []
    n_estimators = []
    criterion = ""
    min_samples_leaf = []
    block_size = ""
    penalty = []
    kernel = ""

    for parameter in hyper_parameters:
        parameter_name = parameter['name']
        value = parameter['value']

        min, max, step = 0, 0, 0
        if "~" in value:
            split_value = value.split(" ~ ")
            min = float(split_value[0])
            split_temp = split_value[1].split(" [")
            max = float(split_temp[0])
            step = float(split_temp[1].replace("]", ""))

        for i in np.arange(min, max, step):
            if parameter_name == "Max iter":
                max_iter.append(int(i))
            elif parameter_name == "Elasticnet param":
                elasticnet_param.append(i)
                if i == 1.0:
                    penalty.append("l1")
                elif i == 0.0:
                    penalty.append("l2")
                else:
                    penalty.append("elasticnet")
            elif parameter_name == "reg param":
                if i > 0.0:
                    reg_param.append(i)
            elif parameter_name == "Max depth":
                max_depth.append(int(i))
            elif parameter_name == "Min samples split":
                min_samples_split.append(int(i))
            elif parameter_name == "Min samples leaf":
                min_samples_leaf.append(int(i))
            elif parameter_name == "N Estimators":
                n_estimators.append(i)

        if parameter_name == "Criterion":
            criterion = value
        elif parameter_name == "block size":
            block_size = value
        elif parameter_name == "kernel":
            kernel = value

    best_result = {}
    train_X = train_data.drop(target, axis=1)
    train_y = train_data[[target]]
    test_X = test_data.drop(target, axis=1)
    test_y = test_data[[target]]

    if algorithm == "Logistic Regression":
        best_result = model_logistic(train_X, test_X, train_y, test_y, max_iter, penalty, reg_param)
    elif algorithm == "LinearSVC":
        best_result = model_svc(train_X, test_X, train_y, test_y, elasticnet_param, reg_param, kernel)
    elif algorithm == "Naïve Bayes":
        best_result = model_naivebayes(train_X, test_X, train_y, test_y, max_depth, min_samples_leaf)
    elif algorithm == "Decision Tree Regression":
        best_result = model_decision_tree(train_X, test_X, train_y, test_y, max_depth, min_samples_split)
    elif algorithm == "Random Forest Regression":
        best_result = model_random_forest(train_X, test_X, train_y, test_y, n_estimators, max_depth, min_samples_leaf, min_samples_split)
    elif algorithm == "Multilayer Perceptron Classifier":
        best_result = model_MultiLayerPerceptron(train_X, test_X, train_y, test_y, max_iter, block_size)
    elif algorithm == "Gradient Boosted Tree Classifier":
        best_result = model_gradient_boosting(train_X, test_X, train_y, test_y, max_iter, max_depth)
    best_result['model_name'] = algorithm

    return best_result


def model_logistic(x_train, x_test, y_train, y_test, max_iter, penalty, reg_param):
    logistic = LogisticRegression()

    parameters = {"C": reg_param if len(reg_param) > 0 and reg_param is not None else np.arange(1.0, 0.1, 0.01),
                  "penalty": list(set(penalty)) if len(penalty) > 0 and penalty is not None else ['l1', 'l2'],
                  "max_iter": max_iter if len(max_iter) > 0 and max_iter is not None else [100, 200, 300, 400, 500]}
    # C : 정규화 매개 변수, 정규화의 강도는 C 에 반비례, 항상 양수 ( 기본 1.0 )
    # penalty : L2( Ridge ) - 일반적으로 사용 ( default ) / L1( Lasso ) - 변수가 많아서 줄여야할 때 사용, 모델의 단순화

    searcher = GridSearchCV(logistic, parameters, cv=5, refit=True)
    # cv : 하나의 파라미터 쌍으로 모델링할 때 train, test 교차검증을 실시하는 횟수
    # refit : 최적의 파라미터를 찾고 그 파라미터로 학습시켜놓음

    result = get_searcher_result(x_train, x_test, y_train, y_test, searcher)

    return result


def model_svc(x_train, x_test, y_train, y_test, elasticnet_param, reg_param, kernel):
    svm = SVC(probability=True)

    parameters = {"C": reg_param if len(reg_param) > 0 and reg_param is not None else np.arange(1, 10, 1),
                  "gamma": list(set(elasticnet_param)) if len(elasticnet_param) > 0 and elasticnet_param is not None else [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
                  "kernel": kernel if kernel is not None and kernel != "" else 'rbf'}
    # C : 정규화 매개 변수, 정규화의 강도는 C 에 반비례, 항상 양수 ( 기본 1.0 )
    # kernel : 알고리즘에서 사용할 커널 유형 지정 ( linear, poly, rbf, sigmoid, precomputed, callable : 기본 rbf )
    # gamma : rbf, poly, sigmoid 에 대한 커널 계수 ( 분류모델에서 사용 )

    searcher = GridSearchCV(svm, parameters, cv=5, refit=True)
    # cv : 하나의 파라미터 쌍으로 모델링할 때 train, test 교차검증을 실시하는 횟수
    # refit : 최적의 파라미터를 찾고 그 파라미터로 학습시켜놓음

    result = get_searcher_result(x_train, x_test, y_train, y_test, searcher)

    return result


def model_naivebayes(x_train, x_test, y_train, y_test, max_depth, min_samples_leaf):
    nb = GaussianNB()

    parameters = {'max_depth': max_depth if len(max_depth) > 0 and max_depth is not None else [5, 10, 15],
                  'min_samples_leaf': min_samples_leaf if len(min_samples_leaf) > 0 and min_samples_leaf is not None else [2, 3, 4, 5]}
    searcher = GridSearchCV(nb, parameters, cv=5, refit=True)

    result = get_searcher_result(x_train, x_test, y_train, y_test, searcher)

    return result


def model_decision_tree(x_train, x_test, y_train, y_test, max_depth, min_samples_split):
    tree = DecisionTreeClassifier()

    parameters = {'max_depth': max_depth if len(max_depth) > 0 and max_depth is not None else [5, 10, 15],
                  'min_samples_split': min_samples_split if len(min_samples_split) > 0 and min_samples_split is not None else [2, 3, 4, 5]}
    # max_depth : 트리의 최대 깊이 ( default=None )
    # min_samples_split : 노드를 분할하기 위한 최소한의 샘플 데이터 수, 과적합 제어

    searcher = GridSearchCV(tree, parameters, cv=5, refit=True)

    result = get_searcher_result(x_train, x_test, y_train, y_test, searcher)

    return result


def model_random_forest(x_train, x_test, y_train, y_test, n_estimators, max_depth, min_samples_leaf, min_samples_split):
    forest = RandomForestClassifier()

    parameters = {'n_estimators': n_estimators if len(n_estimators) > 0 and n_estimators is not None else [10, 15, 20],
                  'max_depth': max_depth if len(max_depth) > 0 and max_depth is not None else [6, 8, 10, 12],
                  'min_samples_leaf': min_samples_leaf if len(min_samples_leaf) > 0 and min_samples_leaf is not None else [8, 12, 18],
                  'min_samples_split': min_samples_split if len(min_samples_split) > 0 and min_samples_split is not None else [8, 16, 20]}
    # n_estimators : 결정트리의 갯수 지정 ( default=10 )
    # max_depth : 트리의 최대 깊이 ( default=None )
    # min_samples_leaf : 리프노드가 되기 위해 필요한 최소한의 샘플 데이터 수, 과적합 제어
    # min_samples_split :  노드를 분할하기 위한 최소한의 샘플 데이터 수, 과적합 제어

    searcher = GridSearchCV(forest, parameters, cv=5, refit=True)

    result = get_searcher_result(x_train, x_test, y_train, y_test, searcher)

    return result


def model_MultiLayerPerceptron(x_train, x_test, y_train, y_test, max_iter, block_size):
    mlp = MLPClassifier(max_iter=max_iter)

    parameters = {'block_size': block_size if block_size != '' and block_size is not None else ""}
    searcher = GridSearchCV(mlp, parameters, cv=5, refit=True)

    result = get_searcher_result(x_train, x_test, y_train, y_test, searcher)

    return result


def model_gradient_boosting(x_train, x_test, y_train, y_test, max_iter, max_depth):
    boosting = GradientBoostingClassifier(max_depth=max_depth)

    parameters = {'n_estimators': max_iter if len(max_iter) > 0 and max_iter is not None else [100, 200, 300, 400, 500],
                  'learning_rate': [0.025, 0.05, 0.075, 0.1]}
    # loss : 경사하강법에서 사용할 cost function 지정 ( default = deviance )
    # n_estimators : 생성할 트리의 갯수를 지정 ( default = 100 )
    # learning_rate : 학습을 진행할 때마다 적용하는 학습률 (0~1) ( default = 0.1 )

    searcher = GridSearchCV(boosting, parameters, cv=5, refit=True)

    result = get_searcher_result(x_train, x_test, y_train, y_test, searcher)

    return result


# 결과 dictionary ( 결과 테이블 출력 데이터 )
def get_searcher_result(x_train, x_test, y_train, y_test, searcher):
    searcher.fit(x_train, y_train)

    best_params = searcher.best_params_
    best_score = searcher.best_score_

    y_pred = searcher.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    score_f1 = f1_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    tn = confusion_mat[0][0]
    fp = confusion_mat[0][1]
    fn = confusion_mat[1][0]
    tp = confusion_mat[1][1]

    result = {"best_score": best_score, "best_params": best_params, "accuracy": accuracy, "recall": recall, "f1": score_f1,
              "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    print(result)

    return result
