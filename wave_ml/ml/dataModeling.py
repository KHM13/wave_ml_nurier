import numpy as np

from wave_ml.ml.data.SparkDataObject import SparkDataObject
from wave_ml.ml.evaluation.EvaluationPrediction import EvaluationPrediction
from wave_ml.ml.model import FDSLinearRegression, FDSLogisticRegression, FDSLinearSVC, FDSGradientBoostingRegressor,\
                            FDSDecisionTreeClassifier, FDSRandomForestClassifier, FDSMultilayerPerceptronClassifier,\
                            FDSNaiveBayes, FDSRandomForestRegressor, FDSDecisionTreeRegressor


def analyze_model(sparkDataObject, algorithm, hyper_parameters, train_size):
    max_iter = []
    elasticnet_param = []
    reg_param = []
    max_depth = []
    min_samples_split = []
    n_estimators = []
    solver = ""
    min_samples_leaf = []
    block_size = ""
    penalty = []

    for parameter in hyper_parameters:
        parameter_name = parameter['name']
        value = parameter['value']

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
        else:
            if parameter_name == "block size":
                block_size = int(value)
            elif parameter_name == "solver":
                solver = str(value)

    if algorithm.__eq__("Linear Regression"):
        fds_model = execute_LinearRegression(sparkDataObject, train_size, max_iter, reg_param, elasticnet_param)
    elif algorithm.__eq__("Decision Tree Regression"):
        fds_model = execute_DecisionTreeRegression(sparkDataObject, train_size, max_depth, min_samples_split)
    elif algorithm.__eq__("Random Forest Regression"):
        fds_model = execute_RandomForestRegression(sparkDataObject, train_size, max_depth, n_estimators)
    elif algorithm.__eq__("Logistic Regression"):
        fds_model = execute_LogisticRegression(sparkDataObject, train_size, max_iter, reg_param, elasticnet_param)
    elif algorithm.__eq__("LinearSVC"):
        fds_model = execute_LinearSVC(sparkDataObject, train_size, max_iter, reg_param)
    elif algorithm.__eq__("Naïve Bayes"):
        fds_model = execute_NaiveBayes(sparkDataObject)
    elif algorithm.__eq__("Decision Tree Classifier"):
        fds_model = execute_DecisionTreeClassifier(sparkDataObject, train_size, max_depth, min_samples_split)
    elif algorithm.__eq__("Random Forest Classifier"):
        fds_model = execute_RandomForestClassifier(sparkDataObject, train_size, max_depth, n_estimators)
    elif algorithm.__eq__("Multilayer Perceptron Classifier"):
        fds_model = execute_MultilayerPerceptronClassifier(sparkDataObject, block_size, solver)
    elif algorithm.__eq__("Gradient Boosted Tree Classifier"):
        fds_model = execute_GradientBoostedTreeClassifier(sparkDataObject, train_size, max_depth, max_iter)
    else:
        fds_model = execute_RandomForestClassifier(sparkDataObject, train_size, max_depth, n_estimators)

    evaluation = EvaluationPrediction(fds_model, sparkDataObject.get_label_column())
    result = evaluation.print_evaluation_model()
    fds_model.log.debug(result)
    best_result = evaluation.get_evaluation_result()
    best_result['model_name'] = algorithm

    return best_result


def execute_LinearRegression(data: SparkDataObject, train_size: float, max_iter: list, reg_param: list, elasticnet_param: list):
    fds_model = FDSLinearRegression.FDSLinearRegression(data).create().set_model(train_size, data.get_label_column(), max_iter, reg_param, elasticnet_param, 0.5)
    fds_model.training()
    fds_model.predicting()
    fds_model.prediction_transform_bucketizer()
    return fds_model


def execute_LogisticRegression(data: SparkDataObject, train_size: float, max_iter: list, reg_param: list, elasticnet_param: list):
    fds_model = FDSLogisticRegression.FDSLogisticRegression(data).create().set_model(train_size, data.get_label_column(), max_iter, reg_param, elasticnet_param)
    fds_model.training()
    fds_model.predicting()
    return fds_model


def execute_DecisionTreeClassifier(data: SparkDataObject, train_size: float, max_depth: list, min_samples_split: list):
    fds_model = FDSDecisionTreeClassifier.FDSDecisionTreeClassifier(data).create().set_model(train_size, data.get_label_column(), max_depth, min_samples_split)
    fds_model.training()
    fds_model.predicting()
    return fds_model


def execute_DecisionTreeRegression(data: SparkDataObject, train_size: float, max_depth: list, min_samples_split: list):
    fds_model = FDSDecisionTreeRegressor.FDSDecisionTreeRegressor(data).create().set_model(train_size, data.get_label_column(), max_depth, min_samples_split)
    fds_model.training()
    fds_model.predicting()
    return fds_model


def execute_LinearSVC(data: SparkDataObject, train_size: float, max_iter: list, reg_param: list):
    fds_model = FDSLinearSVC.FDSLinearSVC(data).create().set_model(train_size, data.get_label_column(), max_iter, reg_param)
    fds_model.training()
    fds_model.predicting()
    return fds_model


def execute_NaiveBayes(data: SparkDataObject):
    fds_model = FDSNaiveBayes.FDSNaiveBayes(data).create().set_model(data.get_label_column())
    fds_model.training()
    fds_model.predicting()
    return fds_model


def execute_RandomForestClassifier(data: SparkDataObject, train_size: float, max_depth: list, n_estimators: list):
    fds_model = FDSRandomForestClassifier.FDSRandomForestClassifier(data).create().set_model(train_size, data.get_label_column(), max_depth, n_estimators)
    fds_model.training()
    fds_model.predicting()
    return fds_model


def execute_RandomForestRegression(data: SparkDataObject, train_size: float, max_depth: list, n_estimators: list):
    fds_model = FDSRandomForestRegressor.FDSRandomForestRegressor(data).create().set_model(train_size, data.get_label_column(), max_depth, n_estimators)
    fds_model.training()
    fds_model.predicting()
    return fds_model


def execute_GradientBoostedTreeClassifier(data: SparkDataObject, train_size: float, max_depth: list, max_iter: list):
    fds_model = FDSGradientBoostingRegressor.FDSGradientBoostedTreeClassifier(data).create().set_model(train_size, data.get_label_column(), max_depth, max_iter)
    fds_model.training()
    fds_model.predicting()
    return fds_model


def execute_MultilayerPerceptronClassifier(data: SparkDataObject, block_size: int, solver: str):
    fds_model = FDSMultilayerPerceptronClassifier.FDSMultilayerPerceptronClassifier(data).create().set_model(block_size, solver, data.get_label_column())
    fds_model.training()
    fds_model.predicting()
    return fds_model
