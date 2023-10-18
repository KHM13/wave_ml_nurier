import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

from wave_ml.ml.data.DataObject import DataObject
from wave_ml.ml.data.DataStructType import ML_Field


class FDSModel:

    def __init__(self, log, model, data):
        self.log = log
        self.model = model
        self.data: DataObject = data

    def __cut_off(self, y, threshold):
        Y = y.copy()
        Y[Y > threshold] = 1
        Y[Y < threshold] = 0
        return Y.astype(int)

    def get_model(self):
        return self.model

    def training_gridsearch_CV(self, parameter_dict):
        cost_function = GridSearchCV(estimator=self.model, param_grid=parameter_dict, scoring='accuracy')
        train_data = self.data.get_training_data()
        label = ML_Field.get_label()
        train_X = train_data.drop(label, axis=1)
        train_y = train_data[label]
        cost_function.fit(train_X, train_y)

        self.log.debug("")
        self.log.debug(f"Grid Search CV : {self.model.__class__.__name__}")
        self.log.debug(f"best parameters : {cost_function.best_params_}")
        self.log.debug(f"best score : {round(cost_function.best_score_, 5)}")
        self.log.debug(pd.DataFrame(cost_function.cv_results_))
        self.model = cost_function.best_estimator_

        return cost_function

    def training_randomizedsearch_CV(self, parameter_dict, max_iter):
        cost_function = RandomizedSearchCV(estimator=self.model, param_distributions=parameter_dict, n_iter=max_iter, scoring='accuracy')
        train_data = self.data.get_training_data()
        label = ML_Field.get_label()
        train_X = train_data.drop(label, axis=1)
        train_y = train_data[label]
        cost_function.fit(train_X, train_y)

        self.log.debug("")
        self.log.debug(f"Randomized Search CV : {self.model.__class__.__name__}")
        self.log.debug(f"best parameters : {cost_function.best_params_}")
        self.log.debug(f"best score : {round(cost_function.best_score_, 5)}")
        self.log.debug(pd.DataFrame(cost_function.cv_results_))
        self.model = cost_function.best_estimator_

        return cost_function

    def predicting_model(self, is_cut_off: bool):
        test_data = self.data.get_test_data()
        label = ML_Field.get_label()
        test_X = test_data.drop(label, axis=1)
        test_y = test_data[label]
        prediction = self.model.predict(test_X)
        if is_cut_off:
            prediction = self.__cut_off(prediction, 0.5)
        self.__evaluate_result(test_y, prediction)
        return prediction

    def __evaluate_result(self, test_y, prediction):
        # confusion matrix
        cfmat = confusion_matrix(test_y, prediction)
        self.log.debug(f"Confusion Matrix :\n{cfmat}")
        self.log.debug(f"Accuracy Score : {accuracy_score(test_y, prediction)}")
        self.log.debug(f"Classification Report : \n{classification_report(test_y, prediction)}")
        self.log.debug(f"ROC AUC Score : {roc_auc_score(test_y, prediction, average='macro')}")

    def feature_importances(self, count):
        train_data = self.data.get_training_data()
        train_x = train_data.drop(ML_Field.get_label(), axis=1)

        feature_importance_values = self.model.feature_importances_
        feature_importance_series = pd.Series(feature_importance_values, index=train_x.columns)
        self.log.debug(f"변수 중요도 Top {count} :\n{feature_importance_series.sort_values(ascending=False)[:count]}")

        return feature_importance_series

    def feature_importances2(self, count):
        train_data = self.data.get_training_data()
        train_x = train_data.drop(ML_Field.get_label(), axis=1)

        feature_importance_values = self.model.coef_[0]
        feat_importances = pd.Series(feature_importance_values, index=train_x.columns)
        self.log.debug(f"변수 중요도 Top {count} : \n{feat_importances.nlargest(count).sort_values(ascending=False)}")

        return feat_importances
