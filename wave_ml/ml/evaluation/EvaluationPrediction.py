from wave_ml.ml.model.FDSModel import FDSModel

from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.ml.tuning import TrainValidationSplit, TrainValidationSplitModel
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import DecisionTreeClassificationModel, RandomForestClassificationModel


class EvaluationPrediction:

    r2: float
    mse: float
    rmse: float
    mae: float

    def __init__(self, model: FDSModel, label: str):
        self.model = model
        self.title = model.model_title
        self.label = label

        self.prediction = model.get_prediction()
        self.prediction_result = model.prediction_result(label)

        self.prediction_result = self.prediction_result.withColumn(self.label, col(self.label).cast(FloatType()))
        self.multiclass_metrics = MulticlassMetrics(self.prediction_result.rdd.map(tuple))
        self.binaryclass_metrics = BinaryClassificationMetrics(self.prediction_result.rdd.map(tuple))

        evaluator = RegressionEvaluator(predictionCol=self.prediction_result.columns.__getitem__(0), labelCol=self.prediction_result.columns.__getitem__(1))
        try:
            self.r2 = evaluator.setMetricName("r2").evaluate(self.prediction_result)
        except Exception as e:
            print(f"{self.__class__} R2 ERROR : {e}")
            model.log.debug("EvaluationPrediction [ R2 ] ERROR")

        try:
            self.mse = evaluator.setMetricName("mse").evaluate(self.prediction_result)
        except Exception as e:
            print(f"{self.__class__} MSE ERROR : {e}")
            model.log.debug("EvaluationPrediction [ MSE ] ERROR")

        try:
            self.rmse = evaluator.setMetricName("rmse").evaluate(self.prediction_result)
        except Exception as e:
            print(f"{self.__class__} RMSE ERROR : {e}")
            model.log.debug("EvaluationPrediction [ RMSE ] ERROR")

        try:
            self.mae = evaluator.setMetricName("mae").evaluate(self.prediction_result)
        except Exception as e:
            print(f"{self.__class__} MAE ERROR : {e}")
            model.log.debug("EvaluationPrediction [ MAE ] ERROR")

    def print_evaluation_model(self):
        for evaluation in self.evaluation_model_result():
            self.model.log.debug(evaluation)

    def evaluation_model_result(self) -> list:
        result = []
        result.append("##############################################")
        result.append("###")
        result.append(f"###              {self.title}")
        result.append("###")
        result.append("##############################################")
        result.append("")
        result.append("")
        result.append("### Prediction Result DataFrame")

        for predictions_row in self.prediction._jdf.showString(20, 50, False).split("\n"):
            result.append(predictions_row)
        result.append("")
        result.append("### Prediction Result DataFrame ( prediction : 1 , label : 0 )")

        for predictions_row in self.prediction.filter("prediction==1.0").filter(f"{self.label}==0.0")._jdf.showString(20, 50, False).split("\n"):
            result.append(predictions_row)

        result.append("")
        result.append("### Evaluation")
        result.append(f"R2 : {self.r2}")
        result.append(f"MSE : {self.mse}")
        result.append(f"RMSE : {self.rmse}")
        result.append(f"MAE : {self.mae}")
        result.append("")

        result.append("Confusion matrix :")

        for split_value in self.multiclass_metrics.confusionMatrix().toArray():
            result.append(f"             {split_value}")
        result.append("")
        result.append(f"Accuracy = {self.multiclass_metrics.accuracy}")
        result.append("")

        result.append(f"Weighted Precision : {self.multiclass_metrics.weightedPrecision}")
        result.append(f"Weighted Recall : {self.multiclass_metrics.weightedRecall}")
        result.append(f"Weighted F1 score : {self.multiclass_metrics.weightedFMeasure()}")
        result.append(f"Weighted False Positive Rate : {self.multiclass_metrics.weightedFalsePositiveRate}")
        result.append("")

        result.append(f"Area Under ROC - AUC    : {self.binaryclass_metrics.areaUnderROC}")
        result.append(f"Area Under PR           : {self.binaryclass_metrics.areaUnderPR}")

        if type(self.model.get_model_result().stages.__getitem__(len(self.model.get_model_result().stages)-1)) is TrainValidationSplit:
            result_model: TrainValidationSplitModel = self.model.get_model_result().stages.__getitem__(len(self.model.get_model_result().stages)-1)
            result.append("")
            result.append("### Best Model Param")

            for split_value in result_model.bestModel.extractParamMap():
                result.append(split_value)
            result.append("")

            if type(result_model.bestModel) is DecisionTreeClassificationModel:
                for split_value in (result_model.bestModel).__str__().split("\n"):
                    result.append(split_value)

            if type(result_model.bestModel) is RandomForestClassificationModel:
                for split_value in (result_model.bestModel).__str__().split("\n"):
                    result.append(split_value)
        return result

    def get_evaluation_result(self):
        # best_param = {}
        best_param = self.model.pipeline.explainParams()

        result = {
            "r2": self.r2,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "confusion_matrix": self.multiclass_metrics.confusionMatrix().toArray(),
            "accuracy": self.multiclass_metrics.accuracy,
            "recall": self.multiclass_metrics.weightedRecall,
            "f1": self.multiclass_metrics.weightedFMeasure(),
            "precision": self.multiclass_metrics.weightedPrecision,
            "best_params": best_param
        }

        return result