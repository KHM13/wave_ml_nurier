from django.db import models
from wave_ml.apps.mlmodel.models import MlModel


class ModelLearning(models.Model):
    mlmodel_id = models.ForeignKey(MlModel, related_name="mlmodel_id", on_delete=models.CASCADE, to_field="id", db_column="mlmodel_id")
    algorithm = models.TextField(blank=False)
    accuracy = models.FloatField(blank=True, null=True)
    recall = models.FloatField(blank=True, null=True)
    f1_score = models.FloatField(blank=True, null=True)
    confusion_matrix = models.TextField(blank=True, null=True)
    features_importance = models.TextField(blank=True, null=True)
    Provability = models.FloatField(blank=True, null=True)
    Prediction = models.FloatField(blank=True, null=True)
    Favorite = models.BooleanField(default=False)


class MlParameter(models.Model):
    model_learning_id = models.ForeignKey(ModelLearning, related_name="model_learning_id", on_delete=models.CASCADE, to_field="id", db_column="model_learning_id")
    algorithm = models.TextField(blank=False)
    parameter_name = models.TextField(blank=False)
    parameter_value = models.TextField(blank=False)
