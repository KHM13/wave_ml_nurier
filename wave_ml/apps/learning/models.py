from django.db import models
from wave_ml.apps.mlmodel.models import MlModel


class ModelLearning(models.Model):
    mlmodel_id = models.ForeignKey(MlModel, related_name="mlmodel_id", on_delete=models.CASCADE, to_field="id", db_column="mlmodel_id")
    algorithm = models.TextField(blank=False)
    accuracy = models.FloatField(blank=True, null=True)
    recall = models.FloatField(blank=True, null=True)
    precision = models.FloatField(blank=True, null=True)
    f1_score = models.FloatField(blank=True, null=True)
    r2 = models.FloatField(blank=True, null=True)
    mse = models.FloatField(blank=True, null=True)
    rmse = models.FloatField(blank=True, null=True)
    mae = models.FloatField(blank=True, null=True)
    confusion_matrix = models.TextField(blank=True, null=True, default='{}')
    best_params = models.TextField(blank=True, null=True, default='{}')
    features_importance = models.TextField(blank=True, null=True, default='[]')
    Provability = models.FloatField(blank=True, null=True)
    Prediction = models.FloatField(blank=True, null=True)
    Favorite = models.BooleanField(default=False)
    learning_date = models.TextField(blank=True, null=True, default='')


class MlParameter(models.Model):
    objects = None
    model_learning_id = models.ForeignKey(ModelLearning, related_name="model_learning_id", on_delete=models.CASCADE, to_field="id", db_column="model_learning_id")
    algorithm = models.TextField(blank=False)
    parameter_name = models.TextField(blank=False)
    parameter_value = models.TextField(blank=False)
