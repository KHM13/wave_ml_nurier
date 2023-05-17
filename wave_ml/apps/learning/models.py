from django.db import models
from wave_ml.apps.mlmodel.models import MlModel


# Create your models here.
class ModelLearning(models.Model):
    mlmodel_id = models.ForeignKey(MlModel, related_name="mlmodel", on_delete=models.CASCADE),
    algorithm = models.TextField(blank=False),
    accuracy = models.FloatField(),
    recall = models.FloatField(),
    f1_score = models.FloatField(),
    confusion_matrix = models.TextField(),
    features_importance = models.TextField(),
    Provability = models.FloatField(),
    Prediction = models.FloatField(),
    Favorite = models.BooleanField()


class MlParameter(models.Model):
    modellearning_id = models.ForeignKey(ModelLearning, related_name="modellearning", on_delete=models.CASCADE),
    algorithm = models.ForeignKey(ModelLearning, related_name="algorithm", on_delete=models.CASCADE),
    parameter_name = models.TextField(),
    parameter_value = models.TextField()
