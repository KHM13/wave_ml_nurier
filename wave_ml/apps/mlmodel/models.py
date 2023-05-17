from django.db import models
from wave_ml.apps.project.models import Project


# Create your models here.
class MlModel(models.Model):
    project_id = models.ForeignKey(Project, related_name="project", on_delete=models.CASCADE),
    model_name = models.TextField(blank=False),
    best_accuracy = models.FloatField(),
    best_recall = models.FloatField(),
    create_date = models.DateTimeField(auto_now_add=True)