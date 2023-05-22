from django.db import models
from wave_ml.apps.project.models import Project


class MlModel(models.Model):
    model_name = models.TextField(blank=False)
    best_accuracy = models.FloatField(blank=True, null=True)
    best_recall = models.FloatField(blank=True, null=True)
    create_date = models.DateTimeField(auto_now_add=True)
    project_id = models.ForeignKey(Project, related_name="project_id", on_delete=models.CASCADE, to_field="id", db_column="project_id")
