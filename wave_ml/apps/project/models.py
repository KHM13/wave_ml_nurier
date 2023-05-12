from django.db import models

# 프로젝트 모델
class Project(models.Model):
    project_type = models.TextField(blank=False)
    project_sub_type = models.TextField(blank=False)
    project_name = models.TextField(blank=False)
    registrant = models.TextField(blank=False)
    project_registration_date = models.DateTimeField(auto_now_add=True)
    project_update_date = models.DateTimeField(auto_now=True)
    project_explanation = models.TextField()
    project_image = models.ImageField(upload_to='static/upload/image/%Y%m%d', default="")

# 파일 모델
class ProjectFile(models.Model):
    project_id = models.ForeignKey(Project, related_name="project", on_delete=models.CASCADE)
    project_file = models.FileField(upload_to='static/upload/file/%Y%m%d', default="")