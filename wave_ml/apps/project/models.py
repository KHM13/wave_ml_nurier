from django.db import models


# Create your models here.
class Project(models.Model):
    project_id = models.CharField(blank=False, max_length=30, primary_key=True)
    project_name = models.TextField(blank=False)
    project_type = models.TextField(blank=False)
    project_sub_type = models.TextField(blank=False)
    registration_date = models.DateTimeField(auto_now_add=True)
    registrant = models.TextField(blank=False)
    update_date = models.DateTimeField(auto_now=True)
    image = models.ImageField(upload_to='static/projectimages', default="")
    train_file = models.FileField()
    test_file = models.FileField()
    project_detail = models.TextField()
