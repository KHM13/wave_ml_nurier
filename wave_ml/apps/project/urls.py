from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name="project-list"),
    path('detail', views.detail, name="project-detail"),
]