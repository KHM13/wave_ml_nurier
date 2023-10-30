from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name="evaluation"),
    path('detail', views.detail, name="evaluation-detail"),
]