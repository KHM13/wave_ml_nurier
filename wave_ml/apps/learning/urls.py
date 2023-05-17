from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name="learning"),
    path('detail', views.detail, name="learning-detail"),
    path('model_save', views.save_model_list, name="model-save"),
]