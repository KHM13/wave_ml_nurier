from django.urls import path
from . import views

urlpatterns = [
    # 프로젝트 메인
    path('', views.main, name="project-list"),
    # 프로젝트 등록, 편집
    path('registration/', views.registration, name="project-registration"),
    # 프로젝트 삭제
    path('remove/', views.remove, name="project-remove"),
    # 프로젝트 복제
    path('clone/', views.clone, name="project-clone"),
    # 프로젝트 상세
    path('detail/', views.detail, name="project-detail"),
]