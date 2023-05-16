from django.urls import path
from . import views

urlpatterns = [
    # 프로젝트 메인
    path('', views.main, name="project-main"),
    # 프로젝트 리스트
    path('list/', views.list, name="project-list"),
    # 프로젝트 등록
    path('registration/', views.registration, name="project-registration"),
    # 아이디로 특정 프로젝트 세부정보 가져오기
    path('project_get_detail/', views.project_get_detail, name="project-project-get"),
    # 프로젝트 편집
    path('modify/', views.modify, name="project-modify"),
    # 프로젝트 복제
    path('clone/', views.clone, name="project-clone"),
    # 프로젝트 삭제
    path('remove/', views.remove, name="project-remove"),
    # 프로젝트 상세
    path('detail/', views.detail, name="project-detail"),
]