from django.core.paginator import Paginator
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from pip._vendor.rich import print

from .models import Project

# 프로젝트 메인 첫페이지
def main(request):
    # 추후에 로그인 구현되었을때 지정 필요 (임시 하드코딩)
    user_id = "TEST_USER1"
    # 정렬 조건 확인
    sort_option = request.POST.get('sort_option')
    if sort_option == None:
        sort_option = 'update'
    # DB에서 정렬 조회
    if sort_option == 'name':
        project_list = Project.objects.filter(registrant=user_id).order_by('-project_name')
    elif sort_option == 'registration':
        project_list = Project.objects.filter(registrant=user_id).order_by('-project_registration_date')
    else:
        project_list = Project.objects.filter(registrant=user_id).order_by('-project_update_date')
    # 한 페이지에 보여줄 항목 수 지정
    projects_per_page = 8
    # 페이징 처리를 위한 Paginator 객체 생성
    paginator = Paginator(project_list, projects_per_page)
    # 요청된 페이지 번호
    page_number = request.GET.get('page', 1)
    # 현재 페이지에 해당하는 프로젝트 리스트 반환
    project_obj = paginator.get_page(page_number)
    # 페이지 버튼의 범위 제한
    if project_obj.number <= 3:
        page_btn_range = range(1, min(6, paginator.num_pages + 1))
    else:
        if project_obj.number <= paginator.num_pages - 2:
            page_btn_range = range(max(project_obj.number - 2, 1), min(project_obj.number + 3, paginator.num_pages + 1))
        else:
            page_btn_range = range(paginator.num_pages - 4, paginator.num_pages + 1)
    # 페이지 버튼 생성
    links = []
    for pr in page_btn_range:
        if pr == int(page_number):
            links.append('<li class="page-item active"><a href="javascript:void(0)" class="page-link">%d</a></li>' % pr)
        else:
            links.append('<li class="page-item"><a href="/project?page=%d" class="page-link">%d</a></li>' % (pr, pr))
    # 템플릿으로 페이지 이동
    return render(
        request,
        'project/project-list.html',
        {
            'page_links': links,
            'project_obj': project_obj,
            'sort_option': sort_option
        }
    )

# 프로젝트 등록, 편집
def registration(request):
    if request.method == 'POST':
        Project.objects.create(
            project_type=request.POST.get('project_type'),
            project_sub_type=request.POST.get('project_sub_type'),
            project_name=request.POST.get('project_name'),
            # 추후에 로그인 구현되었을때 지정 필요
            # registrant=request.POST.get('registrant_name'),
            registrant="TEST_USER1",
            project_explanation=request.POST.get('project_explanation'),
         )
    return HttpResponseRedirect(reverse('project-list'))

# 프로젝트 삭제
def remove(request):
    if request.method == 'POST':
        Project.objects.get(id=request.POST.get('project_id')).delete()
    return HttpResponseRedirect(reverse('project-list'))

# 프로젝트 복제
def clone(request):
    if request.method == 'POST':
        clone_project = Project.objects.get(id=request.POST.get('project_id'))
        clone_project.id = None
        clone_project.project_name = clone_project.project_name + "_복제본"
        clone_project.save()
    return HttpResponseRedirect(reverse('project-list'))

# 프로젝트 상세조회
def detail(request):
    return render(
        request,
        'project/project-detail.html'
    )
