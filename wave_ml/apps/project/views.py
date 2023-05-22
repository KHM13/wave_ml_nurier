import json

from django.core.paginator import Paginator
from django.db.models import Q
from django.http import HttpResponseRedirect, HttpResponse, HttpRequest, JsonResponse
from django.shortcuts import render, redirect

from .models import Project, ProjectFile


# 프로젝트 메인 첫페이지
def main(request):
    sort = request.session.get('sort', 'update')
    page = request.session.get('page', 1)
    keyword = request.session.get('keyword', "")
    page = validate_page(page)

    return render(
        request,
        'project/project-main.html',
        {
            'sort': sort,
            'page': page,
            'keyword': keyword
        }
    )

# 프로젝트 리스트 호출
def list(request):
    if request.method == 'POST':
        # 추후 수정 필요
        user_id = 'TEST_USER'
        
        # 기존 조건 확인
        sort = request.POST.get('sort', request.session.get('sort'))
        page = request.POST.get('page', request.session.get('page'))
        keyword = request.POST.get('keyword', request.session.get('keyword'))
        page = validate_page(page)

        # 리스트 정렬 조회
        project_list = list_sort(sort, user_id, keyword)
        
        # 페이징 처리
        project_obj, links = paginator(page, project_list)
        
        # 세션 처리
        request.session['sort'] = sort
        request.session['page'] = page
        request.session['keyword'] = keyword
        
        # 페이지 이동
        return render(
            request,
            'project/project-list.html',
            {
                'page_links': links,
                'project_obj': project_obj,
                'sort': sort,
                'page': page,
                'keyword': keyword
            }
        )

# 프로젝트 리스트 정렬 조회
def list_sort(sort, user_id, keyword):
    if keyword == "" or keyword == None:
        if sort == 'name':
            project_list = Project.objects.filter(registrant=user_id).order_by('project_name')
        elif sort == 'registration':
            project_list = Project.objects.filter(registrant=user_id).order_by('-project_registration_date')
        else:
            project_list = Project.objects.filter(registrant=user_id).order_by('-project_update_date')
    else:
        if sort == 'name':
            project_list = Project.objects.filter(Q(registrant=user_id) & Q(project_name__contains=keyword)).order_by('project_name')
        elif sort == 'registration':
            project_list = Project.objects.filter(Q(registrant=user_id) & Q(project_name__contains=keyword)).order_by('-project_registration_date')
        else:
            project_list = Project.objects.filter(Q(registrant=user_id) & Q(project_name__contains=keyword)).order_by('-project_update_date')

    return project_list

# 페이징 validation
def validate_page(page):
    if page is None:
        page = 1
    elif page == "":
        page = 1
    elif int(page) <= 0:
        page = 1

    return page

# 페이징 처리
def paginator(page, project_list):
    page = validate_page(page)
    # 한 페이지에 보여줄 항목 수 지정
    projects_per_page = 8
    # 페이징 처리를 위한 Paginator 객체 생성
    paginator = Paginator(project_list, projects_per_page)
    # 현재 페이지에 해당하는 프로젝트 리스트 반환
    project_obj = paginator.get_page(page)
    # 페이지 버튼의 범위 제한
    if project_obj.number <= 3:
        page_btn_range = range(1, min(6, paginator.num_pages + 1))
    else:
        if project_obj.number <= paginator.num_pages - 2:
            page_btn_range = range(max(project_obj.number - 2, 1), min(project_obj.number + 3, paginator.num_pages + 1))
        else:
            page_btn_range = range(max(paginator.num_pages - 4, 1), paginator.num_pages + 1)
    # 페이지 버튼 생성
    links = []
    for pr in page_btn_range:
        if int(pr) == int(page):
            links.append(
                '<li class="page-item active"><a href="javascript:void(0)" class="page-link">%d</a></li>' % pr)
        else:
            links.append(
                '<li class="page-item"><a href="javascript:go_page(%d)" class="page-link">%d</a></li>' % (pr, pr))

    return project_obj, links

# 아이디로 특정 프로젝트 세부정보 가져오기
def project_get_detail(request):
    if request.method == 'POST':
        project_obj = Project.objects.get(id=request.POST.get('project_id'))
        project_files = project_obj.project.all()

        project_image = ""
        if project_obj.project_image:
            project_image = project_obj.project_image.url

        file_name = []
        file_size = []
        if project_files:
            for project_file in project_files:
                file_name.append(project_file.project_file_name)
                file_size.append(project_file.project_file_size)

        project_registration_date = str(project_obj.project_registration_date).split()[0]

        data = {
            'project_id': request.POST.get('project_id'),
            'project_type': project_obj.project_type,
            'project_sub_type': project_obj.project_sub_type,
            'project_name': project_obj.project_name,
            'registrant': project_obj.registrant,
            'project_registration_date': project_registration_date,
            'project_explanation': project_obj.project_explanation,
            'project_image': project_image,
            'project_file_name': file_name,
            'project_file_size': file_size
        }
        data_json = json.dumps(data)

        return HttpResponse(data_json, content_type='application/json')

# 프로젝트 등록
def registration(request):
    if request.method == 'POST':
        # 이미지
        project_img = ""
        if request.FILES.get('project_image') is not None:
            project_img = request.FILES.get('project_image');
        # 프로젝트 모델 생성
        new_project = Project.objects.create(
            project_type=request.POST.get('project_type'),
            project_sub_type=request.POST.get('project_sub_type'),
            project_name=request.POST.get('project_name'),
            # 추후 수정 필요
            registrant='TEST_USER',
            project_explanation=request.POST.get('project_explanation'),
            project_image=project_img
        )
        # 파일
        if request.FILES.getlist('project-file-list') is not None:
            project_files = request.FILES.getlist('project-file-list')
            project_files_size = request.POST.get('project_file_size').split(',')
            project_files_name = request.POST.get('project_file_name').split(',')
            for i in range(len(project_files)):
                ProjectFile.objects.create(
                    project_id=new_project,
                    project_file=project_files[i],
                    project_file_name=project_files_name[i],
                    project_file_size=project_files_size[i]
                )

        return list(request)

# 프로젝트 편집
def modify(request):
    if request.method == 'POST':
        modify_project = Project.objects.get(id=request.POST.get('project_modify_id'))
        # 이미지 유효성 검사
        project_img = ""
        if request.FILES.get('project_image') is not None:
            project_img = request.FILES.get('project_image')
        else:
            if request.POST.get('project_modify_img_check') != "":
                project_img = modify_project.project_image

        modify_project.project_type = request.POST.get('project_type')
        modify_project.project_sub_type = request.POST.get('project_sub_type')
        modify_project.project_name = request.POST.get('project_name')
        modify_project.project_explanation = request.POST.get('project_explanation')
        modify_project.project_image = project_img
        modify_project.save()

        modify_checker = request.POST.get("project_modify_position_check")

        if modify_checker == 'list':
            return list(request)
        else:
            return detail(request)

# 프로젝트 복제
def clone(request):
    if request.method == 'POST':
        # 새 프로젝트 생성
        old_project = Project.objects.get(id=request.POST.get('project_id'))
        new_project = Project.objects.create(
            project_type=old_project.project_type,
            project_sub_type=old_project.project_sub_type,
            project_name=old_project.project_name + "_복제본",
            registrant=old_project.registrant,
            project_explanation=old_project.project_explanation,
            project_image=old_project.project_image
        )
        # 새 파일 생성
        old_files = ProjectFile.objects.filter(project_id=old_project)
        if old_files is not None:
            for old_file in old_files:
                new_file = ProjectFile.objects.create(
                    project_id=new_project,
                    project_file=old_file.project_file,
                    project_file_name=old_file.project_file_name,
                    project_file_size=old_file.project_file_size
                )

        return list(request)

# 프로젝트 삭제
def remove(request):
    if request.method == 'POST':
        Project.objects.get(id=request.POST.get('project_id')).delete()

        return list(request)

# 프로젝트 상세조회
def detail(request):
    if request.method == 'POST':
        project_obj = Project.objects.get(id=int(request.POST.get('project_modify_id')))
    else:
        project_obj = Project.objects.get(id=request.GET.get('project_id'))

    project_image = ""
    if project_obj.project_image:
        project_image = project_obj.project_image

    return render(
        request,
        'project/project-detail.html',
        {
            'project_id': request.GET.get('project_id'),
            'project_type': project_obj.project_type,
            'project_sub_type': project_obj.project_sub_type,
            'project_name': project_obj.project_name,
            'registrant': project_obj.registrant,
            'project_registration_date': project_obj.project_registration_date,
            'project_update_date': project_obj.project_update_date,
            'project_explanation': project_obj.project_explanation,
            'project_image': project_image
        }
    )