import json
import os
import shutil
import urllib
import uuid
import pandas as pd

from django.core.paginator import Paginator
from django.db.models import Q
from django.http import HttpResponse, FileResponse
from django.shortcuts import render

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
        page = validate_page(request.POST.get('page', request.session.get('page')))
        keyword = request.POST.get('keyword', request.session.get('keyword'))

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
    if keyword == "" or keyword is None:
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
    if page == "" or page is None or int(page) <= 0:
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

# 아이디로 프로젝트 세부정보 가져오기
def project_get_detail(request):
    if request.method == 'POST':
        project_obj = Project.objects.get(id=request.POST.get('project_id'))
        project_files = project_obj.project.all()

        project_image = ""
        if project_obj.project_image:
            project_image = project_obj.project_image.url

        file_name = []
        file_size = []
        file_id = []
        if project_files:
            for project_file in project_files:
                file_id.append(project_file.id)
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
            'project_file_id': file_id,
            'project_file_name': file_name,
            'project_file_size': file_size,
            'project_file_cnt': len(file_id)
        }
        data_json = json.dumps(data)

        return HttpResponse(data_json, content_type='application/json')

# 프로젝트 등록
def registration(request):
    if request.method == 'POST':
        # 이미지
        project_img = ""
        if request.FILES.get('project_image'):
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
        if request.FILES.getlist('project-file-list'):
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
        # 이미지 확인
        project_img = ""
        if request.FILES.get('project_image'):
            project_img = request.FILES.get('project_image')
        else:
            if request.POST.get('project_modify_img_check') != "":
                project_img = modify_project.project_image

        # 파일 등록
        if request.FILES.getlist('project-file-list'):
            project_files = request.FILES.getlist('project-file-list')
            project_files_size = request.POST.get('project_file_size').split(',')
            project_files_name = request.POST.get('project_file_name').split(',')
            for i in range(len(project_files)):
                ProjectFile.objects.create(
                    project_id=modify_project,
                    project_file=project_files[i],
                    project_file_name=project_files_name[i],
                    project_file_size=project_files_size[i]
                )

        # 파일 삭제
        if request.POST.get("project_file_id"):
            remove_file_id_list = request.POST.get("project_file_id").split(",")
            if remove_file_id_list:
                for remove_file_id in remove_file_id_list:
                    remove_file = ProjectFile.objects.get(id=remove_file_id)
                    if os.path.isfile(remove_file.project_file.path):
                        os.remove(remove_file.project_file.path)
                        remove_file.delete()

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
        # 프로젝트 복제
        old_project = Project.objects.get(id=request.POST.get('project_id'))
        new_project = Project.objects.create(
            project_type=old_project.project_type,
            project_sub_type=old_project.project_sub_type,
            project_name=old_project.project_name + "_복제본",
            registrant=old_project.registrant,
            project_explanation=old_project.project_explanation,
            project_image=old_project.project_image
        )

        # 파일 복제
        old_files = old_project.project.all()
        new_files = []
        if old_files:
            for old_file in old_files:
                new_file = ProjectFile.objects.create(
                    project_id=new_project,
                    project_file=old_file.project_file,
                    project_file_name=old_file.project_file_name,
                    project_file_size=old_file.project_file_size
                )
                new_files.append(new_file)

        # 파일시스템 데이터 복제
        for old_file, new_file in zip(old_files, new_files):
            old_file_path = old_file.project_file.path
            file_dir, file_name = os.path.split(old_file_path)
            file_name, file_ext = os.path.splitext(file_name)
            unique_id = uuid.uuid4().hex

            file_path_prefix = file_dir.split("\\")[-2] + "/" + file_dir.split("\\")[-1]
            new_file_name = f"{file_name}_복제본_{unique_id}{file_ext}"
            new_file_path = os.path.abspath(f"{file_dir}/{new_file_name}")
            shutil.copy2(old_file_path, new_file_path)
            new_file.project_file = f"{file_path_prefix}/{new_file_name}"
            new_file.save()

        return list(request)

# 프로젝트 삭제
def remove(request):
    if request.method == 'POST':
        project_obj = Project.objects.get(id=request.POST.get('project_id'))
        project_files = project_obj.project.all()

        # media 폴더 파일 삭제
        if project_files:
            for file in project_files:
                if file.project_file:
                    if os.path.isfile(file.project_file.path):
                        os.remove(file.project_file.path)

        # media 폴더 이미지 삭제
        if project_obj.project_image:
            if os.path.isfile(project_obj.project_image.path):
                os.remove(project_obj.project_image.path)

        # 프로젝트 삭제
        project_obj.delete()

        return list(request)

# 프로젝트 상세조회
def detail(request):
    if request.method == 'POST':
        project_obj = Project.objects.get(id=request.POST.get('project_modify_id'))
        request.session['project_id'] = request.POST.get('project_modify_id')
    else:
        project_obj = Project.objects.get(id=request.GET.get('project_id'))
        request.session['project_id'] = request.GET.get('project_id')

    project_image = ""
    if project_obj.project_image:
        project_image = project_obj.project_image

    project_files = project_obj.project.all()
    file_name = []
    file_size = []
    file_size_conversion = []
    file_id = []
    if project_files:
        for project_file in project_files:
            file_id.append(project_file.id)
            file_name.append(project_file.project_file_name)
            file_size.append(project_file.project_file_size)
            file_size_conversion.append(project_file.project_file_size.replace("/", " "))

    file_data = zip(file_name, file_size_conversion, file_id)

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
            'project_image': project_image,
            'project_file_id': file_id,
            'project_file_name': file_name,
            'project_file_size': file_size,
            'project_file_cnt': len(file_id),
            'file_data': file_data
        }
    )

# 프로젝트 상세 엑셀파일 다운로드
def excel_download(request):
    if request.GET.get("file_id"):
        file_id = request.GET.get("file_id")
        file_obj = ProjectFile.objects.get(id=file_id)
        # 파일 절대 경로 반환
        file_path = file_obj.project_file.path

        # 파일 다운로드 응답 생성
        response = FileResponse(open(file_path, 'rb'))

        # 다운로드할 파일의 MIME 타입을 설정
        response['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

        # 한글, 특수문자 인코딩
        encoded_file_name = urllib.parse.quote(file_obj.project_file_name)

        # 다운로드할 파일의 이름 설정
        response['Content-Disposition'] = f'attachment; filename={encoded_file_name}'

        return response