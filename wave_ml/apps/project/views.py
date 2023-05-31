import io
import json
import os
import shutil
import urllib
import uuid
import zipfile

from django.core.paginator import Paginator
from django.db.models import Q
from django.http import HttpResponse, FileResponse
from django.shortcuts import render

from .models import Project, ProjectFile

# 프로젝트 메인 첫페이지
from ..mlmodel.models import MlModel


def main(request):
    sort = request.session.get('sort', 'update')
    page = validate_page(request.session.get('page', 1))
    keyword = request.session.get('keyword', "")

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

        # 세션 처리
        request.session['sort'] = sort
        request.session['page'] = page
        request.session['keyword'] = keyword

        # 리스트 정렬 조회
        project_list = list_sort(sort, user_id, keyword)

        # 페이징 처리
        project_obj, links = paginator(page, project_list, 8)

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
    if keyword:
        if sort == 'name':
            project_list = Project.objects.filter(Q(registrant=user_id) & Q(project_name__contains=keyword)).order_by('project_name')
        elif sort == 'registration':
            project_list = Project.objects.filter(Q(registrant=user_id) & Q(project_name__contains=keyword)).order_by('-project_registration_date')
        else:
            project_list = Project.objects.filter(Q(registrant=user_id) & Q(project_name__contains=keyword)).order_by('-project_update_date')
    else:
        if sort == 'name':
            project_list = Project.objects.filter(registrant=user_id).order_by('project_name')
        elif sort == 'registration':
            project_list = Project.objects.filter(registrant=user_id).order_by('-project_registration_date')
        else:
            project_list = Project.objects.filter(registrant=user_id).order_by('-project_update_date')

    return project_list

# 페이징 validation
def validate_page(page):
    if page == "" or page is None or int(page) <= 0:
        page = 1
    return page

# 페이징 처리
def paginator(page, project_list, projects_per_page):
    page = validate_page(page)
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

        # 프로젝트
        new_project = Project.objects.create(
            project_type=request.POST.get('project_type'),
            project_sub_type=request.POST.get('project_sub_type'),
            project_name=request.POST.get('project_name'),
            # 추후 수정 필요
            registrant='TEST_USER',
            project_explanation=request.POST.get('project_explanation'),
            project_image=project_img
        )

        # 파일 등록
        if request.FILES.getlist('project-file-list'):
            project_files = request.FILES.getlist('project-file-list')
            project_files_size = request.POST.get('project_file_size').split(',')
            project_files_name = request.POST.get('project_file_name').split(',')
            for i in range(len(project_files)):
                ProjectFile.objects.create(
                    project_id=new_project,
                    project_file=project_files[i],
                    project_file_name=project_files_name[i],
                    project_file_size=project_files_size[i],
                    project_file_extension=project_files_name[i].split('.')[1]
                )

        return list(request)

# 프로젝트 편집
def modify(request):
    if request.method == 'POST':
        modify_project = Project.objects.get(id=request.POST.get('project_id'))
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
                    project_file_size=project_files_size[i],
                    project_file_extension=project_files_name[i].split('.')[1]
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
                    project_file_size=old_file.project_file_size,
                    project_file_extension=old_file.project_file_name.split('.')[1]
                )
                new_files.append(new_file)

        # 모델 복제
        old_mlmodels = old_project.project_id.all()
        if old_mlmodels:
            for old_mlmodel in old_mlmodels:
                MlModel.objects.create(
                    project_id=new_project,
                    model_name=old_mlmodel.model_name,
                    best_accuracy=old_mlmodel.best_accuracy,
                    best_recall=old_mlmodel.best_recall
                )

        # 파일시스템 데이터 복제(이미지)
        old_img_path = old_project.project_image.path
        file_dir_img, file_name_img = os.path.split(old_img_path)
        file_name_img, file_ext_img = os.path.splitext(file_name_img)
        unique_id_img = uuid.uuid4().hex

        img_path_prefix = file_dir_img.split("\\")[-2] + "/" + file_dir_img.split("\\")[-1]
        new_img_name = f"{file_name_img}_복제본_{unique_id_img}{file_ext_img}"
        new_img_path = os.path.abspath(f"{file_dir_img}/{new_img_name}")
        shutil.copy2(old_img_path, new_img_path)
        new_project.project_image = f"{img_path_prefix}/{new_img_name}"
        new_project.save()

        # 파일시스템 데이터 복제(파일)
        for old_file, new_file in zip(old_files, new_files):
            old_file_path = old_file.project_file.path
            file_dir, file_name = os.path.split(old_file_path)
            file_name, file_ext = os.path.splitext(file_name)
            unique_id_file = uuid.uuid4().hex

            file_path_prefix = file_dir.split("\\")[-2] + "/" + file_dir.split("\\")[-1]
            new_file_name = f"{file_name}_복제본_{unique_id_file}{file_ext}"
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

# 프로젝트 단일 엑셀파일 다운로드
def single_excel_file_download(file_id):
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

# 프로젝트 상세 엑셀파일 다운로드
def excel_download(request):
    if request.method == 'GET':
        if request.GET.get("file_id"):
            file_id = request.GET.get("file_id")
            return single_excel_file_download(file_id)

# 프로젝트 상세 엑셀파일 일괄 다운로드
def multiple_excel_download(request):
    if request.method == 'GET':
        if request.GET.get("file_id_list"):
            file_id_list = json.loads(request.GET.get("file_id_list"))
            in_memory_zip = io.BytesIO()
            file_cnt = len(file_id_list) - 1
            encoded_file_name = ""

            if len(file_id_list) > 1:
                with zipfile.ZipFile(in_memory_zip, 'w') as zip_file:
                    for index in range(len(file_id_list)):
                        file_obj = ProjectFile.objects.get(id=file_id_list[index])

                        if int(index) == 0:
                            encoded_file_name = file_obj.project_file_name
                        zip_file.write(file_obj.project_file.path, arcname=file_obj.project_file_name)

                file_name = urllib.parse.quote("{} 외 {}건.zip".format(encoded_file_name, file_cnt))
                in_memory_zip.seek(0)
                response = HttpResponse(content_type='application/zip')
                response['Content-Disposition'] = f'attachment; filename=' + file_name
                response.write(in_memory_zip.read())
                return response
            else:
                return single_excel_file_download(file_id_list[0])

# 엑셀 파일 일괄 삭제
def multiple_excel_remove(request):
    if request.method == 'POST':
        if request.POST.get('file_id_list'):
            file_id_list = json.loads(request.POST.get("file_id_list"))
            for index in range(len(file_id_list)):
                remove_file = ProjectFile.objects.get(id=file_id_list[index])
                if os.path.isfile(remove_file.project_file.path):
                    os.remove(remove_file.project_file.path)
                    remove_file.delete()

        return detail_documents(request)

# 파일 데이터 반환
def get_project_files(project_files):
    file_name = []
    file_size = []
    file_id = []
    file_registration_date = []
    if project_files:
        for project_file in project_files:
            file_id.append(project_file.id)
            file_name.append(project_file.project_file_name)
            file_size.append(project_file.project_file_size)
            file_registration_date.append(project_file.project_file_registration_date)
    file_data = zip(file_name, file_size, file_id, file_registration_date)

    return file_data, file_name, file_size, file_id, file_registration_date

# 프로젝트 상세 -> 메인
def detail(request):
    if request.method == 'GET':
        if request.GET.get('project_id'):
            project_obj = Project.objects.get(id=request.GET.get('project_id'))
            request.session['project_id'] = request.GET.get('project_id')
    else:
        if request.POST.get('project_id'):
            project_obj = Project.objects.get(id=request.POST.get('project_id'))
            request.session['project_id'] = request.POST.get('project_id')
    # 이미지
    project_image = ""
    if project_obj.project_image:
        project_image = project_obj.project_image

    # 파일
    project_files = project_obj.project.all()
    file_data, file_name, file_size, file_id, file_registration_date = get_project_files(project_files)

    # 모델
    best_mlmodel = project_obj.project_id.order_by("-best_recall", "-best_accuracy").first()
    mlmodel = project_obj.project_id.order_by("-best_recall", "-best_accuracy").all()

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
            'file_data': file_data,
            'project_files': project_files,
            'best_mlmodel': best_mlmodel,
            'mlmodel': mlmodel,
        }
    )

# 프로젝트 상세 -> 상세
def detail_main(request):
    if request.method == 'POST':
        # 프로젝트
        user_id = request.session.get('project_id', request.POST.get('project_id'))
        project_obj = Project.objects.get(id=user_id)
        # 파일
        project_files = project_obj.project.all()
        file_data, file_name, file_size, file_id, file_registration_date = get_project_files(project_files)
        # 모델
        best_mlmodel = project_obj.project_id.order_by("-best_recall", "-best_accuracy").first()
        mlmodel = project_obj.project_id.order_by("-best_recall", "-best_accuracy").all()

        return render(
            request,
            'project/project-detail-main.html',
            {
                'project_id': user_id,
                'registrant': project_obj.registrant,
                'project_registration_date': project_obj.project_registration_date,
                'project_update_date': project_obj.project_update_date,
                'project_explanation': project_obj.project_explanation,
                'project_file_id': file_id,
                'project_file_name': file_name,
                'project_file_size': file_size,
                'project_file_cnt': len(file_id),
                'file_data': file_data,
                'best_mlmodel': best_mlmodel,
                'mlmodel': mlmodel,
            }
        )

# 프로젝트 상세 -> 모델
def detail_model(request):
    if request.method == 'POST':
        user_id = request.session.get('project_id', request.POST.get('project_id'))
        project_obj = Project.objects.get(id=user_id)
        mlmodel = project_obj.project_id.order_by("-best_recall", "-best_accuracy").all()
        mlmodel_pagination, links = paginator(request.POST.get("page", 1), mlmodel, 12)

        return render(
            request,
            request.POST.get('render_html', 'project/project-detail-main.html'),
            {
                'project_id': user_id,
                'registrant': project_obj.registrant,
                'project_registration_date': project_obj.project_registration_date,
                'project_update_date': project_obj.project_update_date,
                'project_explanation': project_obj.project_explanation,
                'mlmodel_pagination': mlmodel_pagination,
                'page_links': links
            }
        )

# 프로젝트 상세 -> 문서
def detail_documents(request):
    if request.method == 'POST':
        user_id = request.session.get('project_id', request.POST.get('project_id'))
        project_obj = Project.objects.get(id=user_id)
        project_files = project_obj.project.all()
        file_pagination, links = paginator(request.POST.get("page", 1), project_files, 10)

        return render(
            request,
            request.POST.get('render_html', 'project/project-detail-main.html'),
            {
                'project_id': user_id,
                'registrant': project_obj.registrant,
                'project_registration_date': project_obj.project_registration_date,
                'project_update_date': project_obj.project_update_date,
                'project_explanation': project_obj.project_explanation,
                'file_pagination': file_pagination,
                'page_links': links,
                'project_file_cnt': len(project_files)
            }
        )