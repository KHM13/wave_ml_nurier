from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.shortcuts import render
from .models import Project


# 프로젝트 메인 첫페이지
def main(request):
    # 사용자 정보 임시 하드코딩
    user_name = "김소망"
    user_id = "somangk"

    # DB에 저장된 프로젝트 정보 조회
    project_model = Project.objects.filter(registrant=user_id)
    project_size = len(project_model.values())

    # 페이징 처리
    per_page = 8        # 한 화면에 보여줄 목록 갯수
    paginator = Paginator(project_model, per_page)
    page_num = request.POST.get("page_num", 1)
    try:
        current_page = paginator.page(page_num)
    except PageNotAnInteger:    # 첫 진입시 page_num 1로 설정
        current_page = paginator.page(1)
    except EmptyPage:   # 마지막 페이지
        current_page = paginator.page(paginator.num_pages)
    list_data = current_page.object_list

    max_index = 5   # 페이징 버튼 보여줄 최대 갯수
    if paginator.num_pages <= max_index:    # 전체 페이지 갯수가 5보다 작을 경우
        start_index = 1
        end_index = paginator.num_pages
    else:
        start_index = page_num - 2
        set_num = 0
        if start_index < 1:     # 첫 페이지 번호가 1보다 작을 경우
            set_num = start_index - 1
            start_index = 1
        end_index = page_num + 2 - set_num
        if end_index > paginator.num_pages:     # 마지막 페이지 번호가 end_index 보다 작을 경우
            set_num = end_index - paginator.num_pages
            start_index = start_index - set_num
            end_index = paginator.num_pages
    page_range = range(start_index, end_index + 1)  # 페이징 버튼 범위 조정

    links = []
    for pr in page_range:
        if pr == current_page.number:   # 현재 선택된 페이징 버튼
            links.append('<li class="page-item active"><a href="javascript:void(0);" class="page-link">%d</a></li>' % pr)
        else:
            links.append('<li class="page-item"><a href="javascript:go_page(%d);" class="page-link">%d</a></li>' % (pr, pr))

    return render(
        request,
        'project/project-list.html',
        {
            'user_name': user_name,
            'page_links': links,
            'list_models': list_data,
            'project_size': project_size
        }
    )


# 프로젝트 상세조회
def detail(request):
    # 퍼블화면 임시 이동
    return render(
        request,
        'project/project-detail.html'
    )
