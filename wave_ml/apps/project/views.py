from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.shortcuts import render
from django.views.generic import ListView

from .models import Project

# 프로젝트 메인 첫페이지
def main(request):
    # 사용자 정보 임시 하드코딩
    user_name = "손다니엘"
    user_id = "daniel"
    # 한 페이지에 보여줄 항목 수 지정
    projects_per_page = 8
    # DB에서 데이터 조회
    project_list = Project.objects.filter(registrant=user_id)
    # 페이징 처리를 위한 Paginator 객체 생성
    paginator = Paginator(project_list, projects_per_page)
    # 요청된 페이지 번호
    page_number = request.GET.get('page', 1)
    # 현재 페이지에 해당하는 프로젝트 리스트 반환
    page_obj = paginator.get_page(page_number)
    # 페이지 버튼의 범위 제한
    if page_obj.number <= 3:
        page_btn_range = range(1, min(6, paginator.num_pages + 1))
    else:
        if page_obj.number <= paginator.num_pages - 2:
            page_btn_range = range(max(page_obj.number - 2, 1), min(page_obj.number + 3, paginator.num_pages + 1))
        else:
            page_btn_range = range(paginator.num_pages - 4, paginator.num_pages + 1)
    # 페이지 버튼 생성
    links = []
    for pr in page_btn_range:
        if pr == page_number:
            links.append('<li class="page-item active"><a href="javascript:void(0)" class="page-link">%d</a></li>' % pr)
        else:
            links.append('<li class="page-item"><a href="javascript:go_page(%d)" class="page-link">%d</a></li>' % (pr, pr))
    # 페이지 이동
    return render(
        request,
        'project/project-list.html',
        {
            'page_links': links,
            'page_obj': page_obj
        }
    )



# 프로젝트 상세조회
def detail(request):
    # 퍼블화면 임시 이동
    return render(
        request,
        'project/project-detail.html'
    )
