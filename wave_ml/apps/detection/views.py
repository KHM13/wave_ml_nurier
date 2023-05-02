from django.shortcuts import render


# 탐지 모델 화면 첫진입
def main(request):
    # 퍼블화면 임시 이동
    return render(
        request,
        'detection/detection.html'
    )
