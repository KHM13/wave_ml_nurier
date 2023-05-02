from django.shortcuts import render


# 모델 학습 화면 첫진입
def main(request):
    # 퍼블화면 임시 이동
    return render(
        request,
        'learning/model-learning.html'
    )
