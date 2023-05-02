from django.shortcuts import render


# 모델 평가 화면 첫진입
def main(request):
    # 퍼블화면 임시 이동
    return render(
        request,
        'mlmodel/model-result.html'
    )
