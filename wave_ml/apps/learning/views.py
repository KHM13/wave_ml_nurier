from django.http import HttpResponse
from django.shortcuts import render


# 모델 학습 화면 첫진입
from django.views.decorators.csrf import csrf_exempt


def main(request):
    return render(
        request,
        'learning/model-learning.html'
    )


@csrf_exempt
def detail(request):
    algorithm = request.POST.get("algorithm", "")
    explanation = ""

    if algorithm == "Linear Regression":
        explanation = "종속 변수 y와 한 개 이상의 독립 변수 X와의 선형 상관 관계를 모델링하는 회귀분석 기법"
    elif algorithm == "Decision Tree Regression":
        explanation = "오차 제곱합을 가장 잘 줄일 수 있는 변수를 기준으로 분기를 만들어 결과를 예측하는 모델"
    elif algorithm == "Random Forest Regression":
        explanation = "특정 특성을 선택하는 트리를 여러 개 생성하여 이들을 기반으로 작업을 수행하는 앙상블 기법"
    elif algorithm == "Logistic Regression":
        explanation = "독립 변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측하는데 사용되는 통계 기법"
    elif algorithm == "LinearSVC":
        explanation = "퍼셉트론 기반의 모형에 가장 안정적인 판별 경계선을 찾기 위한 제한 조건을 추가한 모형"
    elif algorithm == "Naïve Bayes":
        explanation = "데이터가 각 클래스에 속할 특징 확률을 계산하는 조건부 확률 기반의 분류 기법"
    elif algorithm == "Decision Tree Classifier":
        explanation = "오차 제곱합을 가장 잘 줄일 수 있는 변수를 기준으로 분기를 만들어 결과를 예측하는 모델"
    elif algorithm == "Random Forest Classifier":
        explanation = "특정 특성을 선택하는 트리를 여러 개 생성하여 이들을 기반으로 작업을 수행하는 앙상블 기법"
    elif algorithm == "Multilayer Perceptron Classifier":
        explanation = "입력층과 출력층 사이에 하나 이상의 중간층이 존재하는 신경망으로 계층구조를 갖는 모델"
    elif algorithm == "Gradient Boosted Tree Classifier":
        explanation = "잔차를 이용하여 이전 모형의 약점을 보완하는 새로운 모형을 순차적으로 적합한 뒤 선형 결합하여 얻어진 모형"

    return render(
        request,
        'learning/learning-detail.html',
        {
            'select_algorithm': algorithm,
            'explanation_for_algorithm': explanation
        }
    )

@csrf_exempt
def save_model_list(request):

    return HttpResponse("success", content_type='application/json')
