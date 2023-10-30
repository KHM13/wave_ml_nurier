from django.shortcuts import render

from wave_ml.apps.learning.models import ModelLearning


def main(request):
    mlmodel_id = request.GET.get("mlmodel_id", request.session.get("mlmodel_id"))
    project_id = request.GET.get("project_id", request.session.get("project_id"))
    request.session['mlmodel_id'] = mlmodel_id
    request.session['project_id'] = project_id

    mlmodel = []
    if ModelLearning.objects.filter(mlmodel_id=mlmodel_id).exists():
        mlmodel = ModelLearning.objects.filter(mlmodel_id=mlmodel_id).order_by('-recall').values()
        print(mlmodel)

    return render(
        request,
        'mlmodel/model-result.html',
        {
            'mlmodel': mlmodel
        }
    )


def detail(request):
    algorithm = request.POST.get("algorithm", "")

    return render(
        request,
        'mlmodel/model-result-detail.html',
        {
            'select_algorithm': algorithm,
        }
    )