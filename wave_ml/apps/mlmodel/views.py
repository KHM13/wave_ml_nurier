from django.http import HttpResponse
from django.shortcuts import render
from django.db.models import Q

from wave_ml.apps.learning.models import ModelLearning

import json
from ast import literal_eval


def main(request):
    mlmodel_id = request.GET.get("mlmodel_id", request.session.get("mlmodel_id"))
    project_id = request.GET.get("project_id", request.session.get("project_id"))
    request.session['mlmodel_id'] = mlmodel_id
    request.session['project_id'] = project_id

    mlmodel = []
    if ModelLearning.objects.filter(mlmodel_id=mlmodel_id).exclude(recall=None).exists():
        mlmodel = ModelLearning.objects.filter(mlmodel_id=mlmodel_id).order_by('-recall').values()

    return render(
        request,
        'mlmodel/model-result.html',
        {
            'mlmodel': mlmodel
        }
    )


def detail(request):
    algorithm = request.POST.get("algorithm", "")
    mlmodel_id = request.session.get("mlmodel_id")
    model = None
    confusion_matrix = {}

    try:
        model = ModelLearning.objects.filter(mlmodel_id=mlmodel_id, algorithm=algorithm, learning_date__isnull=False).values().first()
        confusion_matrix = literal_eval(model['confusion_matrix'])

    except Exception as e:
        print(e)

    return render(
        request,
        'mlmodel/model-result-detail.html',
        {
            'select_algorithm': algorithm,
            'evaluation': model,
            'confusion_matrix': confusion_matrix
        }
    )


def control_favorite(request):
    try:
        algorithm = request.POST.get("algorithm", "")
        mlmodel_id = request.session.get("mlmodel_id")
        favorite = request.POST.get("favorite", False)

        ModelLearning.objects.filter(mlmodel_id=mlmodel_id, algorithm=algorithm).update(Favorite=favorite)

        return HttpResponse({"result": "success"}, content_type='application/json')

    except Exception as e:
        print(e)
        return HttpResponse({"result": "error"}, content_type='application/json')
