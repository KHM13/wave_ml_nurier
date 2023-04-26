from django.shortcuts import render


# Create your views here.
def main(request):
    return render(
        request,
        'project/project-list.html'
    )


def detail(request):
    return render(
        request,
        'project/project-detail.html'
    )
