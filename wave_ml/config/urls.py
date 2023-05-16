from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('project/', include('wave_ml.apps.project.urls')),
    path('preprocess/', include('wave_ml.apps.preprocess.urls')),
    path('learning/', include('wave_ml.apps.learning.urls')),
    path('evaluation/', include('wave_ml.apps.mlmodel.urls')),
    path('detection/', include('wave_ml.apps.detection.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
