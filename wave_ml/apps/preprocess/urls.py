from django.urls import path
from . import views


urlpatterns = [
    path('', views.main, name="preprocess"),
    path('process', views.process, name="process"),
    path('process/detail', views.detail, name="preprocess-detail"),
    path('process/data_type_change', views.type_change, name="preprocess-type-change"),
    path('process/process_missing_value', views.process_missing_value, name="process-missing-value"),
    path('process/process_outlier', views.process_outlier, name="process-outlier"),
    path('process/process_replace', views.process_replacing_value, name="process-replacing-value"),
    path('process/process_dummy', views.process_dummy, name="process-dummy"),
    path('process/process_scaler', views.process_scaler, name="process-scaler"),
    path('process/apply', views.process_apply, name="process-apply"),
    path('process/remove', views.process_remove, name="process-remove"),
    path('graph/boxplot', views.graph_boxplot, name="graph-boxplot"),
    path('graph/corr', views.graph_corr, name="graph-corr"),
    path('select', views.feature_select, name="feature_select"),
    path('select/execute', views.execute_feature_select, name="execute-feature-select"),
    path('split', views.data_split, name="data-split"),
    path('sampling', views.data_sampling, name="data-sampling"),
    path('sampling/execute', views.execute_data_sampling, name="execute-data-sampling"),
    path('sampling_apply', views.data_sampling_apply, name="data-sampling-apply"),
    path('data_create', views.data_create, name="data-create"),
    path('<int:page>', views.load_dataset_list),
    # path('dataset_list', views.load_dataset_list, name="dataset-list"),
    path('dataset_save', views.dataset_save, name="dataset-save"),
]
