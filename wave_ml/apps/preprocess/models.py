from django.db import models
from wave_ml.apps.mlmodel.models import MlModel


# Create your models here.
class Process(models.Model):
    mlmodel_id = models.ForeignKey(MlModel, related_name="mlmodel", on_delete=models.CASCADE, to_field="id", db_column="mlmodel_id")
    column_name = models.TextField(blank=False)
    process_type = models.TextField(blank=False)
    work_type = models.TextField(blank=False)
    input_value = models.TextField()
    replace_value = models.TextField()
    sort = models.IntegerField()

    def get_process_list(self, mlmodel_id, column_name):
        result = self.__class__.objects.filter(mlmodel_id=mlmodel_id, column_name=column_name).order_by('sort').values()
        return result

    def get_process_name_list(self, mlmodel_id, column_name):
        model = self.__class__.objects.filter(mlmodel_id=mlmodel_id, column_name=column_name).order_by('sort').values()
        for m in model:
            if m['process_type'] == "missing":
                m['name'] = "결측치 처리"
            elif m['process_type'] == "outlier":
                m['name'] = "이상치 처리"
            elif m['process_type'] == "replace":
                m['name'] = f"문자열 통합:{m['replace_value']}"
            elif m['process_type'] == "datatype":
                m['name'] = "데이터 유형 변경"
            elif m['process_type'] == "dummy":
                m['name'] = "가변수화"
            elif m['process_type'] == "scaler":
                m['name'] = "데이터 정규화"
        return model


class MLSampling(models.Model):
    mlmodel_id = models.ForeignKey(MlModel, related_name="mlmodel_info", on_delete=models.CASCADE, to_field="id", db_column="mlmodel_id")
    dataset_name = models.TextField(blank=True, max_length=30)
    feature_algorithm = models.TextField()
    columns = models.TextField(blank=False)
    column_size = models.IntegerField(blank=False, default=0)
    split_algorithm = models.TextField()
    split_rate = models.IntegerField(blank=False)
    k_value = models.IntegerField(blank=False, default=0)
    sampling_algorithm = models.TextField()

    def get_feature_columns(self, mlmodel_id):
        model = self.__class__.objects.filter(mlmodel_id=mlmodel_id).values()
        result = {}
        if model is not None:
            result['algorithm'] = model.last()['feature_algorithm']
            result['columns'] = model.last()['columns']
        return result

    def get_split_info(self, mlmodel_id):
        model = self.__class__.objects.filter(mlmodel_id=mlmodel_id).values()
        result = {}
        if model is not None:
            result['split_algorithm'] = model.last()['split_algorithm']
            result['split_rate'] = model.last()['split_rate']
            result['k_value'] = model.last()['k_value']
        return result

    def get_info(self, mlmodel_id):
        model = self.__class__.objects.filter(mlmodel_id=mlmodel_id).values()
        if model is not None:
            return model.last()
        else:
            return {}
