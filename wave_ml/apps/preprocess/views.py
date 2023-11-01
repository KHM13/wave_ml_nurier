from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import JsonResponse, HttpResponse
from wave_ml.config.settings.base import MEDIA_ROOT
from wave_ml.ml import dataChecking, dataPreprocessing, dataChart
from wave_ml.ml.common import CommonUtil
from wave_ml.apps.mlmodel.models import MlModel
from wave_ml.apps.project.models import Project, ProjectFile
from .models import Process, MLSampling, MLDatasetFile
from datetime import datetime
import json
import os


# 데이터 전처리 화면 첫진입
# step 1. 데이터 확인 화면
def main(request):
    project_id = request.GET.get("project_id", request.session.get('project_id'))
    request.session.clear()

    # 모델 새로 등록 시
    if request.GET.get("model_name") is not None:
        project_model = Project.objects.get(id=project_id)
        mlmodel = MlModel(model_name=request.GET.get("model_name", "test"), project_id=project_model)
        mlmodel.save()
        request.session['mlmodel_id'] = mlmodel.id

    request.session['project_id'] = project_id
    # 프로젝트에 등록되어있는 파일 불러오기
    project_file_model = ProjectFile.objects.filter(project_id=project_id).values()
    df = dataChecking.files_to_df(project_file_model)

    # 프로젝트 내부에 저장되어있는 테스트용 임시 파일 불러오기
    # file = open('wave_ml/data/heart.csv', 'r', encoding='EUC-KR')
    # df = dataChecking.file_to_df(file)

    df_apply = df.copy()

    json_df = dataChecking.df_to_json(df)
    request.session['df'] = json_df
    mlmodel_id = request.GET.get("mlmodel_id", request.session.get("mlmodel_id"))
    request.session['mlmodel_id'] = mlmodel_id
    target = request.session.get("target", '')

    for column in df_apply.columns:
        process_list = Process().get_process_list(mlmodel_id, column)
        for p in process_list:
            df_apply = dataPreprocessing.process_apply(df_apply, column, p['process_type'], p['work_type'],
                                                       p['input_value'], p['replace_value'])

    if target == '' and MLSampling.objects.filter(mlmodel_id=mlmodel_id).exists():
        sampling = MLSampling().get_info(mlmodel_id)
        target = sampling['target']
        request.session['target'] = target

    json_df_apply = dataChecking.df_to_json(df_apply)
    request.session['df_apply'] = json_df_apply

    # 데이터 탐색 통계지표
    df_info = dataChecking.data_statistics(df_apply)
    # 데이터 상관관계 분석 그래프
    corr_graph = dataChart.correlation_graph(df_apply)

    return render(
        request,
        'preprocess/data-statistics.html',
        {
            'target': target,
            'df_info': df_info,
            'graph': corr_graph
        }
    )


# 목표 변수 설정
def save_target(request):
    target_id = request.POST.get("target_id", '')
    request.session['target'] = target_id

    mlmodel_id = request.session.get("mlmodel_id", request.POST.get("mlmodel_id"))

    if MLSampling.objects.filter(mlmodel_id=mlmodel_id).exists():
        MLSampling.objects.filter(mlmodel_id=mlmodel_id).update(target=target_id)
    else:  # 기존 작업이 저장되어 있지 않은 경우 새로 등록

        mlmodel = MlModel.objects.get(id=mlmodel_id)
        model = MLSampling(mlmodel_id=mlmodel, feature_algorithm="", columns="",
                           column_size=0,
                           split_algorithm="Random Split", split_rate=70, k_value=0, sampling_algorithm="", target=target_id)
        model.save()

    result = json.dumps({"result": "success"})
    return HttpResponse(result, content_type='application/json')


################################## 데이터 확인 END ##################################
################################## 변수 전처리 START ##################################

# step 2. 변수 전처리 화면
def process(request):
    df_apply = request.session['df_apply']
    mlmodel_id = request.session['mlmodel_id']
    df = dataChecking.json_to_df(df_apply)

    if request.POST.get('select_column'):   # 세션에 선택된 컬럼이 있는 경우
        select_column = request.POST.get('select_column')
    else:   # 화면 첫진입 시 변수 목록의 첫번째 컬럼을 기본 선택
        select_column = None
    columns, column = dataChecking.data_column_process(df, select_column)
    data_type = dataChecking.get_data_type(df, column) if not dataChecking.is_data_category(df, column) else "category"

    # 변수별 기존 전처리 작업 불러오기
    process_dict = {}
    for col in columns:
        process_dict[col] = len(Process().get_process_list(mlmodel_id, col))

    df_json = dataChecking.df_to_json(df)
    request.session['df_apply'] = df_json

    return render(
        request,
        'preprocess/data-preprocess.html',
        {
            "columns": columns,
            "column": column,
            "type": data_type,
            "process_dict": process_dict
        }
    )


# 변수별 데이터 상세 화면
def detail(request):
    column = request.POST.get('column')
    df_apply = request.session['df_apply']
    mlmodel_id = request.session['mlmodel_id']
    target = request.session['target']

    df = dataChecking.json_to_df(df_apply)
    # 해당 컬럼의 데이터 목록 조회
    value_list = dataChecking.data_detail(df, column)
    # 해당 컬럼의 데이터 통계값 조회
    
    desc = dataChecking.data_desc(df, column)
    # 해당 컬럼의 데이터 유형 조회 ( 데이터값 종류가 10개 미만이면 데이터유형을 카테고리형으로 선택 )
    data_type = dataChecking.get_data_type(df, column) if not dataChecking.is_data_category(df, column) else "category"

    # 해당 컬럼에 적용된 전처리 유형 목록 불러오기 ( 화면에 처리작업 뱃지 표기 )
    badge = Process().get_process_name_list(mlmodel_id, column)

    df_apply = dataChecking.df_to_json(df)
    request.session['df_preprocess'] = df_apply

    try:
        # 데이터분포 - 데이터 통계 그래프
        main_graph = dataChart.main_graph(df, column, target)
    except Exception as e:
        print(e)
        main_graph = ""

    return render(
        request,
        'preprocess/preprocess-detail.html',
        {
            "column": column,
            "detail": value_list,
            "desc": desc,
            "data_type": data_type,
            "main_graph": main_graph,
            "badge": badge
        }
    )


# 데이터 유형 변환
def type_change(request):
    df_apply = request.session['df_apply']
    target = request.session['target']
    column = request.POST.get('column')
    data_type = request.POST.get('type')
    select_tab = request.POST.get('select_tab')

    df = dataChecking.json_to_df(df_apply)
    # 데이터 유형 변경
    df_data = dataPreprocessing.data_type_control(df, column, data_type)

    # 변경 후 미리보기 화면을 위한 데이터 정보 재조회
    preview = dataChecking.data_preview(df_data, column, target)
    df_preprocess = dataChecking.df_to_json(df_data)
    request.session['df_preprocess'] = df_preprocess

    return render(
        request,
        'preprocess/process-preview.html',
        {
            "column": column,
            "detail": preview['value_list'],
            "desc": preview['desc'],
            "main_graph": preview['main_graph'],
            "outlier": preview['outlier'],
            "box_graph": preview['box_graph'],
            "freq": preview['freq'],
            "corr_graph": preview['corr_graph'],
            "select_tab": select_tab
        }
    )


# 결측치 처리 작업
def process_missing_value(request):
    df_apply = request.session['df_apply']
    target = request.session['target']
    column = request.POST.get('column')
    process = request.POST.get("process")
    input_value = request.POST.get("input_value", '')
    select_tab = request.POST.get("select_tab")

    df = dataChecking.json_to_df(df_apply)
    # 결측치 처리 작업 적용
    df_data = dataPreprocessing.process_missing(df, column, process, input_value)

    # 적용 후 미리보기 화면을 위한 데이터 정보 재조회
    preview = dataChecking.data_preview(df_data, column, target)
    df_preprocess = dataChecking.df_to_json(df_data)
    request.session['df_preprocess'] = df_preprocess

    return render(
        request,
        'preprocess/process-preview.html',
        {
            "column": column,
            "detail": preview['value_list'],
            "desc": preview['desc'],
            "main_graph": preview['main_graph'],
            "outlier": preview['outlier'],
            "box_graph": preview['box_graph'],
            "freq": preview['freq'],
            "corr_graph": preview['corr_graph'],
            "select_tab": select_tab
        }
    )


# 이상치 처리 작업
def process_outlier(request):
    df_apply = request.session['df_apply']
    target = request.session['target']
    column = request.POST.get('column')
    process = request.POST.get("process")
    input_value = request.POST.get("input_value", '')
    select_tab = request.POST.get("select_tab")

    df = dataChecking.json_to_df(df_apply)
    # 이상치 처리 작업 적용
    df_data = dataPreprocessing.process_outlier(df, column, process, input_value)

    # 적용 후 미리보기 화면을 위한 데이터 정보 재조회
    preview = dataChecking.data_preview(df_data, column, target)
    df_preprocess = dataChecking.df_to_json(df_data)
    request.session['df_preprocess'] = df_preprocess

    return render(
        request,
        'preprocess/process-preview.html',
        {
            "column": column,
            "detail": preview['value_list'],
            "desc": preview['desc'],
            "main_graph": preview['main_graph'],
            "outlier": preview['outlier'],
            "box_graph": preview['box_graph'],
            "freq": preview['freq'],
            "corr_graph": preview['corr_graph'],
            "select_tab": select_tab
        }
    )


# 사분위/중앙값 그래프
def graph_boxplot(request):
    if request.session.get('df_preprocess', None):
        df = request.session["df_preprocess"]
    else:
        df = request.session["df_apply"]
    column = request.POST.get("column")

    df_data = dataChecking.json_to_df(df)

    # 데이터 분포 박스 그래프
    result = dataChart.box_plot_graph(df_data, column, 750, 500)

    if result == "":
        return render(
            request,
            'preprocess/tab-boxplot.html',
            {
                "column": column,
                "outlier": "",
                "graph": "",
                "message": "※ 데이터 타입을 확인해주세요."
            }
        )
    
    # 해당 컬럼 이상치 조회
    outlier = dataPreprocessing.detect_outlier(df_data, column)
    return render(
        request,
        'preprocess/tab-boxplot.html',
        {
            "column": column,
            "outlier": outlier,
            "graph": result
        }
    )


# 목표변수와의 분포 그래프
def graph_corr(request):
    if request.session.get('df_preprocess', None):
        df = request.session["df_preprocess"]
    else:
        df = request.session["df_apply"]
    column = request.POST.get("column")

    target = request.session['target']
    df_data = dataChecking.json_to_df(df)
    
    
    # 해당 컬럼의 값 별 목표 변수 값 분포 그래프
    result = dataChart.scatter_graph(df_data, column, target)

    if result == "" :
        # 해당 컬럼이 문자열 or 목표 변수가 문자열
        ch1 = df_data[column].dtype == "object"
        ch2 = df_data[target].dtype == "object"
        message = ""
        if ch1:
            message = "※ 데이터 타입을 확인해주세요."
        if ch2:
            message = "※ 목표변수 타입이 문자열입니다."

        if ch1 or ch2:
            return render(
                request,
                'preprocess/tab-corr.html',
                {
                    "column": column,
                    "freq": "",
                    "graph": "",
                    "message": message
                }
            )

    # 해당 컬럼의 최빈값 조회
    freq = dataChecking.data_freq(df_data, column)

    return render(
        request,
        'preprocess/tab-corr.html',
        {
            "column": column,
            "freq": freq,
            "graph": result
        }
    )


# 문자열 통합 처리 작업
def process_replacing_value(request):
    df_apply = request.session['df_apply']
    target = request.session['target']
    column = request.POST.get('column')
    work_input = request.POST.get("work_input")
    replace_input = request.POST.get("replace_input")
    select_tab = request.POST.get("select_tab")

    df = dataChecking.json_to_df(df_apply)
    # 문자열 통합 처리 작업 적용
    df_data = dataPreprocessing.fs_replace_value(df, column, work_input, replace_input)
    # 적용 후 미리보기 화면을 위한 데이터 정보 재조회
    preview = dataChecking.data_preview(df_data, column, target)
    df_preprocess = dataChecking.df_to_json(df_data)
    request.session['df_preprocess'] = df_preprocess

    return render(
        request,
        'preprocess/process-preview.html',
        {
            "column": column,
            "detail": preview['value_list'],
            "desc": preview['desc'],
            "main_graph": preview['main_graph'],
            "outlier": preview['outlier'],
            "box_graph": preview['box_graph'],
            "freq": preview['freq'],
            "corr_graph": preview['corr_graph'],
            "select_tab": select_tab
        }
    )


# 데이터 가변수화 처리 작업
def process_dummy(request):
    df_apply = request.session['df_apply']
    target = request.session['target']
    column = request.POST.get('column')
    select_tab = request.POST.get('select_tab')

    df = dataChecking.json_to_df(df_apply)
    # 데이터 가변수화 처리 작업 적용
    df_data = dataPreprocessing.process_dummy_data(df, column)

    # 적용 후 미리보기 화면을 위한 데이터 정보 재조회
    preview = dataChecking.data_preview(df_data, column, target)
    df_preprocess = dataChecking.df_to_json(df_data)
    request.session['df_preprocess'] = df_preprocess

    return render(
        request,
        'preprocess/process-preview.html',
        {
            "column": column,
            "detail": preview['value_list'],
            "desc": preview['desc'],
            "main_graph": preview['main_graph'],
            "outlier": preview['outlier'],
            "box_graph": preview['box_graph'],
            "freq": preview['freq'],
            "corr_graph": preview['corr_graph'],
            "select_tab": select_tab
        }
    )


# 데이터 정규화 처리 작업
def process_scaler(request):
    df_apply = request.session['df_apply']
    target = request.session['target']
    column = request.POST.get('column')
    process = request.POST.get("process")
    select_tab = request.POST.get("select_tab")

    df = dataChecking.json_to_df(df_apply)
    # 데이터 정규화 처리 작업 적용
    df_data = dataPreprocessing.process_scaler(df, column, process)
    # 적용 후 미리보기 화면을 위한 데이터 정보 재조회
    preview = dataChecking.data_preview(df_data, column, target)
    df_preprocess = dataChecking.df_to_json(df_data)
    request.session['df_preprocess'] = df_preprocess

    return render(
        request,
        'preprocess/process-preview.html',
        {
            "column": column,
            "detail": preview['value_list'],
            "desc": preview['desc'],
            "main_graph": preview['main_graph'],
            "outlier": preview['outlier'],
            "box_graph": preview['box_graph'],
            "freq": preview['freq'],
            "corr_graph": preview['corr_graph'],
            "select_tab": select_tab
        }
    )

# 변수 삭제
def except_columns(request):
    column = request.POST.get('column')
    process = "delete"
    result = {}
    result['name'] = process
    
    # DB에 프로세스 추가
    mlmodel_id = request.session['mlmodel_id']

    # 해당 컬럼의 마지막 SORT 값 조회
    sort_list = Process().get_process_list(mlmodel_id, column)
    if sort_list is not None and len(sort_list) > 0:
        sort_index = int(sort_list.latest('sort')['sort']) + 1
    elif len(sort_list) == 1:
        sort_index = int(sort_list['sort']) + 1
    else:
        sort_index = 1

    # 삭제 저장(process)
    mlmodel = MlModel.objects.get(id=mlmodel_id)
    process_model = Process(mlmodel_id=mlmodel, column_name=column, process_type=process, work_type="",
                            input_value="", replace_value="", sort=sort_index)
    process_model.save()

    return JsonResponse(result)



# 처리 작업 적용
def process_apply(request):
    column = request.POST.get('column')
    process = request.POST.get('process')
    work = request.POST.get('work', '')
    work_input = request.POST.get('workInput', '')
    replace_input = request.POST.get('replaceInput', '')
    reprocess = request.POST.get('reprocess')

    mlmodel_id = request.session['mlmodel_id']

    # 기존 동일 처리 작업이 있는 경우
    if reprocess == "true":
        prev = request.POST.get('prev', '')
        df = request.session['df']
        df_apply = request.session['df_apply']
        df_apply = dataChecking.json_to_df(df_apply)
        df = dataChecking.json_to_df(df)
        df_apply[column] = df[column]

        # 기존 동일 처리 작업 삭제
        if process == "replace":
            Process.objects.filter(mlmodel_id=mlmodel_id, column_name=column, process_type=process, work_type=prev, replace_value=replace_input).delete()
        elif process == "dummy":
            Process.objects.filter(mlmodel_id=mlmodel_id, column_name=column, process_type=process).delete()
        else:
            Process.objects.filter(mlmodel_id=mlmodel_id, column_name=column, process_type=process, work_type=prev).delete()

    else:
        df = request.session['df_preprocess']
        request.session['df_apply'] = df

    # 해당 컬럼의 마지막 SORT 값 조회
    sort_list = Process().get_process_list(mlmodel_id, column)
    if sort_list is not None and len(sort_list) > 0:
        sort_index = int(sort_list.latest('sort')['sort']) + 1
    elif len(sort_list) == 1:
        sort_index = int(sort_list['sort']) + 1
    else:
        sort_index = 1

    # 새 작업 저장
    mlmodel = MlModel.objects.get(id=mlmodel_id)
    process_model = Process(mlmodel_id=mlmodel, column_name=column, process_type=process, work_type=work,
                            input_value=work_input, replace_value=replace_input, sort=sort_index)
    process_model.save()

    # 기존 동일 처리 작업이 있는 경우 - 해당 컬럼에 적용되어있던 작업 내용 적용하여 미리보기 화면 출력
    if reprocess == "true":
        process_list = Process().get_process_list(mlmodel_id, column)
        target = request.session['target']
        for p in process_list:
            # 해당 컬럼에 적용되어있던 처리 작업 적용
            df_apply = dataPreprocessing.process_apply(df_apply, column, p['process_type'], p['work_type'], p['input_value'], p['replace_value'])
        # 적용 후 미리보기 화면을 위한 데이터 정보 재조회
        preview = dataChecking.data_preview(df_apply, column, target)
        df = dataChecking.df_to_json(df_apply)
        request.session['df_apply'] = df

        return render(
            request,
            'preprocess/process-preview.html',
            {
                "column": column,
                "detail": preview['value_list'],
                "desc": preview['desc'],
                "main_graph": preview['main_graph'],
                "outlier": preview['outlier'],
                "box_graph": preview['box_graph'],
                "freq": preview['freq'],
                "corr_graph": preview['corr_graph']
            }
        )
    else:   # 새 작업처리인 경우 - 뱃지에 표시될 작업내역만 리턴
        result = {}
        if process == 'replace':
            process_name = dataChecking.get_process_name(process)
            result['name'] = f"{process_name}:{work_input}:{replace_input}"
        else:
            result['name'] = dataChecking.get_process_name(process)
        return JsonResponse(result)


# 적용 작업 삭제
def process_remove(request):
    df = request.session['df']
    mlmodel_id = request.session['mlmodel_id']
    target = request.session['target']

    select_column = request.POST.get('column')
    process = request.POST.get('process')
    work = request.POST.get('work', '')
    beforeData = request.POST.get('beforeData', '')
    replace_input = request.POST.get('replace_input', '')

    print(beforeData)
    print(replace_input)

    df = dataChecking.json_to_df(df)
    df_apply = df.copy()

    # 적용되었던 작업 삭제
    if process == "replace":
        Process.objects.filter(mlmodel_id=mlmodel_id, column_name=select_column, process_type=process, input_value=beforeData).delete()
    elif process == "dummy":
        Process.objects.filter(mlmodel_id=mlmodel_id, column_name=select_column, process_type=process).delete()
    elif process == "delete":
        Process.objects.filter(mlmodel_id=mlmodel_id, column_name=select_column, process_type=process).delete()
    else:
        Process.objects.filter(mlmodel_id=mlmodel_id, column_name=select_column, process_type=process, work_type=work).delete()
    

    # 삭제 후 남은 작업 처리 재적용
    for column in df_apply.columns:
        process_list = Process().get_process_list(mlmodel_id, column)
        for p in process_list:
            df_apply = dataPreprocessing.process_apply(df_apply, column, p['process_type'], p['work_type'], p['input_value'], p['replace_value'])

    # 적용 후 미리보기 화면을 위한 데이터 정보 재조회
    preview = dataChecking.data_preview(df_apply, select_column, target)
    df_json = dataChecking.df_to_json(df_apply)
    request.session['df_apply'] = df_json

    return render(
        request,
        'preprocess/process-preview.html',
        {
            "column": select_column,
            "detail": preview['value_list'],
            "desc": preview['desc'],
            "main_graph": preview['main_graph'],
            "outlier": preview['outlier'],
            "box_graph": preview['box_graph'],
            "freq": preview['freq'],
            "corr_graph": preview['corr_graph']
        }
    )

################################## 변수 전처리 END ##################################
################################## 데이터셋 전처리 START ##################################

# step 3. 데이터셋 전처리 화면
def feature_select(request):
    df_apply = request.session['df_apply']
    target = request.session['target']
    df = dataChecking.json_to_df(df_apply)
    mlmodel_id = request.session['mlmodel_id']
    
    # 삭제 변수 제외
    if Process.objects.filter(mlmodel_id=mlmodel_id, process_type="delete").exists():
        deleted_list = Process.objects.filter(mlmodel_id=mlmodel_id, process_type="delete").values()
        for i in deleted_list:
            df = df.drop(i['column_name'], axis=1)

    info = dataChecking.get_data_info(df, target)
    
    # 데이터셋 전처리 작업 불러오기
    if MLSampling.objects.filter(mlmodel_id=mlmodel_id).exists():   # 기존 작업 존재할경우
        sampling = MLSampling().get_feature_columns(mlmodel_id)     # 변수 선택법 적용 내용 불러오기
        select_column = sampling['columns']
        feature_algorithm = sampling['algorithm']
    else:
        select_column = info['columns']
        feature_algorithm = "Customizing"

    return render(
        request,
        'preprocess/dataset-preprocess.html',
        {
            'info': info,
            'select_column': select_column,
            'feature_algorithm': feature_algorithm,
            'target': target
        }
    )


# 변수 선택법 알고리즘 실행
def execute_feature_select(request):
    df_apply = request.session['df_apply']
    df = dataChecking.json_to_df(df_apply)
    target = request.session['target']

    
    # 전체 컬럼 목록 및 사이즈
    columns = dataChecking.get_column_list(df)
    info = {'columns': columns, 'size': len(columns)}

    # 실행할 변수선택법 json 으로 보낸 Array 데이터 리스트로 변환
    feature_select = json.loads(request.POST.get('feature_select'))
    # 선택되어있던 변수선택법 알고리즘명 불러오기
    select_algorithm = request.POST.get("select_algorithm")

    result = []
    for fs in feature_select:
        # 변수선택법 적용
        fs_result = dataPreprocessing.execute_fs(df, target, fs)
        result.append(fs_result)

    return render(
        request,
        'preprocess/feature-select.html',
        {
            'feature_select': result,
            'info': info,
            'select_algorithm': select_algorithm
        }
    )


# 변수선택법 저장
# 데이터 분할 화면 진입
def data_split(request):
    mlmodel_id = request.session['mlmodel_id']

    # 변수선택법 화면에서 넘어올때 적용될 값
    if request.POST.get("select_column") is not None:
        fs_name = request.POST.get("fs_name")
        select_column_list = json.loads(request.POST.get("select_column"))

        df_apply = request.session['df_apply']
        df_preprocess = dataChecking.json_to_df(df_apply)
        select_df = df_preprocess[select_column_list]
        request.session['df_fs'] = dataChecking.df_to_json(select_df)

        select_column = json.dumps(select_column_list)

        # 기존 데이터셋 전처리 작업이 저장되어 있는 경우 수정
        if MLSampling.objects.filter(mlmodel_id=mlmodel_id).exists():
            MLSampling.objects.filter(mlmodel_id=mlmodel_id).update(feature_algorithm=fs_name, columns=select_column, column_size=len(select_column_list))
        else:   # 기존 작업이 저장되어 있지 않은 경우 새로 등록
            mlmodel = MlModel.objects.get(id=mlmodel_id)
            model = MLSampling(mlmodel_id=mlmodel, feature_algorithm=fs_name, columns=select_column, column_size=len(select_column_list),
                               split_algorithm="Random Split", split_rate=70, k_value=0, sampling_algorithm="")
            model.save()

    df_fs = request.session.get('df_fs', None)
    df = None
    if df_fs is None:
        if MLSampling.objects.filter(mlmodel_id=mlmodel_id).exists():
            sampling = MLSampling.objects.get(mlmodel_id=mlmodel_id)
            columns = json.loads(sampling.columns)
            df_apply = request.session.get("df_apply")
            df_apply_temp = dataChecking.json_to_df(df_apply)
            df = df_apply_temp[columns]
    else:
        df = dataChecking.json_to_df(df_fs)
    target = request.session['target']
    info = dataChecking.get_data_info(df, target)

    sampling_model = MLSampling().get_info(mlmodel_id)
    info['size'] = len(df.columns)
    info['columns'] = df.columns

    return render(
        request,
        'preprocess/data-split.html',
        {
            'info': info,
            'split_algorithm': sampling_model['split_algorithm'],
            'split_rate': sampling_model['split_rate'],
            'k_value': sampling_model['k_value']
        }
    )


# 데이터 분할 알고리즘 적용
# 불균형 데이터 보정 화면 진입
def data_sampling(request):
    mlmodel_id = request.session['mlmodel_id']
    target = request.session['target']

    df_fs = request.session.get('df_fs', None)
    df = None
    is_df_apply = False
    if df_fs is None:
        if MLSampling.objects.filter(mlmodel_id=mlmodel_id).exists():
            sampling = MLSampling.objects.get(mlmodel_id=mlmodel_id)
            columns = json.loads(sampling.columns)
            df_apply = request.session.get("df_apply")
            df_apply_temp = dataChecking.json_to_df(df_apply)
            df = df_apply_temp[columns]
            is_df_apply = True
    else:
        df = dataChecking.json_to_df(df_fs)

    models = MLSampling().get_info(mlmodel_id)
    columns = df.columns
    columns = columns.drop(target)
    select_sampling_algorithm = models['sampling_algorithm']

    # 데이터 분할 화면에서 넘어올때 적용될 값
    if request.POST.get("algorithm") is not None or is_df_apply:
        algorithm = request.POST.get("algorithm", models['split_algorithm'])
        split_value = request.POST.get("split_value").replace('%', '') if request.POST.get("split_value") != '' and request.POST.get("split_value") is not None else models['split_rate']
        k_value = request.POST.get("k_value") if request.POST.get("k_value") != '' and request.POST.get("k_value") is not None else models['k_value']

        # 데이터 분할 알고리즘 적용 저장
        if not is_df_apply:
            MLSampling.objects.filter(mlmodel_id=mlmodel_id).update(split_algorithm=algorithm, split_rate=int(split_value), k_value=int(k_value))

        if algorithm == "Random Split":
            train_data, test_data = dataPreprocessing.train_test_data_division(df, target, columns, int(split_value))
            request.session['train_data'] = dataChecking.df_to_json(train_data)
            request.session['test_data'] = dataChecking.df_to_json(test_data)
        elif algorithm == "K-fold Cross Validation":
            train_list, test_list = dataPreprocessing.k_fold_cross_validation(df, columns, int(k_value), target)
            for idx, train_data in enumerate(train_list):
                if idx == 0:
                    request.session['train_data'] = dataChecking.df_to_json(train_data)
                else:
                    request.session[f'train_data{idx}'] = dataChecking.df_to_json(train_data)
            for idx, test_data in enumerate(test_list):
                if idx == 0:
                    request.session['test_data'] = dataChecking.df_to_json(test_data)
                else:
                    request.session[f'test_data{idx}'] = dataChecking.df_to_json(test_data)
        else:
            train_list, test_list = dataPreprocessing.shuffle_split(df, columns, int(split_value), int(k_value), target)
            for idx, train_data in enumerate(train_list):
                if idx == 0:
                    request.session['train_data'] = dataChecking.df_to_json(train_data)
                else:
                    request.session[f'train_data{idx}'] = dataChecking.df_to_json(train_data)
            for idx, test_data in enumerate(test_list):
                if idx == 0:
                    request.session['test_data'] = dataChecking.df_to_json(test_data)
                else:
                    request.session[f'test_data{idx}'] = dataChecking.df_to_json(test_data)

    train_data = dataChecking.json_to_df(request.session['train_data'])
    test_data = dataChecking.json_to_df(request.session['test_data'])
    info = dataChecking.get_train_test_info(train_data, test_data, target)

    return render(
        request,
        'preprocess/data-sampling.html',
        {
            'info': info,
            'sampling_algorithm': select_sampling_algorithm
        }
    )


# 불균형 데이터 보정 알고리즘 실행
def execute_data_sampling(request):
    data_sampling = json.loads(request.POST.get("data_sampling"))
    train_data = request.session['train_data']
    df_train = dataChecking.json_to_df(train_data)
    target = request.session['target']

    sampling_algorithm = request.POST.get("sampling_algorithm")
    if sampling_algorithm == "":
        sampling_algorithm = "OriginalData"

    info = dataChecking.get_target_info(df_train, target)
    samplig_list = []

    # 알고리즘별 불균형 데이터 보정 실행 적용
    for algorithm in data_sampling:
        balanced_data = dataPreprocessing.set_imbalanced_data(algorithm, df_train, target)
        balanced_info = dataChecking.get_target_info(balanced_data, target)
        balanced_info['algorithm'] = algorithm

        if int(balanced_info['train']) > int(info['train']):
            balanced_info['arrow'] = "up"
            balanced_info['percent'] = int(int(balanced_info['train'])/int(info['train'])*100) - 100
        elif int(balanced_info['train']) < int(info['train']):
            balanced_info['arrow'] = "down"
            balanced_info['percent'] = 100 - int(int(balanced_info['train'])/int(info['train'])*100)
        else:
            balanced_info['arrow'] = "dash"
            balanced_info['percent'] = 0
        samplig_list.append(balanced_info)

    return render(
        request,
        'preprocess/imbalanced-data.html',
        {
            'train_info': info,
            'samplig_list': samplig_list,
            'sampling_algorithm': sampling_algorithm
        }
    )


# 불균형 데이터 보정 적용 저장
def data_sampling_apply(request):
    try:
        select_algorithm = request.POST.get("select_algorithm")
        train_data = request.session['train_data']
        train_df = dataChecking.json_to_df(train_data)
        mlmodel_id = request.session['mlmodel_id']
        target = request.session['target']

        # 선택된 불균형 데이터 보정 알고리즘 저장
        MLSampling.objects.filter(mlmodel_id=mlmodel_id).update(sampling_algorithm=select_algorithm)

        if select_algorithm != "OriginalData":
            balanced_data = dataPreprocessing.set_imbalanced_data(select_algorithm, train_df, target)
            request.session['train_data'] = dataChecking.df_to_json(balanced_data)

        result = json.dumps({"result": "success"})
        return HttpResponse(result, content_type='application/json')

    except Exception as e:
        print(e)
        result = json.dumps({"result": "error"})
        return HttpResponse(result, content_type='application/json')


################################## 데이터셋 전처리 END ##################################
################################## 학습 데이터 생성 START ##################################


# step 4. 학습 데이터 생성 화면
def data_create(request):
    df_apply = request.session['df_apply']
    target = request.session['target']
    df = dataChecking.json_to_df(df_apply)
    data_info = dataChecking.get_data_info(df, target)

    project_id = request.session.get('project_id', request.POST.get('project_id'))
    project_obj = Project.objects.get(id=project_id)

    mlmodel_id = request.session['mlmodel_id']
    mlmodel = MlModel.objects.get(id=mlmodel_id)

    # 현재 모델에 적용된 전처리 작업 리스트
    process_models = Process.objects.filter(mlmodel_id=mlmodel_id).values()
    process_size = len(process_models)

    # 현재 모델에 적용된 데이터셋 처리 작업 리스트
    sampling_models = MLSampling().get_info(mlmodel_id)

    # 현재 모델에 적용된 데이터셋을 제외한 프로젝트 내 데이터셋 리스트
    list_models = MLSampling.objects.exclude(mlmodel_id=mlmodel_id)
    dataset_size = len(list_models.values())

    page_number = request.GET.get('page', 1)
    # 페이징 처리
    page_obj, links = CommonUtil.paginator(page_number, 2, list_models)

    return render(
        request,
        'preprocess/data-create.html',
        {
            'mlmodel': mlmodel,
            'project_id': project_id,
            'registrant': project_obj.registrant,
            'data_info': data_info,
            'process_size': process_size,
            'sampling_models': sampling_models,
            'dataset_size': dataset_size,
            'list_models': page_obj,
            'page_links': links
        }
    )


# 페이징처리
def load_dataset_list(request, page):
    mlmodel_id = request.session['mlmodel_id']
    list_models = MLSampling.objects.exclude(mlmodel_id=mlmodel_id)
    dataset_size = len(list_models.values())
    # 페이징 처리
    page_obj, links = CommonUtil.paginator(page, 2, list_models)

    return render(
        request,
        'preprocess/dataset-list.html',
        {
            'dataset_size': dataset_size,
            'list_models': page_obj,
            'page_links': links
        }
    )


# 데이터셋 저장
def dataset_save(request):
    mlmodel_id = request.session['mlmodel_id']
    dataset_name = request.POST.get("dataset_name")

    df_apply = request.session.get("df_fs", None)
    is_df_apply = False
    if df_apply is None:
        is_df_apply = True
        df_apply = request.session.get("df_apply")
    df = dataChecking.json_to_df(df_apply)

    try:
        # 데이터셋전처리 메뉴에서 바로 학습데이터 생성 화면으로 넘어왔을 때
        if MLSampling.objects.filter(mlmodel_id=mlmodel_id).exists():
            mlsampling = MLSampling.objects.get(mlmodel_id=mlmodel_id)
            columns = json.loads(mlsampling.columns)
            if is_df_apply:
                df = df[columns]

        file_name = f'{dataset_name}_{CommonUtil.CommonUtil().now_type4}'
        file_path = f'{MEDIA_ROOT}\\file\\dataset\\{CommonUtil.CommonUtil().now_type3}\\'
        os.makedirs(file_path, exist_ok=True)
        df.to_csv(f'{file_path}{file_name}.csv', sep=",", mode="w", encoding="EUC-KR", index=False, header=True, chunksize=100)

        if MLSampling.objects.filter(mlmodel_id=mlmodel_id).exists():
            MLSampling.objects.filter(mlmodel_id=mlmodel_id).update(dataset_name=dataset_name)

        if MLDatasetFile.objects.filter(mlmodel_id=mlmodel_id).exists():
            exists_file = MLDatasetFile.objects.get(mlmodel_id=mlmodel_id)
            if os.path.isfile(f'{exists_file.dataset_file}.{exists_file.dataset_file_extension}'):
                os.remove(f'{exists_file.dataset_file}.{exists_file.dataset_file_extension}')
            exists_file.delete()

        mlmodel = MlModel.objects.get(id=mlmodel_id)

        MLDatasetFile.objects.create(
            mlmodel_id=mlmodel,
            dataset_name=dataset_name,
            dataset_file=f'{file_path}{file_name}',
            dataset_file_extension='csv'
        )

        result = json.dumps({"result": "success"})
        return HttpResponse(result, content_type='application/json')
    except Exception as e:
        print(e)
        result = json.dumps({"result": "error"})
        return HttpResponse(result, content_type='application/json')
