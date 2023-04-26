from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import JsonResponse, HttpResponse
from wave_ml.ml import dataChecking, dataPreprocessing, dataChart
from .models import Process, MLSampling
import json


def main(request):
    if request.session.get('df') is None:
        file = open('wave_ml/data/heart.csv', 'r', encoding='EUC-KR')
        df = dataChecking.file_to_df(file)
        df_apply = df.copy()

        project_id = "test_project_17"
        json_df = dataChecking.df_to_json(df)
        request.session['df'] = json_df

        for column in df_apply.columns:
            process_list = Process().get_process_list(project_id, column)
            for p in process_list:
                df_apply = dataPreprocessing.process_apply(df_apply, column, p['process_type'], p['work_type'],
                                                           p['input_value'], p['replace_value'])

        json_df_apply = dataChecking.df_to_json(df_apply)
        request.session['df_apply'] = json_df_apply
        request.session['project_id'] = project_id
    else:
        df_json = request.session['df_apply']
        request.session['df_apply'] = request.session['df_apply']
        df_apply = dataChecking.json_to_df(df_json)

    df_info = dataChecking.data_statistics(df_apply)
    corr_graph = dataChart.correlation_graph(df_apply)

    return render(
        request,
        'preprocess/data-statistics.html',
        {
            'df_info': df_info,
            'graph': corr_graph
        }
    )

################################## 데이터 확인 END ##################################
################################## 변수 전처리 START ##################################

@csrf_exempt
def process(request):
    df_apply = request.session['df_apply']
    project_id = request.session['project_id']
    df = dataChecking.json_to_df(df_apply)
    if request.POST.get('select_column'):
        select_column = request.POST.get('select_column')
    else:
        select_column = None
    columns, column = dataChecking.data_column_process(df, select_column)
    data_type = dataChecking.get_data_type(df, column) if not dataChecking.is_data_category(df, column) else "category"

    process_dict = {}
    for col in columns:
        process_dict[col] = len(Process().get_process_list(project_id, col))

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


@csrf_exempt
def detail(request):
    column = request.POST.get('column')
    df_apply = request.session['df_apply']
    project_id = request.session['project_id']

    df = dataChecking.json_to_df(df_apply)
    value_list = dataChecking.data_detail(df, column)
    desc = dataChecking.data_desc(df, column)
    data_type = dataChecking.get_data_type(df, column) if not dataChecking.is_data_category(df, column) else "category"

    badge = Process().get_process_name_list(project_id, column)

    df_apply = dataChecking.df_to_json(df)
    request.session['df_preprocess'] = df_apply

    main_graph = dataChart.main_graph(df, column)

    return render(
        request,
        'preprocess/preprocess-detail.html',
        {
            "column": column,
            "detail": value_list,
            "desc": desc,
            "type": data_type,
            "main_graph": main_graph,
            "badge": badge
        }
    )


@csrf_exempt
def type_change(request):
    df_apply = request.session['df_apply']
    column = request.POST.get('column')
    data_type = request.POST.get('type')
    select_tab = request.POST.get('select_tab')

    df = dataChecking.json_to_df(df_apply)
    df_data = dataPreprocessing.data_type_control(df, column, data_type)

    preview = dataChecking.data_preview(df_data, column)
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


@csrf_exempt
def process_missing_value(request):
    df_apply = request.session['df_apply']
    column = request.POST.get('column')
    process = request.POST.get("process")
    input_value = request.POST.get("input_value", '')
    select_tab = request.POST.get("select_tab")

    df = dataChecking.json_to_df(df_apply)
    df_data = dataPreprocessing.process_missing(df, column, process, input_value)

    preview = dataChecking.data_preview(df_data, column)
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


@csrf_exempt
def process_outlier(request):
    df_apply = request.session['df_apply']
    column = request.POST.get('column')
    process = request.POST.get("process")
    input_value = request.POST.get("input_value", '')
    select_tab = request.POST.get("select_tab")

    df = dataChecking.json_to_df(df_apply)
    df_data = dataPreprocessing.process_outlier(df, column, process, input_value)

    preview = dataChecking.data_preview(df_data, column)
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


@csrf_exempt
def graph_boxplot(request):
    if request.session.get('df_preprocess', None):
        df = request.session["df_preprocess"]
    else:
        df = request.session["df_apply"]
    column = request.POST.get("column")

    df_data = dataChecking.json_to_df(df)
    outlier = dataPreprocessing.detect_outlier(df_data, column)
    result = dataChart.box_plot_graph(df_data, column, 750, 500)
    return render(
        request,
        'preprocess/tab-boxplot.html',
        {
            "column": column,
            "outlier": outlier,
            "graph": result
        }
    )


@csrf_exempt
def graph_corr(request):
    if request.session.get('df_preprocess', None):
        df = request.session["df_preprocess"]
    else:
        df = request.session["df_apply"]
    column = request.POST.get("column")

    df_data = dataChecking.json_to_df(df)
    result = dataChart.scatter_graph(df_data, column)
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


@csrf_exempt
def process_replacing_value(request):
    df_apply = request.session['df_apply']
    column = request.POST.get('column')
    work_input = request.POST.get("work_input")
    replace_input = request.POST.get("replace_input")
    select_tab = request.POST.get("select_tab")

    df = dataChecking.json_to_df(df_apply)
    df_data = dataPreprocessing.replace_value(df, column, work_input, replace_input)

    preview = dataChecking.data_preview(df_data, column)
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


@csrf_exempt
def process_dummy(request):
    df_apply = request.session['df_apply']
    column = request.POST.get('column')
    select_tab = request.POST.get('select_tab')

    df = dataChecking.json_to_df(df_apply)
    df_data = dataPreprocessing.process_dummy_data(df, column)

    preview = dataChecking.data_preview(df_data, column)
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


@csrf_exempt
def process_scaler(request):
    df_apply = request.session['df_apply']
    column = request.POST.get('column')
    process = request.POST.get("process")
    select_tab = request.POST.get("select_tab")

    df = dataChecking.json_to_df(df_apply)
    df_data = dataPreprocessing.process_scaler(df, column, process)
    preview = dataChecking.data_preview(df_data, column)
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


@csrf_exempt
def process_apply(request):
    column = request.POST.get('column')
    process = request.POST.get('process')
    work = request.POST.get('work', '')
    work_input = request.POST.get('workInput', '')
    replace_input = request.POST.get('replaceInput', '')
    reprocess = request.POST.get('reprocess')

    project_id = request.session['project_id']

    if reprocess == "true":
        prev = request.POST.get('prev', '')
        df = request.session['df']
        df_apply = request.session['df_apply']
        df_apply = dataChecking.json_to_df(df_apply)
        df = dataChecking.json_to_df(df)
        df_apply[column] = df[column]

        if process == "replace":
            Process.objects.filter(project_id=project_id, column_name=column, process_type=process, work_type=prev, replace_value=replace_input).delete()
        elif process == "dummy":
            Process.objects.filter(project_id=project_id, column_name=column, process_type=process).delete()
        else:
            Process.objects.filter(project_id=project_id, column_name=column, process_type=process, work_type=prev).delete()

    else:
        df = request.session['df_preprocess']
        request.session['df_apply'] = df

    sort_list = Process().get_process_list(project_id, column)
    if sort_list is not None and len(sort_list) > 0:
        sort_index = int(sort_list.latest('sort')['sort']) + 1
    elif len(sort_list) == 1:
        sort_index = int(sort_list['sort']) + 1
    else:
        sort_index = 1

    process_model = Process(project_id=project_id, column_name=column, process_type=process, work_type=work,
                            input_value=work_input, replace_value=replace_input, sort=sort_index)
    process_model.save()

    if reprocess == "true":
        process_list = Process().get_process_list(project_id, column)
        for p in process_list:
            df_apply = dataPreprocessing.process_apply(df_apply, column, p['process_type'], p['work_type'], p['input_value'], p['replace_value'])
        preview = dataChecking.data_preview(df_apply, column)
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
    else:
        result = {}
        if process == 'replace':
            process_name = dataChecking.get_process_name(process)
            result['name'] = f"{process_name}:{replace_input}"
        else:
            result['name'] = dataChecking.get_process_name(process)
        return JsonResponse(result)


@csrf_exempt
def process_remove(request):
    df = request.session['df']
    project_id = request.session['project_id']

    select_column = request.POST.get('column')
    process = request.POST.get('process')
    work = request.POST.get('work', '')
    replace_input = request.POST.get('replace_input', '')

    df = dataChecking.json_to_df(df)
    df_apply = df.copy()

    if process == "replace":
        Process.objects.filter(project_id=project_id, column_name=select_column, process_type=process, work_type=work, replace_value=replace_input).delete()
    elif process == "dummy":
        Process.objects.filter(project_id=project_id, column_name=select_column, process_type=process).delete()
    else:
        Process.objects.filter(project_id=project_id, column_name=select_column, process_type=process, work_type=work).delete()

    for column in df_apply.columns:
        process_list = Process().get_process_list(project_id, column)
        for p in process_list:
            df_apply = dataPreprocessing.process_apply(df_apply, column, p['process_type'], p['work_type'], p['input_value'], p['replace_value'])

    preview = dataChecking.data_preview(df_apply, select_column)
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

@csrf_exempt
def feature_select(request):
    df_apply = request.session['df_apply']
    df = dataChecking.json_to_df(df_apply)
    info = dataChecking.get_data_info(df)

    project_id = request.session['project_id']

    if MLSampling.objects.filter(project_id=project_id).exists():
        sampling = MLSampling().get_feature_columns(project_id)
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
            'feature_algorithm': feature_algorithm
        }
    )


@csrf_exempt
def execute_feature_select(request):
    df_apply = request.session['df_apply']
    df = dataChecking.json_to_df(df_apply)

    columns = dataChecking.get_column_list(df)
    info = {'columns': columns, 'size': len(columns)}

    feature_select = json.loads(request.POST.get('feature_select'))
    select_algorithm = request.POST.get("select_algorithm")

    result = []
    for fs in feature_select:
        fs_result = dataPreprocessing.execute_fs(df, "output", fs)
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


@csrf_exempt
def data_split(request):
    df_apply = request.session['df_apply']
    df = dataChecking.json_to_df(df_apply)
    info = dataChecking.get_data_info(df)

    project_id = request.session['project_id']

    if request.POST.get("select_column") is not None:
        fs_name = request.POST.get("fs_name")
        select_column_list = json.loads(request.POST.get("select_column"))
        select_column = json.dumps(select_column_list)

        if MLSampling.objects.filter(project_id=project_id).exists():
            MLSampling.objects.filter(project_id=project_id).update(feature_algorithm=fs_name, columns=select_column, column_size=len(select_column_list))
        else:
            model = MLSampling(project_id=project_id, feature_algorithm=fs_name, columns=select_column, column_size=len(select_column_list),
                               split_algorithm="Random Split", split_rate=70, k_value=0, sampling_algorithm="")
            model.save()

    sampling_model = MLSampling().get_info(project_id)
    columns = json.loads(sampling_model['columns'])
    info['size'] = sampling_model['column_size']
    info['columns'] = columns

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


@csrf_exempt
def data_sampling(request):
    project_id = request.session['project_id']
    df_apply = request.session['df_apply']
    df = dataChecking.json_to_df(df_apply)
    models = MLSampling().get_info(project_id)
    columns = json.loads(models['columns'])
    columns.remove("output")
    select_sampling_algorithm = models['sampling_algorithm']

    if request.POST.get("algorithm") is not None:
        algorithm = request.POST.get("algorithm")
        split_value = request.POST.get("split_value").replace('%', '') if request.POST.get("split_value") != '' else 0
        k_value = request.POST.get("k_value") if request.POST.get("k_value") != '' else 0
        MLSampling.objects.filter(project_id=project_id).update(split_algorithm=algorithm, split_rate=int(split_value), k_value=int(k_value))

        if algorithm == "Random Split":
            train_data, test_data = dataPreprocessing.train_test_data_division(df, columns, int(split_value))
            request.session['train_data'] = dataChecking.df_to_json(train_data)
            request.session['test_data'] = dataChecking.df_to_json(test_data)
        elif algorithm == "K-fold Cross Validation":
            train_list, test_list = dataPreprocessing.k_fold_cross_validation(df, columns, int(k_value))
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
            train_list, test_list = dataPreprocessing.shuffle_split(df, columns, int(split_value), int(k_value))
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
    info = dataChecking.get_train_test_info(train_data, test_data)

    return render(
        request,
        'preprocess/data-sampling.html',
        {
            'info': info,
            'sampling_algorithm': select_sampling_algorithm
        }
    )


@csrf_exempt
def execute_data_sampling(request):
    data_sampling = json.loads(request.POST.get("data_sampling"))
    train_data = request.session['train_data']
    df_train = dataChecking.json_to_df(train_data)
    target = "output"

    sampling_algorithm = request.POST.get("sampling_algorithm")
    if sampling_algorithm == "":
        sampling_algorithm = "OriginalData"

    info = dataChecking.get_target_info(df_train, target)
    samplig_list = []

    for algorithm in data_sampling:
        balanced_data = dataPreprocessing.set_imbalanced_data(algorithm, df_train)
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


@csrf_exempt
def data_sampling_apply(request):
    try:
        select_algorithm = request.POST.get("select_algorithm")
        train_data = request.session['train_data']
        train_df = dataChecking.json_to_df(train_data)
        test_data = request.session['test_data']
        test_df = dataChecking.json_to_df(test_data)
        project_id = request.session['project_id']

        MLSampling.objects.filter(project_id=project_id).update(sampling_algorithm=select_algorithm)

        if select_algorithm != "OriginalData":
            balanced_data = dataPreprocessing.set_imbalanced_data(select_algorithm, train_df)
            df_apply = dataChecking.dataframe_concat(balanced_data, test_df)
        else:
            df_apply = dataChecking.dataframe_concat(train_df, test_df)

        request.session['df_apply'] = dataChecking.df_to_json(df_apply)
        result = json.dumps({"result": "success"})
        return HttpResponse(result, content_type='application/json')

    except Exception as e:
        print(e)
        result = json.dumps({"result": "error"})
        return HttpResponse(result, content_type='application/json')

################################## 데이터셋 전처리 END ##################################
################################## 학습 데이터 생성 START ##################################

@csrf_exempt
def data_create(request):
    df_apply = request.session['df_apply']
    df = dataChecking.json_to_df(df_apply)
    data_info = dataChecking.get_data_info(df)

    project_id = request.session['project_id']
    process_models = Process.objects.filter(project_id=project_id).values()
    process_size = len(process_models)

    sampling_models = MLSampling().get_info(project_id)

    per_page = 9
    list_models = MLSampling.objects.exclude(project_id=project_id)
    dataset_size = len(list_models.values())

    paginator = Paginator(list_models, per_page)
    page_num = request.POST.get("page_num", 1)
    try:
        current_page = paginator.page(page_num)
    except PageNotAnInteger:
        current_page = paginator.page(1)
    except EmptyPage:
        current_page = paginator.page(paginator.num_pages)
    list_data = current_page.object_list

    max_index = 5
    if paginator.num_pages <= max_index:
        start_index = 1
        end_index = paginator.num_pages
    else:
        start_index = page_num - 2
        set_num = 0
        if start_index < 1:
            set_num = start_index - 1
            start_index = 1
        end_index = page_num + 2 - set_num
        if end_index > paginator.num_pages:
            set_num = end_index - paginator.num_pages
            start_index = start_index - set_num
            end_index = paginator.num_pages
    page_range = range(start_index, end_index + 1)

    links = []
    for pr in page_range:
        if pr == current_page.number:
            links.append('<li class="page-item active"><a href="javascript:void(0);" class="page-link">%d</a></li>' % pr)
        else:
            links.append('<li class="page-item"><a href="javascript:go_page(%d);" class="page-link">%d</a></li>' % (pr, pr))

    return render(
        request,
        'preprocess/data-create.html',
        {
            'project_id': project_id,
            'data_info': data_info,
            'process_size': process_size,
            'sampling_models': sampling_models,
            'dataset_size': dataset_size,
            'list_models': list_data,
            'page_links': links
        }
    )


@csrf_exempt
def load_dataset_list(request):
    project_id = request.session['project_id']
    per_page = 9
    list_models = MLSampling.objects.exclude(project_id=project_id)
    dataset_size = len(list_models.values())

    paginator = Paginator(list_models, per_page)
    page_num = int(request.POST.get("page_num", 1))
    try:
        current_page = paginator.page(page_num)
    except PageNotAnInteger:
        current_page = paginator.page(1)
    except EmptyPage:
        current_page = paginator.page(paginator.num_pages)
    list_data = current_page.object_list

    max_index = 5
    if paginator.num_pages <= max_index:
        start_index = 1
        end_index = paginator.num_pages
    else:
        start_index = page_num - 2
        set_num = 0
        if start_index < 1:
            set_num = start_index - 1
            start_index = 1
        end_index = page_num + 2 - set_num
        if end_index > paginator.num_pages:
            set_num = end_index - paginator.num_pages
            start_index = start_index - set_num
            end_index = paginator.num_pages
    page_range = range(start_index, end_index+1)

    links = []
    for pr in page_range:
        if pr == current_page.number:
            links.append(
                '<li class="page-item active"><a href="javascript:void(0);" class="page-link">%d</a></li>' % pr)
        else:
            links.append(
                '<li class="page-item"><a href="javascript:go_page(%d);" class="page-link">%d</a></li>' % (pr, pr))

    return render(
            request,
            'preprocess/dataset-list.html',
            {
                'dataset_size': dataset_size,
                'list_models': list_data,
                'page_links': links
            }
        )