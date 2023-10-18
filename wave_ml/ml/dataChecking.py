import pandas as pd
import warnings

from collections import Counter
from . import dataChart, dataPreprocessing

warnings.filterwarnings("ignore")


# csv 파일 Dataframe 으로 변환
def file_to_df(file):
    df = pd.read_csv(file)
    return df


def files_to_df(files):
    data = []
    for file in files:
        try:
            df = pd.read_csv(f"wave_ml/media/{file['project_file']}", encoding='euc-kr', index_col=None, header=0)
        except Exception as e:
            print(e)
            df = pd.read_csv(f"wave_ml/media/{file['project_file']}", encoding='utf-8', index_col=None, header=0)
        data.append(df)

    return pd.concat(data, axis=0, ignore_index=True)


# json 데이터를 Dataframe 으로 변환
def json_to_df(df_json):
    df = pd.read_json(df_json)
    return df


# Dataframe 데이터를 json 으로 변환
def df_to_json(df_data):
    df = df_data.to_json(orient='columns')
    return df


# Dataframe 컬럼 확인
def data_column_process(df, column):
    columns = list(df.columns)
    select_column = columns.__getitem__(0) if column is None else column
    return list(df.columns), select_column


# Data 타입 확인
def get_data_type(df, column):
    data_type = df[column].dtype.name
    return data_type


# column list
def get_column_list(df):
    return df.columns


# DataFrame 기본정보
def get_data_info(df, target):
    df_shape = df.shape
    columns = get_column_list(df)
    result = {'count': df_shape[0], 'size': len(columns), 'columns': columns, 'target': target}
    return result


def get_train_test_info(train_data, test_data, target):
    train_shape = train_data.shape
    test_shape = test_data.shape
    columns = get_column_list(train_data)
    result = {'train_count': train_shape[0], 'size': len(columns), 'test_count': test_shape[0], 'columns': columns, 'target': target}
    return result


def get_target_info(df, target):
    count = Counter(df[target])
    result = {}
    if count[0] >= count[1]:
        result['large'] = count[0]
    else:
        result['large'] = count[1]
    result['train'] = df.shape[0]
    result['percent_0'] = int(count[0]/result['large']*100)
    result['percent_1'] = int(count[1]/result['large']*100)
    result['target_0'] = count[0]
    result['target_1'] = count[1]

    return result


# 데이터 통계 확인
def data_desc(df, column):
    desc = df[column].describe(include='all')
    result = desc.to_dict()

    # 유일값
    result['unique'] = len(df[column].unique())

    # 결측값
    result['missing'] = df[column].isnull().sum()

    return result


# 데이터 통계 전체 확인
def data_statistics(df):
    df_desc = pd.DataFrame(df.describe(include='all').transpose())
    df_desc = df_desc.drop(['25%', '50%', '75%'], axis=1)

    temp_dict = {}
    for col in list(df.columns):
        temp_dict[col] = len(df[col].unique())

    # 유일값
    df_unique = pd.DataFrame(temp_dict, index=["unique"]).transpose()

    # 결측값
    df_temp = df.isnull().sum().to_dict()
    df_missing = pd.DataFrame(df_temp, index=["missing"]).transpose()

    # 데이터 타입
    df_type = pd.Series(df.dtypes, name='type')

    result = pd.concat([df_desc, df_unique, df_missing, df_type], axis=1)

    return result.to_dict(orient='split')


# 데이터 상세조회 정보
def data_detail(df, column):
    list = sorted(df[column].unique())
    return list


# 카테고리 컬럼 여부
def is_data_category(df, column) -> bool:
    unique_count = len(df[column].unique())
    if unique_count <= 10:
        return True
    else:
        return False


# 데이터 최빈값 확인
def data_freq(df, column):
    freq = df[column].mode(dropna=False)
    return freq[0]


# 전처리 적용 전 미리보기
def data_preview(df, column, target):
    result: dict = {'value_list': data_detail(df, column), 'desc': data_desc(df, column),
                    'main_graph': dataChart.main_graph(df, column, target),
                    'outlier': dataPreprocessing.detect_outlier(df, column),
                    'box_graph': dataChart.box_plot_graph(df, column, 750, 500),
                    'corr_graph': dataChart.scatter_graph(df, column, target),
                    'freq': data_freq(df, column)}
    return result


def get_process_name(process):
    if process == "missing":
        return "결측치 처리"
    elif process == "outlier":
        return "이상치 처리"
    elif process == "replace":
        return "문자열 통합"
    elif process == "datatype":
        return "데이터 유형 변경"
    elif process == "dummy":
        return "가변수화"
    elif process == "scaler":
        return "데이터 정규화"


def dataframe_concat(df1, df2):
    df_merged = pd.concat([df1, df2], ignore_index=True, sort=False)
    return df_merged
