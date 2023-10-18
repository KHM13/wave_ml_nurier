from datetime import datetime
from wave_ml.fds.common.LoggingHandler import LoggingHandler
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from multipledispatch import dispatch
import math


class CommonUtil:
    __FASTDATE_TYPE_0 = "%Y-%m-%d_%H:%M:%S.%f"
    __FASTDATE_TYPE_1 = "%Y-%m-%d_%H:%M:%S"
    __FASTDATE_TYPE_2 = "%Y%m%d%H%M%S"
    __FASTDATE_TYPE_3 = "%Y%m%d"
    __FASTDATE_TYPE_4 = "%H%M%S"

    def __init__(self):
        self.logger = LoggingHandler(f"{prop().get_result_log_file_path()}{self.now_type3()}_output", "a", "DEBUG")
        self.logger = self.logger.get_log()

    def log_time(self):
        return self.__now(self.__FASTDATE_TYPE_0)

    def now(self):
        return self.__now(self.__FASTDATE_TYPE_1)

    def now_type2(self):
        return self.__now(self.__FASTDATE_TYPE_2)

    def now_type3(self):
        return self.__now(self.__FASTDATE_TYPE_3)

    def now_type4(self):
        return self.__now(self.__FASTDATE_TYPE_4)

    def __now(self, format_type: str):
        return datetime.now().strftime(format_type)

    @staticmethod
    def get_int_from_string(value: str) -> int:
        if value.isnumeric():
            return int(value)
        else:
            return 0

    @staticmethod
    def get_float_from_string(value: str) -> float:
        value = value.replace(".", "")
        if value.isnumeric():
            return float(value)

    @staticmethod
    def convert_string(array_string) -> str:
        return ', '.join(array_string)

    @staticmethod
    def check_null_string(input: str) -> str:
        if not(input and input.strip()):
            return ''
        elif input.strip() == 'null':
            return ''
        else:
            input.strip()

    @staticmethod
    def get_model_training_drop_column_list(mediacode: str) -> str:
        if mediacode == "IB":
            return prop().get_model_training_drop_column_ib_list()
        elif mediacode == "SB":
            return prop().get_model_training_drop_column_sb_list()
        else:
            return ""


# 페이징 validation
def validate_page(page):
    if page == "" or page is None or int(page) <= 0:
        page = 1
    return page


# 페이징 처리
@dispatch(object, int, object)
def paginator(page, item_per_page, item_list):
    return paginator(page, item_per_page, item_list, 5)


@dispatch(object, int, object, int)
def paginator(page, item_per_page, item_list, limit):
    page = validate_page(page)

    # 페이징 처리를 위한 Paginator 객체 생성
    paginator_obj = Paginator(item_list, item_per_page)

    # 현재 페이지에 해당하는 아이템 리스트 반환
    item_obj = paginator_obj.get_page(page)

    # 페이지 버튼의 범위 제한
    page_btn_range = setting_page_btn_range(limit, paginator_obj, item_obj)

    # 페이지 버튼 생성
    links = []
    for pr in page_btn_range:
        if int(pr) == int(page):
            links.append(
                '<li class="page-item active"><a href="javascript:void(0)" class="page-link">%d</a></li>' % pr)
        else:
            links.append(
                '<li class="page-item"><a href="javascript:go_page(%d)" class="page-link">%d</a></li>' % (pr, pr))

    return item_obj, links


# 페이지 버튼 범위제한
def setting_page_btn_range(limit, paginator_obj, item_obj):
    if item_obj.number <= math.ceil(limit/2):
        page_btn_range = range(1, min((limit + 1), paginator_obj.num_pages + 1))
    else:
        if item_obj.number <= paginator_obj.num_pages - math.floor(limit/2):
            page_btn_range = range(max(item_obj.number - math.floor(limit / 2), 1), min(item_obj.number + math.ceil(limit / 2), paginator_obj.num_pages + 1))
        else:
            page_btn_range = range(max(paginator_obj.num_pages - (limit - 1), 1), paginator_obj.num_pages + 1)

    return page_btn_range
