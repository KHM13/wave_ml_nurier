from django.core.paginator import Paginator
from django.test import TestCase
import math
import pandas as pd

# Create your tests here.
# objects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9]
# p = Paginator(objects, 2)
# item_obj = p.get_page(15)
# limit = 10
#
# print(f"ceil : {math.ceil(limit / 2)}")
# print(f"floor : {math.floor(limit / 2)}")
#
# if item_obj.number <= math.ceil(limit / 2):
#     page_btn_range = range(1, min(limit + 1, p.num_pages + 1))
#     print(f"1 : {page_btn_range}")
# else:
#     if item_obj.number <= p.num_pages - math.floor(limit / 2):
#         page_btn_range = range(max(item_obj.number - math.floor(limit / 2), 1), min(item_obj.number + math.ceil(limit / 2), p.num_pages + 1))
#         print(f"2 : {page_btn_range}")
#     else:
#         page_btn_range = range(max(p.num_pages - (limit - 1), 1), p.num_pages + 1)
#         print(f"3 : {page_btn_range}")

# columns =['age', 'sex', 'cp', 'chol', 'fbs', 'output']
# file = open('D:\wave_ml_nurier\wave_ml\data\heart.csv', 'r', encoding='EUC-KR')
# df = pd.read_csv(file)
#
# columns_list = df.columns
# print(columns_list)
#
# columns_list = columns_list.drop('output')
# print(columns_list)

list_dup = []

list_dup.append("l1")
list_dup.append("elasticnet")
list_dup.append("elasticnet")
list_dup.append("elasticnet")
list_dup.append("l2")

for inx, l in enumerate(list_dup):
    print(inx)
    print(l)
