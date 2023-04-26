from django import template

register = template.Library()


@register.filter
def get_value_from_dict(dictionary, key):
    return dictionary.get(key, '')


@register.filter
def get_value_from_list(lst, index):
    return lst[index]
