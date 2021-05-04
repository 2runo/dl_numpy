"""
오차 함수
(도함수를 정의할 필요 없음)
"""
from ..core.base import check_tensor_decorator
from ..core import math


@check_tensor_decorator(class_method=False)
def mse(y_true, y_pred):
    return math.mean((y_true - y_pred) ** 2) * 0.5


@check_tensor_decorator(class_method=False)
def cross_entropy(y_true, y_pred, delta=1e-9):
    return -(math.mean(math.log(y_pred + delta) * y_true))
