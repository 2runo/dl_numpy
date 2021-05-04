"""
활성화 함수
(도함수를 정의할 필요 없음)
"""
from ..core import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    c = math.max(x, axis=1, keepdim=True)
    exp_x = math.exp(x-c)
    sum_exp_x = math.sum(exp_x, axis=1, keepdim=True)
    y = exp_x / sum_exp_x
    return y


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def relu(x):
    return x * (x.value > 0)


def gelu(x):
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
