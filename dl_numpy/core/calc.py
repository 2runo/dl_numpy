"""
연산 정의
"""
import numpy as np
from .decorators import _graph_debugging_decorator
from .base import Calc, Backward
from scipy.special import erf as scipy_erf


class Log(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a):
        # 자연 로그
        self.a, self.b = a, None
        self.value = np.log(a.value)


class Sqrt(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a):
        self.a, self.b = a, None
        self.value = np.sqrt(a.value)


class Sum(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a, axis=None, keepdim=False):
        self.a, self.b = a, None
        self.value = np.sum(a.value, axis=axis, keepdims=keepdim)


class Matmul(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.value = np.matmul(a.value, b.value)

    def da(self, signal):
        return np.matmul(signal, self.b.value.T)

    def db(self, signal):
        return np.matmul(self.a.value.T, signal)


class Max(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a, axis=None, keepdim=False):
        self.a, self.b = a, None
        self.mask = (a.value == np.max(a.value, axis=axis, keepdims=True)).astype(int)  # max 부분만 1, 나머지는 0
        self.value = np.max(a.value, axis=axis, keepdims=keepdim)

    def da(self, signal):
        return signal * self.mask

    def db(self, signal):
        return None


class Min(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a, axis=None, keepdim=False):
        self.a, self.b = a, None
        self.mask = (a.value == np.min(a.value, axis=axis, keepdims=True)).astype(int)  # max 부분만 1, 나머지는 0
        self.value = np.min(a.value, axis=axis, keepdims=keepdim)

    def da(self, signal):
        return signal * self.mask

    def db(self, signal):
        return None


class Erf(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a):
        self.a, self.b = a, None
        self.value = scipy_erf(a.value)
