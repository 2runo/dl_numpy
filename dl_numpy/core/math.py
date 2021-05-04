"""
Math
"""
import numpy as np
from .decorators import for_all_methods
from .base import check_tensor_decorator, Tensor, Add, Mul, Div, Pow
from .calc import Log, Sqrt, Sum, Matmul, Max, Min, Erf


@for_all_methods(check_tensor_decorator())
class Math:
    def add(self, x, y):
        return Add(x, y)

    def mul(self, x, y):
        return Mul(x, y)

    def div(self, x, y):
        return Div(x, y)

    def pow(self, x, y):
        return Pow(x, y)

    def log(self, x):
        return Log(x)

    def exp(self, x):
        return Pow(Tensor(np.e, var=False), x)

    def sqrt(self, x):
        return Sqrt(x)

    def sum(self, x, axis=None, keepdim=False):
        return Sum(x, axis=axis, keepdim=keepdim)

    def mean(self, x, axis=None, keepdim=False):
        n = x.value.shape[0 if axis is None else axis]
        return Sum(x, axis=axis, keepdim=keepdim) / n

    def matmul(self, x, y):
        return Matmul(x, y)

    def max(self, x, axis=None, keepdim=False):
        return Max(x, axis=axis, keepdim=keepdim)

    def min(self, x, axis=None):
        return Min(x, axis=axis)

    def erf(self, x):
        return Erf(x)
