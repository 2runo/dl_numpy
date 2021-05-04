"""
연산 magic method 정의
"""
import numpy as np
from .decorators import for_all_methods, _graph_debugging_decorator
from ..autograd.backward import Backward
from ..utils.core import ValueRepr


def check_tensor_decorator(class_method=True):
    # 인자를 Tensor로 변환하는 decorator
    def _check_tensor_decorator(fn):
        def to_tensor(x):
            try:
                if x.is_tensor:
                    return x
            except AttributeError:
                pass
            return Tensor(x, var=False)

        def inner(*args, **kwargs):
            if class_method:
                args = [args[0]] + [to_tensor(i) for i in args[1:]]
            else:
                args = [to_tensor(i) for i in args]
            return fn(*args, **kwargs)

        return inner
    return _check_tensor_decorator


@for_all_methods(check_tensor_decorator())
class Calc(ValueRepr):
    value = None
    is_tensor = True

    def __add__(self, other):
        # self + other
        return Add(self, other)

    def __radd__(self, other):
        # other + self
        return Add(other, self)

    def __sub__(self, other):
        # self - other
        return Add(self, -other)

    def __rsub__(self, other):
        # other - self
        return Add(other, -self)

    def __neg__(self):
        # -self
        return -1 * self

    def __mul__(self, other):
        # self * other
        return Mul(self, other)

    def __rmul__(self, other):
        # other * self
        return Mul(other, self)

    def __truediv__(self, other):
        # self / other
        return Div(self, other)

    def __rtruediv__(self, other):
        # other / self
        return Div(other, self)

    def __pow__(self, other):
        # self ** other
        return Pow(self, other)

    def __rpow__(self, other):
        # other ** self
        return Pow(other, self)


class Tensor(Calc):
    def __init__(self, value, var=True):
        if type(value) == Tensor:
            self.value = value.value
        self.value = np.array(value)
        self.var = var  # 변수인가? (기울기 계산 대상인가?)


class Add(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.value = a.value + b.value


class Mul(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.value = a.value * b.value


class Div(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.value = a.value / b.value


class Pow(Calc, Backward):
    @_graph_debugging_decorator
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.value = a.value ** b.value
