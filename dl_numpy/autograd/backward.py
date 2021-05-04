"""
역전파 구현
"""
import numpy as np
from .derivative import DERIVATIVES
from ..utils.autograd import fit_shape


class Backward:
    a, b = None, None

    def backward(self, signal=None, grad_dict=None, acc_grad=True):
        # 역전파 진행
        if signal is None:
            signal = 1
            if self.value.ndim != 0:
                raise RuntimeError("backward is only available for scalar outputs.")
        if grad_dict is None:
            grad_dict = {}

        try:
            # signal accumulation
            signal = np.sum(list(grad_dict[id(self)].values()), axis=0)
        except KeyError:
            pass

        if type(self.a).__name__ == "Tensor":
            # 단말 노드
            if self.a.var:
                a_signal = self.da(signal)  # 편미분
                a_signal = fit_shape(self.a.value, a_signal)
                if id(self.a) in grad_dict:  # gradient 저장
                    grad_dict[id(self.a)][id(self)] = a_signal
                else:
                    grad_dict[id(self.a)] = {id(self): a_signal}
        else:
            # a 쪽으로 역전파
            a_signal = self.da(signal)  # 편미분
            a_signal = fit_shape(self.a.value, a_signal)
            if id(self.a) in grad_dict:  # gradient 저장
                grad_dict[id(self.a)][id(self)] = a_signal
            else:
                grad_dict[id(self.a)] = {id(self): a_signal}
            grad_dict = self.a.backward(a_signal, grad_dict=grad_dict, acc_grad=False)  # 다음 노드

        if self.b is not None:
            if type(self.b).__name__ == "Tensor":
                # 단말 노드
                if self.b.var:
                    b_signal = self.db(signal)  # 편미분
                    b_signal = fit_shape(self.b.value, b_signal)
                    if id(self.b) in grad_dict:  # gradient 저장
                        grad_dict[id(self.b)][-id(self)] = b_signal
                    else:
                        grad_dict[id(self.b)] = {-id(self): b_signal}
            else:
                # b 쪽으로 역전파
                b_signal = self.db(signal)  # 편미분
                b_signal = fit_shape(self.b.value, b_signal)
                if id(self.b) in grad_dict:  # gradient 저장
                    grad_dict[id(self.b)][-id(self)] = b_signal
                else:
                    grad_dict[id(self.b)] = {-id(self): b_signal}
                grad_dict = self.b.backward(b_signal, grad_dict=grad_dict, acc_grad=False)  # 다음 노드

        if acc_grad:
            # gradient accumulation
            grad_dict = {k: np.sum(list(v.values()), axis=0) for k, v in grad_dict.items()}
        return grad_dict

    def da(self, signal):
        # a에 대해 편미분
        return signal * self.get_derivative(0)

    def db(self, signal):
        # b에 대해 편미분
        return signal * self.get_derivative(1)

    def get_derivative(self, i):
        # 미분값 반환
        name = type(self).__name__
        return DERIVATIVES[name][i](self.a.value, self.b.value if self.b is not None else None)
