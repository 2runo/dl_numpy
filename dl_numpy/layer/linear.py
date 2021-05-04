"""
Linear (W·x + b) 모듈
"""
from ..core.base import Tensor
from ..core import math
from ..utils.layer import init_weights
from ..layer.module import Module


class Linear(Module):
    def __init__(self, in_units, out_units, use_bias=True, activation=None):
        # units : 유닛 수
        # use_bias : 바이어스 사용 여부
        self.params = {
            'W': Tensor(init_weights((in_units, out_units))),
        }

        self.use_bias = use_bias
        if self.use_bias:
            self.params['b'] = Tensor(init_weights((out_units,)))

        self.activation = activation

    def forward(self, x):
        _out = math.matmul(x, self.params['W'])
        if self.use_bias:
            _out = _out + self.params['b']
        if self.activation is not None:
            _out = self.activation(_out)
        return _out
