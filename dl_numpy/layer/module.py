"""
커스텀 모듈 구현
"""


class Module:
    params = dict()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def apply_grad(self, grad_dict: dict, lr: float):
        # gradient 적용
        for k, tensor in self.params.items():
            grad = grad_dict[id(tensor)]
            self.params[k].value -= grad * lr
