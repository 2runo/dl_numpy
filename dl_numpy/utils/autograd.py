"""
`autograd`에서 사용하는 도구들
"""
import numpy as np


def merge_dict(a: dict, b: dict, overwrite=False) -> dict:
    # 두 dictionary를 합친다.
    # ex) f({'a':1, 'b':2}, {'a':2, 'c':1}) -> {'a':3, 'b':2, 'c':1}
    # ex) f({'a':1, 'b':2}, {'a':2, 'c':1}, overwrite=True) -> {'a':2, 'b':2, 'c':1}
    for k, v in b.items():
        if overwrite:
            a[k] = v
        else:
            if k in a:
                a[k] = a[k] + v
            else:
                a[k] = v
    return a


def fit_shape(a, b):
    # a.shape == b.shape이도록 sum(b)
    while a.shape != b.shape:
        if a.ndim == b.ndim:
            b = np.sum(b, axis=np.argmax((np.array(a.shape) != np.array(b.shape)).astype(int)), keepdims=True)
        else:
            b = np.sum(b, axis=0)
    return b
