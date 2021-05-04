"""
`layer`에서 사용하는 도구들
"""
import numpy as np


def init_weights(size: tuple) -> np.ndarray:
    return np.random.uniform(-0.1, 0.1, size=size)
