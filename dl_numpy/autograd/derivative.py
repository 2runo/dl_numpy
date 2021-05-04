"""
미분
(기초 연산에 대한 편미분만 사전에 정의한다)
"""
import numpy as np


DERIVATIVES = {
    'Add': (  # 덧셈
        lambda a, b: np.ones_like(a),
        lambda a, b: np.ones_like(b),
    ),
    'Mul': (  # 곱셈
        lambda a, b: b,
        lambda a, b: a,
    ),
    'Div': (  # 나눗셈
        lambda a, b: 1/b,
        lambda a, b: -a/b**2,
    ),
    'Pow': (  # 거듭제곱
        lambda a, b: a**(b-1) * b,
        lambda a, b: a**b * np.log(a),
    ),
    'Log': (  # 자연로그
        lambda a, b: 1/a,
    ),
    'Sqrt': (  # 제곱근
        lambda a, b: 1/(2*np.sqrt(a)),
    ),
    'Sum': (  # 합계
        lambda a, b: np.ones_like(a),
    ),
    'Erf': (  # erf
        lambda a, b: 2/np.sqrt(np.pi) * np.e ** (-a**2),
    )
}
