"""
관련 decorator 정의
"""
from ..viz import graph
from copy import deepcopy


def for_all_methods(decorator):
    # 해당 클래스의 모든 메서드에 decorator 적용
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


def _graph_debugging_decorator(fn):
    # 연산할 때마다 그래프에 노드 추가하는 decorator

    def decorate(*args, **kwargs):
        r = fn(*args, **kwargs)
        graph_debugging(*args)
        return r
    return decorate


def graph_debugging(*args):
    # 그래프에 연산 노드 추가
    def get_id_name(arg):
        return str(id(arg)), type(arg).__name__
    args = list(args)

    # a
    a_id, a_name = get_id_name(args[1])
    graph.node(a_id, a_name + r'\n' + str(args[1].value.shape))
    if len(args) >= 3:
        # b
        b_id, b_name = get_id_name(args[2])
        graph.node(b_id, b_name + r'\n' + str(args[2].value.shape))

    # self
    self_id, self_name = get_id_name(args[0])
    graph.node(self_id, self_name + r'\n' + str(args[0].value.shape))

    graph.edge(a_id, self_id)
    if len(args) >= 3:
        graph.edge(b_id, self_id)

    args[0].graph = deepcopy(graph)
