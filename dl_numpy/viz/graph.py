"""
계산 그래프
"""
from graphviz import Digraph


class CalcGraph:
    def __init__(self):
        self.dot = Digraph(graph_attr={'rankdir': 'LR'})
        self.history = []

    def node(self, *args, **kwargs):
        # 노드 추가
        if args[0] in self.history:  # 노드 중복 방지
            return False
        else:
            self.history.append(args[0])
        if args[1].startswith('Tensor'):
            kwargs['shape'] = 'rect'
            kwargs['color'] = '#fada5e'
            kwargs['style'] = 'filled'
        else:
            kwargs['shape'] = 'rect'
            kwargs['style'] = 'rounded'
        return self.dot.node(*args, **kwargs)

    def edge(self, *args, **kwargs):
        # 간선 추가
        return self.dot.edge(*args, **kwargs)

    def render(self, fn, format='svg', **kwargs):
        # 파일로 저장
        return self.dot.render(fn, format=format, **kwargs)
