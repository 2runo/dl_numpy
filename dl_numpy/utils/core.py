"""
`core`에서 사용하는 도구들
"""


class ValueRepr:

    def __str__(self):
        return f"{type(self).__name__}: {self.value}"

    def __repr__(self):
        return f"{type(self).__name__}: {self.value}"
