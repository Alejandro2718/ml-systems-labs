import math


class Value:
    __slots__ = ('data', 'grad', 'parents', 'op', '_backward')

    def __init__(self, data, parents=(), op=''):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.op = op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def backward():
            if exponent != 0:
                self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def backward():
            self.grad += out.data * out.grad

        out._backward = backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def backward():
            self.grad += out.grad / self.data

        out._backward = backward
        return out

    def sigmoid(self):
        return (1 + (-self).exp()) ** -1

    def backward(self):
        ordered, visited = [], set()

        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    visit(parent)
                ordered.append(node)

        visit(self)
        for node in ordered:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(ordered):
            node._backward()
