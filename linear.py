
from toydl.module import Module
from toydl.tensor import NumpyTensor


class Linear(Module):
  """Fully-connected layer.
  """
  def __init__(self, in_features, out_features, bias=True,
               is_input_layer=False):
    self.in_features = in_features
    self.out_features = out_features
    self.bias = bias
    self.is_input_layer = is_input_layer

    self.W = NumpyTensor(shape=(out_features, in_features))
    self.b = NumpyTensor(shape=(out_features, ), init_value=0.01) if bias else None

  def parameters(self):
    res = [self.W]
    if self.b:
      res.append(self.b)
    return res

  def forward(self, x):
    assert len(x.shape) == 2
    assert x.shape[1] == self.in_features
    self._X = x
    x = self.W.matmul(x.transpose()) # (o, i) (i, b) -> (o, b)
    x = x.transpose() # (b, o)
    if self.bias:
      return x + self.b.reshape((1, -1))
    else:
      return x

  def backward(self, grad):
    # grad : (b, o)
    self.W.grad = grad.transpose().matmul(self._X).data # (o, b) (b, i)
    if self.b:
      self.b.grad = grad.data.sum(axis=0)
    if self.is_input_layer:
      return
    # (i, o) (o, b) -> (i, b) -> (b, i)
    input_grad = self.W.transpose().matmul(grad.transpose()).transpose()
    return input_grad


class Add(Module):
  def forward(self, a, b):
    return a + b

  def backward(self, grad):
    return grad, grad


class ReLU(Module):
  def forward(self, x):
    self.m = (x > 0).astype(x.type)
    return self.m * x

  def backward(self, grad):
    return grad * self.m
