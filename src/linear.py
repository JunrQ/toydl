

from module import Module



class Linear(Module):
  """Fully-connected layer.
  """

  def __init__(self, in_features, out_features, bias=True):
    self.in_features = in_features
    self.out_features = out_features
    self.bias = bias

    self.W = TensorType((out_features, in_features))
    self.b = TensorType((output_fature, )) if bias else None

  def forward(self, x):
    assert len(x.shape) == 2
    assert x.shape[1] = self.in_features

    x = self.W.matmul(x)
    if self.bias:
      return x.add(self.b)
    else:
      return x



# class Add(Module):



# class MatMul(Module):



# class ScalarMul(Module):

