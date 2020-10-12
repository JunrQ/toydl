
from .tensor import Tensor


class OptimizerBase(object):
  """Base class for Optimizer."""


class Optimizer(OptimizerBase):
  """Optimizer class.
  """
  def __init__(self, parameters, lr, weight_decay=None):
    assert isinstance(parameters, (list, tuple))
    for param in parameters:
      assert isinstance(param, Tensor)
    
    self._parameters = parameters
    self._base_lr = lr

  def step(self):
    raise NotImplementedError()

  def zero_grad(self):
    for param in self._parameters:
      param.zero_grad()



class SGD(Optimizer):
  """Implements stochastic gradient descent.
  """
  def __init__(self, parameters, lr):
    super(SGD, self).__init__(parameters, lr)

  def step(self):
    for param in self._parameters:
      param.__add__(-self._base_lr * param.grad)

