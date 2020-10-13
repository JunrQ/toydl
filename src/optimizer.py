
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

  def _step(self):
    raise NotImplementedError()

  def step(self):
    self._step()

  def zero_grad(self):
    for param in self._parameters:
      param.zero_grad()



class SGD(Optimizer):
  """Implements stochastic gradient descent.
  """
  def __init__(self, parameters, lr):
    super(SGD, self).__init__(parameters, lr)

  def _step(self):
    for param in self._parameters:
      param.add(-self._base_lr * param.grad)
