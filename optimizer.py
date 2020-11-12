
from .tensor import Tensor


class OptimizerBase(object):
  """Base class for Optimizer."""


class Optimizer(OptimizerBase):
  """Optimizer class.
  """
  def __init__(self, parameters, lr, scheduler=None,
               weight_decay=0):
    assert isinstance(parameters, (list, tuple))
    for param in parameters:
      assert isinstance(param, Tensor)
    
    self._parameters = parameters
    self._base_lr = lr
    self.scheduler = scheduler
    self.wd = weight_decay

    self._steps = 0

  def _step(self):
    raise NotImplementedError()

  def step(self):
    self._step()
    self._steps += 1

  def zero_grad(self):
    for param in self._parameters:
      param.zero_grad()


class SGD(Optimizer):
  """Implements stochastic gradient descent.
  """
  def __init__(self, parameters, lr, scheduler=None,
               weight_decay=0):
    super(SGD, self).__init__(parameters, lr,
                              scheduler=scheduler, weight_decay=weight_decay)

  def _step(self):
    if self.scheduler is not None:
      lr = self.scheduler(self._base_lr, self._steps)
    else:
      lr = self._base_lr
    for param in self._parameters:
      param.add(-lr * (param.grad + self.wd * param))
