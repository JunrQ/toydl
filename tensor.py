

import numpy as np


class Tensor(object):
  """Base class for Tensor"""


class ScalarTensor(Tensor):
  def __init__(self, value):
    self._array = value
  @property
  def data(self):
    return self._array
  @data.setter
  def data(self, value):
    self._array = value


def _wrapper_scalar(cls):
  def _inner_wrapper(func):
    def _func(*args, **kwargs):
      n_args = []
      for i, a in enumerate(args):
        if not isinstance(a, Tensor):
          n_args.append(cls(a))
        else:
          n_args.append(a)
      for k, v in kwargs.items():
        if not isinstance(v, Tensor):
          kwargs[k] = cls(v)
      return func(*n_args, **kwargs)
    return _func
  return _inner_wrapper


class NumpyTensor(Tensor):
  def __init__(self, array=None,
               shape=None,
               init_value=None):
    """

    """
    super(NumpyTensor, self).__init__()

    if array is not None:
      assert shape is None
      init_value = array

    if shape is not None:
      assert array is None
      if init_value is not None:
        init_value = np.ones(shape) * init_value
      else:
        init_value = np.random.rand(*shape)
 
    self._array = init_value
    self._shape = init_value.shape
    self._grad = np.zeros_like(init_value)

  def zero_grad(self):
    self._grad.fill(0)

  @property
  def grad(self):
    return self._grad

  @grad.setter
  def grad(self, g):
    if g.shape != self._shape:
      raise ValueError("Expected shape %s but got %s" % (self._shape, g.shape))
    self._grad = g

  @property
  def data(self):
    return self._array

  @data.setter
  def data(self, a):
    if a.shape != self._shape:
      raise ValueError("Expected shape %s but got %s" % (self._shape, a.shape))
    self._array = a

  @_wrapper_scalar(ScalarTensor)
  def add_(self, other):
    self._array += other.data
    return self

  @_wrapper_scalar(ScalarTensor)
  def add(self, other):
    return NumpyTensor(array=self.data + other.data)

  @_wrapper_scalar(ScalarTensor)
  def __add__(self, other):
    return self.add(other)

  @_wrapper_scalar(ScalarTensor)
  def sub_(self, other):
    self._array -= other.data
    return self

  @_wrapper_scalar(ScalarTensor)
  def sub(self, other):
    return NumpyTensor(array=self.data - other.data)

  @_wrapper_scalar(ScalarTensor)
  def __sub__(self, other):
    return self.sub(other)

  @_wrapper_scalar(ScalarTensor)
  def __mul__(self, other):
    return NumpyTensor(array=self.data * other.data)

  @_wrapper_scalar(ScalarTensor)
  def __equal__(self, other):
    return NumpyTensor(self.data == other.data)

  @_wrapper_scalar(ScalarTensor)
  def matmul(self, other):
    a = np.matmul(self.data, other.data)
    return NumpyTensor(a)

  @_wrapper_scalar(ScalarTensor)
  def matmul_(self, other):
    self._array = np.matmul(self.data, other.data)
    return self

  @_wrapper_scalar(ScalarTensor)
  def transpose(self):
    return NumpyTensor(np.transpose(self.data))

  @_wrapper_scalar(ScalarTensor)
  def transpose_(self):
    self._array = np.transpose(self._array)
    self._shape = self._array.shape
    return self

  @_wrapper_scalar(ScalarTensor)
  def __le__(self, v):
    a = self._array <= v.data
    return NumpyTensor(a)
  
  @_wrapper_scalar(ScalarTensor)
  def __lt__(self, v):
    a = self._array < v.data
    return NumpyTensor(a)

  @_wrapper_scalar(ScalarTensor)
  def __ge__(self, v):
    a = self._array >= v.data
    return NumpyTensor(a)

  @_wrapper_scalar(ScalarTensor)
  def __gt__(self, v):
    a = self._array > v.data
    return NumpyTensor(a)

  def reshape(self, shape):
    a = np.reshape(self._array, shape)
    return NumpyTensor(a)

  def reshape_(self, shape):
    self._array = self._array.reshape(shape)
    self._shape = self._array.shape
    return self

  @property
  def type(self):
    return self._array.dtype

  def astype(self, new_type):
    self._array = self._array.astype(new_type)
    return self

  @property
  def shape(self):
    return self._shape
