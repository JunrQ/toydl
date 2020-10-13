

import numpy as np


from .tensor import Tensor



class NumpyTensor(Tensor):

  def __init__(self, name, inputs=[], outputs=[],
               shape=None, init_value=None):
    """

    Parameters
    ----------

    """
    assert shape is not None or init_value is not None

    if shape is not None:
      if init_value is not None:
        init_value = np.ones(shape) * init_value
      else:
        init_value = np.random.rand(shape)
    assert isinstance(init_value, np.ndarray), "NumpyTensor only accepts np.ndarray"
 
    self._array = init_value
    self._shape = init_value.shape
    self._grad = np.zeros_like(init_value)
    super(NumpyTensor, self).__init__(name)

  def zero_grad(self):



  @property
  def grad(self):
    """
    """


  @grad.setter
  def grad(self):


  @property
  def data(self):



  @data.setter
  def data(self):




  def add(self, other):



  def __add__(self, other):
    return self.add(other)

  def matmul(self, other):
  

  def elemul(self, other):
  


  def __mul__(self, other):
    return self.elemul(other)


  def scalar_mul


    