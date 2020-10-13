


class Node(object):
  def __init__(self, value, inputs):




class Graph(object):

  def __init__(self):

    self._nodes = []



  def add_node(self):

  



class TensorBase(object):
  """Base class for Tensor."""


class Tensor(TensorBase):
  _GLOBAL_TENSORS_MAP = {}

  @staticmethod
  def get_all_tensors(cls):
    return _GLOBAL_TENSORS_MAP

  def __init__(self, name, inputs=[], outputs=[]):
    if name in _GLOBAL_TENSORS_MAP:
      raise ValueError("%s already exists" % name)

    _GLOBAL_TENSORS_MAP[name] = self
    
    # Build graph info
    self._node_index = self._get_node_index()
    self._inputs = []
    self._outputs = []


  def _get_node_index(self):




  def get_input_nodes(self):




  def get_output_nodes(self):




