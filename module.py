

from tensor import Tensor
from utils import parse_tensors


"""
Tensor should compose a graph as a node.

In order to build the graph, we should give every tensor a
index built.

And to achieve that, we need reimplement its `__init__` or
`__new__` function.

"""

class Node(object):
  def __init__(self, name=None, version=None):




class Graph(object):

  _instance = None

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super(cls, Graph).__new__(*args, **kwargs)
    return cls._instance

  def __init__(self):
    self._nodes = []
    self._G = {}




  def add_node(self, node):


  def _sort(self):




class ModuleBase(object):
  """Base Class for Module."""



class Module(ModuleBase):
  # def __repr__(self):
  def __init__(self, name):
    self.name = name

  def parameters(self):
    return []

  def __call__(self, *args, **kwargs):
    # Build Graph
    inputs_nodes = []
    pred = lambda x : isinstance(x, Tensor)
    parse_tensors(args, inputs_nodes, pred)
    parse_tensors(kwargs, inputs_nodes, pred)

    outputs = self.forward(*args, **kwargs)

    outputs_nodes = []
    parse_tensors(outputs, outputs_nodes, pred)

    for n in inputs_nodes:
      # add output


    # For all nodes in outputs, it should has a dependence of
    # nodes in inputs, we can build a graph according to this info



