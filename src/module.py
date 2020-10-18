





class ModuleBase(object):
  """Base Class for Module."""



class Module(ModuleBase):
  def __repr__(self):



  def __call__(self, *args, **kwargs):

    for arg in args:



    for k, arg in kwargs.items():

    

    outputs = self.forward(*args, **kwargs)


    # For all nodes in outputs, it should has a dependence of
    # nodes in inputs, we can build a graph according to this info



