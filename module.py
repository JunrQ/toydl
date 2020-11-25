class ModuleBase(object):
  """Base Class for Module."""


class Module(ModuleBase):
  """Module Class actully used.
  """
  def parameters(self):
    return []

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)
