
def parse_tensors(objs, target_list, predicate):
  """
  """
  if predicate(objs):
    target_list.append(objs)
  elif isinstance(objs, (list, tuple)):
    for o in objs:
      parse_tensors(o, target_list, predicate)
  elif isinstance(objs, dict):
    for v in objs.values():
      parse_tensors(v, target_list, predicate)
