import numpy as np


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


def batch_wrapper(iterator, batch_size, transform, shuffle=True):
  r = []
  while True:
    index = list(range(len(iterator)))
    if shuffle:
      np.random.shuffle(index)
    for i in index:
      if len(r) == batch_size:
        res = []
        for j in range(len(r[0])):
          a = [np.expand_dims(_r[j], 0) for _r in r]
          res.append(np.concatenate(a, axis=0))
        yield res
        r = []
      else:
        sample = list(iterator[i])
        if transform:
          sample[0] = transform(sample[0])
        r.append(sample)
