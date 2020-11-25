import numpy as np
import matplotlib.pyplot as plt

def parse_log(path):
  losses = []
  steps = []
  with open(path, 'r') as f:
    for l in f.readlines():
      l = l.strip().split('|')
      loss = float(l[-2].strip().split()[-1])
      s = int(l[-3].strip().split()[-1])

      losses.append(loss)
      steps.append(s)
  
  return steps, losses


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == '__main__':
  steps, losses = parse_log('./log/log.log')
  losses = moving_average(losses, 8)
  plt.plot(steps[:len(losses)], losses)

  plt.xlabel("Step")
  plt.ylabel("Loss")
  plt.show()
