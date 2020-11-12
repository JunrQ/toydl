import sys
sys.path.insert(0, '../../')

import logging
import argparse
import shutil
import time
import os
import numpy as np

from torchvision import datasets

from toydl.linear import Linear, ReLU
from toydl.loss import CrossEntropyLoss
from toydl.optimizer import SGD
from toydl.utils import batch_wrapper
from toydl.tensor import NumpyTensor


parser = argparse.ArgumentParser(description='Toydl MNIST')
parser.add_argument('--dir', type=str, default='./log', help='Save directory')
args = parser.parse_args()
save_dir = args.dir

msg = []
logger = logging.getLogger('toydl-mnist')
logger.setLevel(logging.INFO)
if not os.path.isdir(args.dir):
  msg.append('%s not exist, make it' % args.dir)
  os.mkdir(args.dir)
log_file_path = os.path.join(args.dir, 'log.log')
if os.path.isfile(log_file_path):
  target_path = log_file_path + '.%s' % time.strftime("%Y%m%d%H%M%S")
  msg.append('Log file exists, backup to %s' % target_path)
  shutil.move(log_file_path, target_path)
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


class Trainer(object):
  def __init__(self, num_classes=10, lr=0.1, logger=logging):
    self.logger = logger

    self.l1 = Linear(28*28, 256, is_input_layer=True)
    self.a1 = ReLU()
    self.l2 = Linear(256, 128)
    self.a2 = ReLU()
    self.l3 = Linear(128, 64)
    self.a3 = ReLU()
    self.l4 = Linear(64, 32)
    self.a4 = ReLU()
    self.l5 = Linear(32, num_classes)

    self.layers = [self.l1, self.a1,
                   self.l2, self.a2,
                   self.l3, self.a3,
                   self.l4, self.a4,
                   self.l5]
    self.loss = CrossEntropyLoss()

    params = []
    for l in self.layers:
      params += l.parameters()
    self.optimizer = SGD(params, lr=lr,
                         # scheduler=lambda b, s: 0.9**s * b,
                         weight_decay=0)

  def forward(self, x):
    for l in self.layers:
      x = l(x)
    return x

  def backward(self, x, y):
    self.loss_value = self.loss(x, y)

    g = self.loss.backward()
    for l in self.layers[::-1]:
      g = l.backward(g)

  def update(self):
    self.optimizer.step()
    self.optimizer.zero_grad()

  def run(self, dataset, **kwargs):
    steps = 0
    for x, y in dataset:
      x = NumpyTensor(x)
      y = NumpyTensor(y)

      logit = self.forward(x)
      self.backward(logit, y)
      self.update()
      steps += 1
      if steps % kwargs['log_frequence'] == 0:
        loss_value = self.loss_value
        pred = logit.argmax(axis=1)
        acc_value = (pred == y).mean()
        self.logger.info("[Train] [Steps] %d | Loss: %.3f | Acc: %.3f" % 
                          (steps, loss_value, acc_value))
      
      if steps >= kwargs['max_steps']:
        break


if __name__ == '__main__':
  trainer = Trainer(num_classes=10, lr=0.0001, logger=logger)
  params = {
    'max_steps' : 30000,
    'batch_size' : 32,
    'log_frequence' : 50
  }
  dataset = datasets.MNIST('../data', train=True, download=True)
  dataset = batch_wrapper(dataset, batch_size=params['batch_size'],
                          transform=lambda x : ((np.asarray(x) - 127.5) / 127.5).flatten(),
                          shuffle=True)
  trainer.run(dataset, **params)
