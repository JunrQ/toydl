from toydl.module import Module
from toydl.tensor import NumpyTensor


class BaseLoss(Module):
  """Base class for Loss"""


class MSELoss(BaseLoss):
  def forward(self, pred, gt):
    if pred.shape != gt.shape:
      raise ValueError("Different shape %s vs %s" % 
                       (pred.shape, gt.shape))
    self._diff = gt - pred
    loss = self._diff ** 2
    loss = loss.mean() / 2.0
    return loss

  def backward(self):
    pred_grad = -self._diff
    gt_grad = self._diff
    return pred_grad, gt_grad


class CrossEntropyLoss(BaseLoss):
  def forward(self, pred, logit):
    b, c = pred.shape
    self.p = pred.softmax(axis=1)
    self.logit = logit
    m_ce = NumpyTensor(self.p.data[list(range(b)), logit.data]).log().mean()
    return -m_ce

  def backward(self):
    b, _ = self.p.shape
    zeros = NumpyTensor.zeros(self.p.shape)
    zeros.data[list(range(b)), self.logit.data] = 1
    self.p.data = self.p.data - zeros.data
    return self.p
