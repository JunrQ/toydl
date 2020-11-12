
from toydl.module import Module


class BaseLoss(Module):
  """Base class for Loss"""


class MSELoss(BaseLoss):
  def forward(self, pred, gt):
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



  def backward(self):
    
