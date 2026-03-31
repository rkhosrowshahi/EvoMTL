import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class IEW(AbsWeighting):
    r"""InEqual Weighting (IEW).

    The loss weight for each task is given as predefined preference.

    """
    def __init__(self):
        super(IEW, self).__init__()
        
    def backward(self, losses, **kwargs):
        loss_weights = kwargs['loss_weights']
        if not isinstance(loss_weights, torch.Tensor):
            loss_weights = torch.Tensor(loss_weights).to(losses.device)
        loss = torch.mul(losses, loss_weights).sum() / loss_weights.sum()
        loss.backward()
        return loss_weights.cpu().numpy()