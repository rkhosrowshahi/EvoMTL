# The code of this file was partly taken from https://github.com/TorchJD/torchjd (see UPGrad.py for its MIT license).

import torch

from LibMTL.weighting.abstract_weighting import AbsWeighting
from LibMTL.weighting.UPGrad import _compute_gramian, _normalize, _project_weights, _regularize


class DualProj(AbsWeighting):
    r"""Dual Projection (DualProj).

    The average of the task gradients projected onto the dual cone of the gradients (the update with the smallest
    distance to the average that conflicts with no task), as the ``DualProj`` aggregator of `TorchJD
    <https://github.com/TorchJD/torchjd>`_ (`Jacobian Descent for Multi-Objective Optimization
    <https://arxiv.org/pdf/2406.16232>`_). It is UPGrad with one projection, that of the average, instead of the
    average of the projections of every gradient; same normalized and regularized Gramian.
    """

    def __init__(self):
        super().__init__()

    def backward(self, losses, **kwargs):
        norm_eps = kwargs['DualProj_norm_eps']
        reg_eps = kwargs['DualProj_reg_eps']

        if self.rep_grad:
            raise ValueError('No support for method DualProj with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward')

        pref_vector = torch.ones(len(grads), device=grads.device, dtype=grads.dtype) / len(grads)
        G = _regularize(_normalize(_compute_gramian(grads), norm_eps), reg_eps)
        weights = _project_weights(pref_vector, G)

        self._backward_new_grads(weights, grads=grads)
        return weights.detach().cpu().numpy()
