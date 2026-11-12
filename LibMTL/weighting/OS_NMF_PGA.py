from LibMTL.weighting.abstract_weighting import AbsWeighting


class OS_NMF_PGA(AbsWeighting):
    r"""Principal Gradient Aggregation by orthogonal semi-NMF (OS-NMF-PGA).

    The task gradients are approximated by ``W H`` with orthonormal columns in ``W`` (in the span of the gradients)
    and nonnegative entries in ``H``, starting from the SVD-PGA basis; the update is the average of the columns of
    ``W``, as the ``OSNMFPGA`` aggregator of `TorchJD <https://github.com/TorchJD/torchjd>`_. The computation is that
    of the MO-OS-NMF-PGA-OpenES runner (:mod:`LibMTL.evomtl.moea.pga`), here on the backpropagated gradients.

    Args:
        PGA_rank (int, default=None): The number of columns of ``W`` (None: the rank of the gradients).
        PGA_scale ({'none', 'mean_norm'}, default='none'): The length of the update: the average of the unit-norm
            columns (``none``), or that times the mean norm of the task gradients (``mean_norm``).
    """

    def __init__(self):
        super().__init__()

    def backward(self, losses, **kwargs):
        from LibMTL.evomtl.moea.pga import os_nmf_pga_weights  # here: the ES package imports the weightings

        if self.rep_grad:
            raise ValueError('No support for method OS_NMF_PGA with representation gradients (rep_grad=True)')
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward')
        scale = None if kwargs['PGA_scale'] == 'none' else kwargs['PGA_scale']
        weights = os_nmf_pga_weights(grads @ grads.T, rank=kwargs['PGA_rank'], scale=scale)
        self._backward_new_grads(weights, grads=grads)
        return weights.detach().cpu().numpy()
