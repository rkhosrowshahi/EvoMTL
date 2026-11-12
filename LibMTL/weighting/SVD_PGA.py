from LibMTL.weighting.abstract_weighting import AbsWeighting


class SVD_PGA(AbsWeighting):
    r"""Principal Gradient Aggregation by SVD (SVD-PGA).

    The update is the average of the principal directions of the task gradients (the left singular vectors of the
    Jacobian), each one unit norm and oriented towards the side where most of the gradients lie, as the ``SVDPGA``
    aggregator of `TorchJD <https://github.com/TorchJD/torchjd>`_. The computation is that of the MO-SVD-PGA-OpenES
    runner (:mod:`LibMTL.evomtl.moea.pga`), here on the backpropagated gradients.

    Args:
        PGA_rank (int, default=None): The number of principal directions kept (None: all of them).
        PGA_scale ({'none', 'mean_norm'}, default='none'): The length of the update: the average of the unit-norm
            directions (``none``), or that times the mean norm of the task gradients (``mean_norm``).
    """

    def __init__(self):
        super().__init__()

    def backward(self, losses, **kwargs):
        from LibMTL.evomtl.moea.pga import svd_pga_weights  # here: the ES package imports the weightings

        if self.rep_grad:
            raise ValueError('No support for method SVD_PGA with representation gradients (rep_grad=True)')
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward')
        scale = None if kwargs['PGA_scale'] == 'none' else kwargs['PGA_scale']
        weights = svd_pga_weights(grads @ grads.T, rank=kwargs['PGA_rank'], scale=scale)
        self._backward_new_grads(weights, grads=grads)
        return weights.detach().cpu().numpy()
