"""RFC (readout-frame credit): forward-only gradient estimates measured exactly at the readout, sampled below it.

The Noise speedrun records (https://github.com/qlabs-eng/noise) went past DUST by replacing random draws with exact
forward differences wherever the network after a stream is cheap to rerun ("readout credit"), and by crediting
deeper layers against that measured readout instead of rerunning the whole network. RFC carries this over to a
network seen as an encoder, one shared representation ``h = encoder(x)`` (a tensor), and the decoders that turn
``h`` into the task losses (one per task, or one module shared by every task). One generation, on one mini-batch:

1. A clean forward caches every layer's input (a layer: a module with a weight matrix, as in DUST), ``h`` and the
   per-sample objectives ``L_m[b]`` (the summed task loss, or every task loss with an aggregator).
2. Readout frame. Every coordinate ``j`` of ``h`` (flattened per sample) is moved by ``+-sigma`` at every sample and
   only the decoders are rerun. The central differences give the readout credit
   ``c_m[b, j] = (L_m(h + sigma e_j)[b] - L_m(h - sigma e_j)[b]) / (2 sigma) ~ dL_m[b] / dh[b, j]``, exact up to
   ``O(sigma^2)``: a full frame leaves no sampling noise. With an aggregator all ``K`` task credits come from the
   same reruns.
3. Decoder layers. The same coordinate differences on the outputs of every layer inside the decoders (decoder-only
   reruns from the cached ``h``; a shared decoder's calls are its coordinates together) give their output errors,
   and the exact local linear map with the cached inputs gives their weight gradients, as in DUST.
4. Encoder layers. Each gets independent jitter ``y + sigma a_n`` (as DUST), rerunning only the encoder. A draw is
   scored by the linearized readout, ``dL_{m,n}[b] = <c_m[b], h_n[b] - h[b]>`` (the first-order change of ``L_m``,
   as DUST rewards its attention internals by alignment with the attention-output error), so no decoder runs and
   one draw scores every objective. ``e_m[b] = 1 / (N sigma) sum_n dL_{m,n}[b] a_n[b]`` over the ``N`` draws, then
   the exact local linear map. The score is zero without jitter, so it needs no baseline: no chunk centering (and
   none of its ``1 - 1 / chunk`` shrinkage), and every draw is its own direction (antithetic pairs would halve the
   directions at the same cost, which loses at equal evaluations on the conv layers).
5. Parameters outside the weight layers: two-sided weight-space ES, as in DUST. One optimizer step on ``theta``.

Cost. The draw counts (``pop_size``, ``allocation``, ``draws``) count encoder passes of the batch and are spent on
the encoder layers only; ``chunk`` is the draws per pass. The frame adds
``2 (d_h + sum of the decoder layers' output sizes)`` decoder passes per sample, in passes of ``2 * frame_chunk``
moves. Decoder passes are cheap when the decoders are small next to the encoder, as in MultiMNIST (a 50-wide
representation and two 10-way linear heads: 140 decoder passes, no encoder pass), and the frame is unsuited to a
wide representation or a deep decoder.

RFC does not chain credit from one encoder layer to the next (that would be backpropagation by finite differences):
every encoder layer is measured by its own jitter. It needs an architecture that calls ``encoder`` once and feeds its
single output to every decoder (LibMTL's HPS, with or without a shared decoder).

References:
- DUST: :mod:`~LibMTL.evomtl.moea.dust`; https://qlabs.sh/research/dust
- Noise: https://github.com/qlabs-eng/noise (identity-path and readout-cycle credit)
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Dict, Tuple

import torch

from LibMTL.architecture.HPS import HPS

from .dust import DUST, run_dust


class RFC(DUST):
    """DUST's hooks and local maps, with readout-frame credit for ``h`` and the decoder layers (module docstring)."""

    name = "RFC"

    def __init__(self, *args: Any, frame_chunk: int = 64, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, HPS):
            raise NotImplementedError("RFC needs one encoder output shared by the decoders (the HPS architecture).")
        self.frame_chunk = int(frame_chunk)
        self.rep = None
        self.hooks.append(self.model.encoder.register_forward_hook(self._rep_hook))

    def _rep_hook(self, module, args, output):
        if not torch.is_tensor(output):
            raise NotImplementedError("RFC needs the encoder to return one tensor.")
        if self.capturing:
            self.rep = output

    def _setup(self) -> None:
        """Layers the forward calls; decoder layers get the frame, encoder layers the draws."""
        self.layers = {n: m for n, m in self.layers.items() if n in self.inputs}
        covered = {id(p) for m in self.layers.values() for p in m.parameters(recurse=False)}
        self.free = [p for p in self.model.parameters() if id(p) not in covered]
        self.decoder_layers = [n for n in self.inputs if n.startswith("decoders.")]
        widths = {n: sum(math.prod(s[1:]) for s in self.shapes[n]) for n in self.inputs if n.startswith("encoder.")}
        self.draws = self._allocate(widths, list(widths))
        frame = math.prod(self.rep.shape[1:]) + sum(math.prod(s[1:]) for n in self.decoder_layers
                                                    for s in self.shapes[n])
        print(f"[EvoMTL] {self.name}: readout frame of {frame} coordinates (h and "
              f"{', '.join(self.decoder_layers)}): {2 * frame} decoder passes | {sum(self.draws.values())} encoder "
              f"draws ({self.allocation}): "
              + ", ".join(f"{n}={d}" for n, d in self.draws.items())
              + (f" | weight-space ES on {sum(p.numel() for p in self.free)} other weights: {self.es_draws} pairs"
                 if self.free else ""), flush=True)

    def _decode(self, rep: torch.Tensor, gts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Objectives ``[M, N]`` of the decoders alone on representations ``rep`` (``N`` = len(rep))."""
        self.calls = {}
        preds = self.trainer.process_preds({t: self.model.decoders[t](rep) for t in self.trainer.task_name})
        rows = [self.sample_loss[t](preds[t], gts[t]) for t in self.trainer.task_name]
        return self.W @ torch.stack([r.reshape(len(r), -1).mean(1) for r in rows])

    def _frame(self, shapes: list, gts: Dict[str, torch.Tensor], evaluate) -> list:
        """Coordinate central differences of the objectives over the flattened per-sample tensors of ``shapes`` (one
        per call, joined): ``evaluate(moves, gts)`` takes one ``[2n B, ...]`` tensor per call (``+sigma`` on ``n``
        coordinates, then ``-sigma`` on the same) and returns ``[M, 2n B]``. Returns ``[M, B, ...]`` per call."""
        B = shapes[0][0]
        sizes = [math.prod(s[1:]) for s in shapes]
        d = sum(sizes)
        out = None
        for start in range(0, d, self.frame_chunk):
            n = min(self.frame_chunk, d - start)
            eye = torch.zeros(n, d, device=self.trainer.device)
            eye[torch.arange(n), torch.arange(start, start + n)] = self.sigma
            moves = torch.cat([eye, -eye]).repeat_interleave(B, 0)  # [2n B, d]: move-major, then sample
            parts = [p.reshape(-1, *s[1:]) for p, s in zip(moves.split(sizes, 1), shapes)]
            g = {t: v.repeat(2 * n, *(1,) * (v.dim() - 1)) for t, v in gts.items()}
            obj = evaluate(parts, g).view(-1, 2, n, B)
            slope = ((obj[:, 0] - obj[:, 1]) / (2 * self.sigma)).transpose(1, 2)  # [M, B, n]
            out = slope if out is None else torch.cat([out, slope], 2)
        return [e.reshape(*e.shape[:2], *s[1:]) for e, s in zip(out.split(sizes, 2), shapes)]

    def _encoder_layer(self, name: str, x: torch.Tensor, credit: torch.Tensor,
                      grads: Dict[torch.nn.Parameter, torch.Tensor]) -> None:
        """Independent jitter of one encoder layer, each draw scored by the linearized readout ``credit``."""
        M, B, _ = credit.shape
        n = self.chunk
        h0 = self.rep.reshape(1, B, -1)
        errors = [torch.zeros(M, *s, device=credit.device) for s in self.shapes[name]]
        xr = x.repeat(n, *(1,) * (x.dim() - 1))
        for _ in range(self.draws[name] // n):
            noise = self._sample_noise(name, n * B)
            self.jitter = {name: [self.sigma * a for a in noise]}
            self.calls = {}
            try:
                h = self.model.encoder(xr).reshape(n, B, -1)
            finally:
                self.jitter = {}
            change = torch.einsum("mbd,nbd->mnb", credit, h - h0)  # first-order loss change of every draw
            for e, a in zip(errors, noise):
                e += torch.einsum("mnb,nb...->mb...", change, a.view(n, B, *a.shape[1:])) / (
                    self.sigma * self.draws[name])
        self._reassemble(self.layers[name], errors, self.inputs[name], grads)

    @torch.no_grad()
    def estimate(self, inputs: torch.Tensor, gts: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gradient estimates ``J`` (``[M, D]``) of the M objectives (mean over the batch) and the task losses."""
        task_losses = self._capture(inputs, gts)
        rep = self.rep
        B = len(rep)
        grads: Dict[torch.nn.Parameter, torch.Tensor] = {}
        # readout credit at h: decoders only, on h moved coordinate by coordinate
        reps = rep.repeat(2 * self.frame_chunk, *(1,) * (rep.dim() - 1))  # move-major copies of h, sliced per pass
        credit = self._frame([rep.shape], gts, lambda m, g: self._decode(reps[:len(m[0])] + m[0], g))[0]
        credit = credit.reshape(*credit.shape[:2], -1)  # [M, B, d_h]
        for name in self.decoder_layers:  # decoder layers: their output coordinates, decoders only

            def jittered(moves, g, name=name):
                self.jitter = {name: moves}
                try:
                    return self._decode(reps[:len(moves[0])], g)
                finally:
                    self.jitter = {}

            self._reassemble(self.layers[name], self._frame(self.shapes[name], gts, jittered), self.inputs[name],
                             grads)
        for name in self.draws:
            self._encoder_layer(name, inputs, credit, grads)
        if self.free:
            self._weight_space(inputs, gts, grads, credit.shape[0])
        J = torch.cat([grads[p].reshape(credit.shape[0], -1) if p in grads else
                       torch.zeros(credit.shape[0], p.numel(), device=rep.device) for p in self.model.parameters()],
                      dim=1)
        return J / B, task_losses.mean(1)


def run_rfc(trainer: Any, train_dataloaders: Any, frame_chunk: int = 64, method: str = "RFC",
            **kwargs: Any) -> Dict[str, Any]:
    """RFC: :func:`~.dust.run_dust` (every other argument) with the :class:`RFC` estimator; ``pop_size`` /
    ``draws`` count encoder passes for the encoder layers, ``chunk`` draws per pass, ``frame_chunk`` coordinates are
    moved per decoder pass. With ``aggregate`` (MO-*-RFC) every task is credited from the same frame and draws."""
    return run_dust(trainer, train_dataloaders, method=method,
                    _estimator=partial(RFC, frame_chunk=frame_chunk), **kwargs)
