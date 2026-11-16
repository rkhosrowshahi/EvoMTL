"""Joint DUST: DUST with several sites jittered in the same forward pass and credited with one shared loss.

DUST (:mod:`.dust`) jitters one site at a time, holding the others clean, so every forward pass estimates one site.
Joint DUST splits the sites, in forward order, into groups (``groups``: their sizes) and jitters every site of a group
in the same pass, with noise drawn independently per site, per sample and per draw. Every site of the group is credited
with the same per-sample loss of that pass and divides by its own noise:

    e_l[b] = 1 / (N sigma) sum_n (L_n[b] - c[b]) a_{l,n}[b]   ~   dL[b] / dy_l[b]

over the ``N`` evaluations of the group, with the baseline ``c`` the mean over the ``chunk`` draws of a forward pass
(as in DUST) or, with ``antithetic``, the clean loss. To first order ``L_n - L_0 = sum_m sigma g_m . a_{m,n}``; the
noise of different sites is independent, so the other sites' terms vanish in expectation and the estimate stays
unbiased, but they add the variance ``d_l / N sum_{m != l} ||g_m||^2`` (equal sigma): the price of joint jitter. In
return each pass serves every site of the group, so at the same number of forward passes every site gets more draws.

Inputs. A site after a jittered site sees a jittered input, different at every draw, so its weight gradient is the
local linear map of every draw's own error and input,

    G_l = 1 / (N sigma) sum_n sum_b (L_n[b] - c[b]) a_{l,n}[b] x_{l,n}[b]^T

(the correlation for a convolution, a scatter-add for an embedding), not the map of the averaged error and the clean
input that DUST uses. The inputs are recorded during each draw. This estimates the gradient of the expected loss of
the jittered network, which approaches the clean loss as sigma shrinks. The per-draw map, and drawing noise for every
site of the group in every pass, make a generation slower than DUST's at the same number of forward evaluations.

Antithetic (``antithetic``, Joint-A-DUST): each draw is evaluated with the jitter of every site of the group flipped
together, ``+sigma a`` and ``-sigma a``. With the clean baseline each member of the pair contributes
``(L_+- - L_0) (+-a) x_+-^T``, which is ``(L_+ - L_-) a x^T`` when the input is the same in both. The flip cancels the
even-order terms of the loss, the quadratic cross-site ones among them, but not the first-order interference.

Groups of one site are DUST (A-DUST with ``antithetic``: the clean baseline cancels in a pair whose input is the same)
for the same noise, the sites taken in forward order. The first site of a group sees the clean input at every draw, so
its errors are summed over the draws and mapped once with the cached input, as in DUST. The draw counts
(``pop_size``, ``allocation``, ``draws``) are per group, in forward evaluations; a group's width for ``"width"`` and
``"log"`` is the summed output size of its sites. The parameters outside the sites get DUST's weight-space ES.
Single-objective: the rewards come from the summed task loss.

Balanced noise (``balance``, Balanced-Joint-DUST). With a noise scale ``sigma_l`` per site the interference on site
``l`` is ``d_l / N sum_{m != l} (sigma_m / sigma_l)^2 ||g_m||^2``: a site whose output barely moves the loss (an early
convolution) drowns in the loss changes caused by the others (the task heads). Balancing sets the scales so that every
site of a group moves the loss equally, ``sigma_l^2 E_l`` the same for all, with ``E_l = sum_b ||g_l[b]||^2``; the
interference ratio of every site is then the number of other sites. The energies are estimated forward-only from the
same draws: the draws of a group are split into two halves by forward pass, each half gives an independent estimate
``e_l^(h)`` of the output error, and ``sum_b e_l^(1)[b] . e_l^(2)[b]`` is unbiased for ``E_l`` once divided by the
halves' shares and by the chunk centering's shrinkage ``(1 - 1 / chunk)^2``. The estimates are averaged over
generations: equally over the first ``1 / (1 - balance_decay)``, then with an exponential moving average of decay
``balance_decay``, so the noisy estimates of the first generation (equal scales) fade fast. The site with the smallest
energy keeps ``sigma``, the others get ``sigma sqrt(E_min / E_l)``, at least ``MIN_SCALE sigma``; the first generation
uses ``sigma`` everywhere. No site is jittered more than with ``sigma``, so the smoothing of the loss does not grow.

References:
- DUST: :mod:`~LibMTL.evomtl.moea.dust`; https://qlabs.sh/research/dust
"""

from __future__ import annotations

import functools
import itertools
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .dust import DUST, run_dust

MIN_SCALE = 0.05  # smallest noise scale of a balanced site, relative to sigma
BALANCE_LOG_EVERY = 50  # generations between printed noise scales


class JointDUST(DUST):
    """DUST whose draws jitter a group of sites at once (``groups``: group sizes in forward order; ``None``: all
    sites in one group), optionally with balanced per-site noise scales (``balance``)."""

    def __init__(self, *args: Any, groups: Optional[list] = None, balance: bool = False, balance_decay: float = 0.9,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.name = ("Balanced-" if balance else "") + ("Joint-A-DUST" if self.antithetic else "Joint-DUST")
        self.group_sizes = [int(g) for g in groups] if groups else None
        if self.group_sizes and min(self.group_sizes) < 1:
            raise ValueError(f"Joint-DUST group sizes must be positive, got {groups}.")
        if not 0 <= balance_decay < 1:
            raise ValueError(f"Joint-DUST balance_decay must be in [0, 1), got {balance_decay}.")
        self.balance, self.balance_decay = bool(balance), float(balance_decay)
        self.groups: Dict[str, list] = {}  # group name -> its sites, in forward order
        self.recording: set = set()
        self.drawn: Dict[str, list] = {}  # site -> input of every call (current draw)
        self.energy: Dict[str, float] = {}  # site -> moving average of its estimated output-gradient energy
        self.scale: Dict[str, float] = {}  # site -> its noise scale relative to sigma
        self.updates: Dict[str, int] = {}  # group -> balance updates so far

    def _hook(self, name: str):
        dust_hook = super()._hook(name)

        def hook(module, args, output):
            if name in self.recording:
                self.drawn.setdefault(name, []).append(args[0])
            return dust_hook(module, args, output)
        return hook

    def _setup(self) -> None:
        self.sites = {n: m for n, m in self.sites.items() if n in self.inputs}
        covered = {id(p) for m in self.sites.values() for p in m.parameters(recurse=False)}
        self.free = [p for p in self.model.parameters() if id(p) not in covered]
        order = [n for n in self.inputs if n in self.sites]  # sites in the order the forward pass calls them
        sizes = self.group_sizes or [len(order)]
        if sum(sizes) != len(order):
            raise ValueError(f"Joint-DUST group sizes {sizes} must add up to the {len(order)} sites "
                             f"({', '.join(order)}).")
        bounds = [0, *itertools.accumulate(sizes)]
        self.groups = {"+".join(order[i:j]): order[i:j] for i, j in zip(bounds, bounds[1:])}
        widths = {g: sum(math.prod(s[1:]) for n in sites for s in self.shapes[n]) for g, sites in self.groups.items()}
        self.draws = self._allocate(widths, list(self.groups))
        if self.balance and any(d < 2 * self.chunk for g, d in self.draws.items() if len(self.groups[g]) > 1):
            raise ValueError(f"Balanced Joint-DUST needs at least two forward passes (2 * chunk = {2 * self.chunk} "
                             f"draws) per group of several sites, got {self.draws}.")
        total = sum(self.draws.values())
        print(f"[EvoMTL] {self.name}: {total} forward evaluations per generation"
              + (f" ({total // 2} antithetic pairs)" if self.antithetic else "") + f" ({self.allocation}): "
              + ", ".join(f"[{g}]={d}" for g, d in self.draws.items())
              + (f" | weight-space ES on {sum(p.numel() for p in self.free)} other weights: {self.es_draws} pairs"
                 if self.free else ""), flush=True)

    def _estimate_site(self, group: str, clean: torch.Tensor, x: torch.Tensor,
                       y: Dict[str, torch.Tensor], grads: Dict[nn.Parameter, torch.Tensor]) -> None:
        """Estimate the gradients of every site of ``group`` from draws that jitter them all at once."""
        sites = self.groups[group]
        K, B = clean.shape
        n = self.chunk
        m = n // 2 if self.antithetic else n
        draws = self.draws[group]
        passes = draws // n
        balance = self.balance and len(sites) > 1
        sigma = {s: self.sigma * self.scale.get(s, 1.0) for s in sites}
        # the first site of the group sees the clean input at every draw (if called once): sum its errors over the
        # copies and map them once with the cached input, as DUST does
        first = sites[0] if len(self.shapes[sites[0]]) == 1 else None
        first_error = torch.zeros(K, *self.shapes[sites[0]][0], device=clean.device)
        # balance: output errors of every site from the even and the odd forward passes
        halves = {s: [[torch.zeros(K, *shape, device=clean.device) for shape in self.shapes[s]] for _ in range(2)]
                  for s in sites} if balance else {}
        for p in range(passes):
            # unit noise per copy of the batch; antithetic: +a on the first m copies, -a on the last m
            unit = {s: [torch.cat([a, -a]) if self.antithetic else a for a in self._sample_noise(s, m * B)]
                    for s in sites}
            self.jitter = {s: [sigma[s] * u for u in units] for s, units in unit.items()}
            self.recording, self.drawn = set(sites) - {first}, {}
            try:
                loss = (self.W @ self.losses(x, y)).view(K, n, B)
            finally:
                self.jitter, self.recording = {}, set()
            change = loss - (clean.unsqueeze(1) if self.antithetic else loss.mean(1, keepdim=True))
            for s in sites:
                reward = change / (sigma[s] * draws)
                if s == first or balance:  # per-sample output errors, summed over the copies
                    sums = [torch.einsum("knb,nb...->kb...", reward, u.view(n, B, *u.shape[1:])) for u in unit[s]]
                    for h, e in zip(halves.get(s, [[], []])[p % 2], sums):
                        h += e
                if s == first:
                    first_error += sums[0]
                else:  # each copy's error times its own input
                    errors = [reward.reshape(K, n * B, *(1,) * (u.dim() - 1)) * u for u in unit[s]]
                    self._reassemble(self.sites[s], errors, self.drawn[s], grads)
        if first is not None:
            self._reassemble(self.sites[first], [first_error], self.inputs[first], grads)
        if balance:
            # each half holds its share of the passes, and the chunk centering shrinks it by 1 - 1 / chunk:
            # E[e^(0) . e^(1)] = share_0 share_1 shrink^2 E_l
            shrink = 1.0 if self.antithetic else 1 - 1 / n
            shares = ((passes + 1) // 2 / passes) * (passes // 2 / passes) * shrink ** 2
            t = self.updates.get(group, 0) + 1
            decay = min(self.balance_decay, 1 - 1 / t)  # plain mean until the moving-average window is filled
            for s in sites:
                energy = float(sum((a * b).sum() for a, b in zip(*halves[s]))) / shares
                self.energy[s] = decay * self.energy.get(s, 0.0) + (1 - decay) * energy
            top = max(self.energy[s] for s in sites)
            floor = {s: max(self.energy[s], MIN_SCALE ** 2 * top) for s in sites}  # noisy or negative estimates
            low = min(floor.values())
            self.scale.update({s: math.sqrt(low / floor[s]) for s in sites})
            self.updates[group] = t
            if t % BALANCE_LOG_EVERY == 1:
                print(f"[EvoMTL] {self.name} noise scales x sigma (energy): "
                      + ", ".join(f"{s}={self.scale[s]:.3f} ({self.energy[s]:.3g})" for s in sites), flush=True)


def run_joint_dust(trainer: Any, train_dataloaders: Any, groups: Optional[list] = None, antithetic: bool = False,
                   balance: bool = False, balance_decay: float = 0.9, **kwargs: Any) -> Dict[str, Any]:
    """Joint DUST: :func:`~.dust.run_dust` (every argument) with the sites of each group jittered together;
    ``groups`` are the group sizes in forward order (``None``: one group), ``antithetic`` flips a group's jitter in
    pairs (``chunk`` even), ``balance`` gives every site the noise scale that equalizes its effect on the loss."""
    name = ("Balanced-" if balance else "") + ("Joint-A-DUST" if antithetic else "Joint-DUST")
    return run_dust(trainer, train_dataloaders, antithetic=antithetic, method=name,
                    _estimator=functools.partial(JointDUST, groups=groups, balance=balance,
                                                 balance_decay=balance_decay), **kwargs)
