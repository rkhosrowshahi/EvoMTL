FlattenLoRA with rank=16: z dim = 19220 (conv=4880, linear=14310, other=30) | model params = 42350
[EvoMTL] MOEA (parameter sharing): num_parameters=42350 | num_dimensions=19220

FlattenLoRA with rank=8: z dim = 9660 (conv=2440, linear=7190, other=30) | model params = 42350
[EvoMTL] MOEA (parameter sharing): num_parameters=42350 | num_dimensions=9660

FlattenLoRA with rank=1: z dim = 1295 (conv=305, linear=960, other=30) | model params = 42350
[EvoMTL] MOEA (parameter sharing): num_parameters=42350 | num_dimensions=1295
## Persistent Pareto centers

`evo_moea: multi_center_pareto_openes` learns multiple persistent networks with
vector antithetic OpenES and coordinated, bounded spreading. See the
[algorithm and runnable MultiMNIST example](moea/MULTI_CENTER_PARETO_OPENES.md)
for configuration, checkpoint semantics, tests and stochastic limitations.

## RDUST

Select `evo_moea: rdust` for a reverse sequential DUST iteration: estimate one layer's gradient, update that layer
immediately, then estimate the next upstream layer using the updated downstream network. It uses the existing
`evo_dust_*` and `evo_es_*` settings. Its standalone [estimator and training loop](moea/rdust.py) own the
reverse pass; DUST keeps its simultaneous update loop. With W&B enabled, initialization is logged at step 0.
An evaluation frequency of 50 then tests generations 1, 50, 100, ... and the final generation.

## LW-DUST

Select `evo_moea: lw_dust` for layer-wise DUST: train one layer at a time, from the first layer to the last,
with `evo_dust_inner_steps` optimizer steps per layer before moving on. Every inner step samples fresh
mini-batches, estimates that layer's gradient with DUST's activation jitter (its `evo_dust_*` draw count),
and updates only that layer. Give one count for every layer, or one per layer in forward order (`0` freezes a
layer). One iteration is one pass over the layers and uses `sum(T)` mini-batches: set `evo_iterations` to
the batch budget divided by `sum(T)`. Normalization affines and other parameters outside the weight layers are
their own groups, trained by DUST's weight-space ES.

`evo_moea: rlw_dust` (RLW-DUST) is the same method in reverse: it starts from the last layer (the task heads)
and moves to the first. `evo_dust_inner_steps` is still given in forward order, like `evo_dust_draws`.

## CADUST

Select `evo_moea: cadust` to use curvature-aware sampling with DUST on the summed task loss. CADUST starts with
standard Gaussian activation noise. It estimates a curvature proxy from the uncentered second moment of DUST's
per-sample activation gradient estimates, discounts the previous proxy each batch, and keeps its leading directions
using an incremental thin SVD. A shared layer uses one covariance over its concatenated call outputs.

For a site with `D` activations per sample, the next batch draws independent standard Gaussians `z` and `u` and
samples `sqrt(alpha) z + U (sqrt(lambda) * u)`. The isotropic variance is `alpha = 1 - subspace_fraction`. The
remaining variance is distributed across the columns of `U` in proportion to their curvature eigenvalues, so
`alpha * D + sum(lambda) = D`. CADUST applies the inverse covariance to the reward-weighted activation estimate
before the weight update. The update uses the same forward evaluations as DUST and stores a thin factor per site.

| Setting | Default | Effect |
| --- | --- | --- |
| `evo_cadust_rank` | `8` | Maximum directions per site, capped at its activation width. `0` recovers DUST. |
| `evo_cadust_decay` | `0.9` | Discount on the previous curvature proxy. `0` uses the current batch only. |
| `evo_cadust_subspace_fraction` | `0.5` | Fraction of noise variance in the learned directions. `0` recovers DUST; must be below `1`. |

The usual `evo_dust_*` and `evo_es_*` settings still apply. `evo_es_grad_cos_freq: 0` disables the optional
backprop diagnostic.

## Joint-DUST

Select `evo_moea: joint_dust` to jitter several layers in the same forward pass instead of one at a time. The
layers are split, in forward order, into groups of the sizes in `evo_dust_groups` (default: one group with every
layer). Each draw adds independent noise to every layer of a group and credits all of them with the same per-sample
loss. Each layer then correlates that loss with its own noise and divides by its own noise scale. The other layers'
contributions cancel in expectation but add variance. A layer after a jittered layer sees a different input at
every draw, so its weight gradient uses each draw's own input; the first layer of a group uses the cached clean
input. `evo_dust_antithetic: true` (Joint-A-DUST) flips the jitter of a whole group together in pairs.

The draw counts (`evo_pop_size`, `evo_dust_allocation`, `evo_dust_draws`) are per group and count forward
evaluations. Groups of one layer reproduce DUST (A-DUST with `evo_dust_antithetic`). Every pass draws noise for
every layer of its group and maps each later layer's weight gradient per draw, so a generation costs more time than
DUST's at the same number of forward evaluations.

With `evo_dust_balance: true` (Balanced-Joint-DUST) each layer gets its own jitter scale, chosen so that every layer
of a group moves the loss by the same amount. This stops layers with a large effect on the loss, such as the task
heads, from swamping the estimates of layers with a small one, such as an early convolution. Each layer's gradient
energy is estimated without backprop from the same draws: the draws are split into two halves, and the product of the
two half estimates is unbiased. A moving average over generations (`evo_dust_balance_decay`) smooths it. The layer
with the least energy keeps `evo_es_sigma`, and the others get smaller scales, at least 5% of it.

## RFC

Select `evo_moea: rfc` for readout-frame credit, the Noise speedrun's readout credit carried over to an encoder
and its decoders. Each generation moves every coordinate of the shared representation `h` by `+-evo_es_sigma`
and reruns only the decoders. The central differences give each objective's per-sample credit at `h`, exact up to
`O(sigma^2)` and free of sampling noise. The same coordinate differences on the outputs of every layer inside the
decoders give their exact weight gradients. Each encoder layer gets independent jitter draws and reruns only the
encoder. Each draw is scored by the linearized readout `<credit, h_jittered - h>`, so one draw scores every task
loss, and the score needs no baseline (no chunk centering).

`evo_pop_size` (and `evo_dust_allocation` / `evo_dust_draws`) counts encoder passes and is spent on the encoder
layers only. The frame adds `2 (d_h + decoder output sizes)` decoder passes per
image, `evo_rfc_frame_chunk` coordinates per pass. RFC needs the HPS architecture (one encoder output shared by
the decoders, with or without `--shared_decoder`) and suits a narrow representation with small decoders.
