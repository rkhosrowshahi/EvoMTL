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
