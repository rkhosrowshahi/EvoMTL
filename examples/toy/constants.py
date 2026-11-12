"""The toy problems and their initial points, keyed by short name (``problem.name`` of the configs), as in
``tests/trajectories/_constants.py`` of SimplexLab/TorchJD.

``PROBLEMS`` (classes in :mod:`problems`):

* ``vb`` / ``vb01``: the Valley-Bowl problem with scale 1 (Aligned-MTL, CAGrad) / 0.1 (Nash-MTL) on the first loss.
* The quadratic problems of SimplexLab/trajectories (``src/trajectories/constants.py``):

  - ``ewq`` (element-wise quadratic): ``f = (x_1^2, x_2^2)``; orthogonal gradients, Pareto set the origin.
  - ``cqf`` (convex quadratic form, Eq. 14 of https://arxiv.org/abs/2406.16232): two ill-conditioned bowls
    (condition numbers 100 and 300) rotated by +-pi/16, centred at ``(+-1, 0)``; curved Pareto set.
  - ``cqf2``: the same with explicit matrices (condition numbers 108 and 41); TorchJD's ``CQF``.
  - ``hqf`` (homogeneous quadratic form): one matrix, scaled by 1 and 10, centred at ``(1, 0)`` and ``(-10, 0)``;
    the Pareto set is the segment between the centres.
  - ``mn2`` / ``mn20`` (multinorm): ``a = (1, 10)`` / ``(1, ..., 20)``: round bowls of very different sizes, with 20
    objectives for ``mn20``.

``INITIAL_POINTS``: the default initial points of each problem (``[D]`` points; a config's ``inits`` replaces them).
``vb`` / ``vb01``: those of Nash-MTL's ``experiments/toy/trainer.py``; the quadratic problems: those of
SimplexLab/trajectories.

``SEARCH_PLOT_LIMS`` / ``VALUE_PLOT_LIMS`` and the rest: the settings of the figures of :mod:`utils` (the families of
methods, their colors and markers).
"""

import os
from math import pi, sqrt

import numpy as np

from problems import ElementwiseQuadratic, HomogeneousQuadraticForm, Multinorm, QuadraticForms, ValleyBowl, rotated_form

PROBLEM_SETTINGS = ("name", "loss_offset")  # keys of a config's ``problem``
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)  # parallel runs (initial points x seeds) of main.py

PROBLEMS = {
    "vb": ValleyBowl(),
    "vb01": ValleyBowl(scale=0.1),
    "ewq": ElementwiseQuadratic(2),
    "cqf": QuadraticForms(
        A=[rotated_form(pi / 16, [1.0, 0.1]), rotated_form(-pi / 16, [sqrt(3.0), 0.1])],
        u=[[1.0, 0.0], [-1.0, 0.0]],
    ),
    "cqf2": QuadraticForms(
        A=[[[1.0, 0.2], [0.2, 0.05]], [[3.0, -0.6], [-0.6, 0.2]]],
        u=[[1.0, 0.0], [-1.0, 0.0]],
    ),
    "hqf": HomogeneousQuadraticForm(
        A=[[2.0, -1.0], [-1.0, 2.0]],
        scales=[1.0, 10.0],
        u=[[1.0, 0.0], [-10.0, 0.0]],
    ),
    "mn2": Multinorm(a=[1.0, 10.0]),
    "mn20": Multinorm(a=np.arange(1.0, 21.0)),
}

_VB_INITS = [
    [-8.5, 7.5],
    [0.0, 0.0],
    [9.0, 9.0],
    [-7.5, -0.5],
    [9.0, -1.0],
]
INITIAL_POINTS = {
    "vb": _VB_INITS,
    "vb01": _VB_INITS,
    "ewq": [
        [3.0, -2.0],
        [0.0, -3.0],
        [-4.0, 4.0],
        [-3.0, 4.0],
        [-3.5, -0.75],
    ],
    "cqf": [
        [0.5, 0.5],
        [-1.0, 7.0],
        [0.0, 0.0],
        [1.0, 6.0],
    ],
    "cqf2": [
        [0.5, 0.5],
        [-0.3, 7.0],
        [0.0, 0.0],
    ],
    "hqf": [
        [-6.0, 4.0],
        [-3.0, -1.5],
        [1.5, 2.0],
        [2.5, 5.5],
    ],
    "mn2": [
        [0.0, 0.0],
        [-5.0, 5.0],
        [10.0, 5.0],
        [10.0, 0.0],
        [20.0, 0.0],
    ],
    "mn20": [
        [0.0] * 20,
    ],
}

# Fixed limits and ticks of the figures of the 2-D problems, as PLOT_PARAMS_LIMS / PLOT_VALUES_LIMS of TorchJD's
# _constants.py: search plots (parameter space, theta_1 x theta_2; equal axes) and value plots (loss space, L_1 x L_2).
# They cover the initial points and the Pareto set (search) or their losses and the Pareto front (value).
_VB_SEARCH = {"xlim": (-10.5, 10.5), "ylim": (-10.5, 10.5), "xticks": [-10, -5, 0, 5, 10], "yticks": [-10, -5, 0, 5, 10]}
_QF_SEARCH = {"xlim": (-4.5, 4.5), "ylim": (-0.5, 8.5), "xticks": [-4, -2, 0, 2, 4], "yticks": [0, 2, 4, 6, 8]}
SEARCH_PLOT_LIMS = {
    "vb": _VB_SEARCH,
    "vb01": _VB_SEARCH,
    "ewq": {"xlim": (-4.5, 4.5), "ylim": (-4.5, 4.5), "xticks": [-4, -2, 0, 2, 4], "yticks": [-4, -2, 0, 2, 4]},
    "cqf": _QF_SEARCH,
    "cqf2": _QF_SEARCH,
    "hqf": {"xlim": (-11.5, 4.0), "ylim": (-5.5, 10.0), "xticks": [-10, -5, 0], "yticks": [-5, 0, 5, 10]},
    "mn2": {"xlim": (-7.5, 22.0), "ylim": (-10.0, 19.5), "xticks": [-5, 0, 5, 10, 15, 20],
            "yticks": [-10, -5, 0, 5, 10, 15]},
}
_QF_VALUE = {"xlim": (-0.25, 4.25), "ylim": (-0.75, 12.75), "xticks": [0, 1, 2, 3, 4], "yticks": [0, 4, 8, 12]}
VALUE_PLOT_LIMS = {
    "vb": {"xlim": (-21.0, 8.5), "ylim": (-21.0, 8.5), "xticks": [-20, -15, -10, -5, 0, 5],
           "yticks": [-20, -15, -10, -5, 0, 5]},
    "vb01": {"xlim": (-2.1, 1.1), "ylim": (-21.0, 8.5), "xticks": [-2, -1, 0, 1], "yticks": [-20, -15, -10, -5, 0, 5]},
    "ewq": {"xlim": (-1.0, 17.0), "ylim": (-1.0, 17.0), "xticks": [0, 4, 8, 12, 16], "yticks": [0, 4, 8, 12, 16]},
    "cqf": _QF_VALUE,
    "cqf2": _QF_VALUE,
    "hqf": {"xlim": (-12.0, 254.0), "ylim": (-120.0, 2540.0), "xticks": [0, 100, 200], "yticks": [0, 1000, 2000]},
    "mn2": {"xlim": (-18.0, 380.0), "ylim": (-250.0, 5250.0), "xticks": [0, 100, 200, 300], "yticks": [0, 2500, 5000]},
}

# families of methods in the comparison figures: a row of panels per family; a method's position in its family sets
# its color and marker (COLORS, MARKERS), so the methods at the same position in different families match
FAMILIES = {
    "OpenES": ["openes", "mo_upgrad_openes", "mo_cagrad_openes", "mo_mgda_openes",
               "mo_mgda_l2_norm_openes", "mo_mgda_loss_norm_openes", "mo_mgda_lossplus_norm_openes",
               "mo_aligned_mtl_openes", "mo_svd_pga_openes", "mo_os_nmf_pga_openes"],
    "GuidedES": ["guidedes", "mo_upgrad_guidedes", "mo_cagrad_guidedes", "mo_mgda_guidedes",
                 "mo_mgda_l2_norm_guidedes", "mo_mgda_loss_norm_guidedes", "mo_mgda_lossplus_norm_guidedes",
                 "mo_aligned_mtl_guidedes", "mo_svd_pga_guidedes", "mo_os_nmf_pga_guidedes"],
    # gradient counterparts of the ES methods above (LibMTL weightings; same positions, so same colors)
    "Grad": ["ew", "upgrad", "cagrad", "mgda", "mgda_l2_norm", "mgda_loss_norm", "mgda_lossplus_norm",
             "aligned_mtl", "svd_pga", "os_nmf_pga"],
    # the other LibMTL weightings (not GradDrop)
    "LibMTL": ["gradnorm", "uw", "dwa", "gls", "pcgrad", "gradvac", "imtl", "nash_mtl", "rlw", "moco", "db_mtl",
               "stch", "excessmtl", "fairgrad", "famo", "modo", "sdmgrad", "dualproj"],
}
SUFFIXES = ("_adam",)  # protocol variants of a method, in the same results folder
# convergence curves, by position of the method in its family (FAMILIES): baseline, MO-UPGrad, MO-CAGrad, MO-MGDA,
# MO-MGDA (L2 Norm), ... (Grad: EW, UPGrad, CAGrad, MGDA, MGDA (L2 Norm), ...)
COLORS = ["C0", "C4", "C2", "C1", "C3", "C5", "C7", "C6", "C8", "C9", "#393b79", "#b5179e", "#637939", "#bd9e39",
          "#ad494a", "#17a589", "#e377c2", "#8c6d31", "#7b4173", "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363"]
MARKERS = ["o", "s", "^", "P", "D", "X", "v", "<", ">", "p", "h", "*", "d", "8", "H", "1", "2", "3", "4", "+", "x",
           "|", "_", "."]
# trajectory panels: the colors of Nash-MTL's plot_2d_pareto for every method
FRONT_COLOR = "#72727A"
TRAJ_CMAP, TRAJ_RANGE = "magma_r", (0.1, 0.6)  # matplotlib colormap: orange (start) to purple (end)
# spacing of the loss-plane grids: room for the axis labels at the ends of the axes (L_1 right of the x axis, L_2
# above the y axis, as in Nash-MTL's plot_2d_pareto) between neighboring panels
LOSS_PLANE_SPACING = {"wspace": 0.45, "hspace": 0.5}
