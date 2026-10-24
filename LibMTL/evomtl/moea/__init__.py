"""Multi-objective evolutionary algorithms used by :class:`~LibMTL.evomtl.evo_trainer.EvoMTLTrainer`."""

from .comocma import run_comocma
from .dcpsl_es import run_dcpsl_es
from .emapsl_es import run_emapsl_es
from .hv import (
    marginal_hypervolume_weights,
    pareto_center_from_weights,
    pareto_center_solution,
    pareto_center_weights_from_marginal_hv,
    pareto_front_indices,
)
from .mo_openes import run_mo_openes
from .mopso import run_mopso
from .nsga2 import run_nsga2
from .pcd_guidedes import run_pcd_guidedes
from .pcd_openes import run_pcd_openes
from .pcd_openes_libmtl import run_pcd_openes_libmtl
from .psl_es import run_psl_es
from .psl_taskguide import run_psl_taskguide
from .tpsl_es import run_tpsl_es

__all__ = [
    "marginal_hypervolume_weights",
    "pareto_center_from_weights",
    "pareto_center_solution",
    "pareto_center_weights_from_marginal_hv",
    "pareto_front_indices",
    "run_comocma",
    "run_dcpsl_es",
    "run_emapsl_es",
    "run_mo_openes",
    "run_mopso",
    "run_nsga2",
    "run_pcd_guidedes",
    "run_pcd_openes",
    "run_pcd_openes_libmtl",
    "run_psl_es",
    "run_psl_taskguide",
    "run_tpsl_es",
]
