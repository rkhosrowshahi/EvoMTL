"""Multi-objective evolutionary algorithms used by :class:`~LibMTL.evomtl.evo_trainer.EvoMTLTrainer`."""

from .comocma import run_comocma
from .dcpsl_es import run_dcpsl_es
from .guidedes import run_guidedes
from .hv import (
    marginal_hypervolume_weights,
    pareto_center_from_weights,
    pareto_center_solution,
    pareto_center_weights_from_marginal_hv,
    pareto_front_indices,
)
from .mo_mean_openes import run_mo_mean_openes
from .mo_mgda_guidedes import run_mo_mgda_guidedes
from .mo_mgda_openes import run_mo_mgda_openes
from .mo_mgda_openes_libmtl import run_mo_mgda_openes_libmtl
from .mo_upgrad_guidedes import run_mo_upgrad_guidedes
from .mo_upgrad_openes import run_mo_upgrad_openes
from .mopso import run_mopso
from .nsga2 import run_nsga2
from .openes import run_openes
from .pgpe import run_pgpe
from .psl_es import run_psl_es
from .psl_taskguide import run_psl_taskguide
from .sc_de import run_sc_de
from .snes import run_snes
from .tpsl_es import run_tpsl_es

__all__ = [
    "marginal_hypervolume_weights",
    "pareto_center_from_weights",
    "pareto_center_solution",
    "pareto_center_weights_from_marginal_hv",
    "pareto_front_indices",
    "run_comocma",
    "run_dcpsl_es",
    "run_guidedes",
    "run_mo_mean_openes",
    "run_mo_mgda_guidedes",
    "run_mo_mgda_openes",
    "run_mo_mgda_openes_libmtl",
    "run_mo_upgrad_guidedes",
    "run_mo_upgrad_openes",
    "run_mopso",
    "run_nsga2",
    "run_openes",
    "run_pgpe",
    "run_psl_es",
    "run_psl_taskguide",
    "run_sc_de",
    "run_snes",
    "run_tpsl_es",
]
