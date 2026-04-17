"""Central pymoo imports for MOEA runners (single ImportError message)."""

from __future__ import annotations

try:
    from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.core.survival import Survival
    from pymoo.core.termination import NoTermination
    from pymoo.indicators.hv import Hypervolume
    from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
    from pymoo.problems.static import StaticProblem
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "EvoMTLTrainer requires pymoo. Install with: pip install pymoo"
    ) from e
