from . import architecture
from . import model
from . import weighting
from .trainer import Trainer
from . import config
from . import loss
from . import metrics
#from .record import PerformanceMeter
from . import utils


def __getattr__(name):
    if name == "EvoMTLTrainer":
        from .evomtl.evo_trainer import EvoMTLTrainer

        return EvoMTLTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "architecture",
    "model",
    "weighting",
    "Trainer",
    "EvoMTLTrainer",
    "config",
    "loss",
    "metrics",
    "utils",
]
