from .parameter_sharing import (
    RandomProjection,
    LayerwiseRandomBlocking,
    LayerwiseRandomProjection,
    LayerwiseScaledRandomProjection,
    FlattenLoRA,
    DictLoRA,
    LinearOnlyLoRA,
    ModulationLoRA,
    SpectralAllSVD,
    SpectralLoRA,
)


def __getattr__(name):
    if name == "EvoMTLTrainer":
        from .evo_trainer import EvoMTLTrainer

        return EvoMTLTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RandomProjection",
    "LayerwiseRandomBlocking",
    "LayerwiseRandomProjection",
    "LayerwiseScaledRandomProjection",
    "FlattenLoRA",
    "DictLoRA",
    "LinearOnlyLoRA",
    "ModulationLoRA",
    "SpectralAllSVD",
    "SpectralLoRA",
    "EvoMTLTrainer",
]
