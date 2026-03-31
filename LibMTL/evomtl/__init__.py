from .parameter_sharing import (
    RandomProjectionParameterSharing,
    LayerwiseRandomProjectionParameterSharing,
    LayerwiseScaledRandomProjectionParameterSharing,
    FlattenLoRAParameterSharing,
    DictLoRAParameterSharing,
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
    "RandomProjectionParameterSharing",
    "LayerwiseRandomProjectionParameterSharing",
    "LayerwiseScaledRandomProjectionParameterSharing",
    "FlattenLoRAParameterSharing",
    "DictLoRAParameterSharing",
    "LinearOnlyLoRA",
    "ModulationLoRA",
    "SpectralAllSVD",
    "SpectralLoRA",
    "EvoMTLTrainer",
]
