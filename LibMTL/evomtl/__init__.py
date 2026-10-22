def __getattr__(name):
    if name == "EvoMTLTrainer":
        from .evo_trainer import EvoMTLTrainer

        return EvoMTLTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EvoMTLTrainer",
]
