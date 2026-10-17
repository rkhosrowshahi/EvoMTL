"""Generate the 10-epoch protocol configs (PaMaL's MultiMNIST budget) from the existing configs.

10 epochs of the 60,000 training images with batch 256 is 2,344 ES generations (one batch per generation).
Every config is a copy of an existing one with only the budget, the seeds and the paths changed:

* ES methods (from ``no_scheduler/mnist/evomtl/``): ``evo_iterations: 2344``. The five single-network ES
  methods and the four Pareto-set-learning variants.
* gradient-descent methods (every other folder of ``no_scheduler/mnist/``): ``epochs: 10`` and Adam with
  ``lr: 1.0e-3``, PaMaL's optimiser settings (other settings, e.g. weight decay, unchanged).

Output: ``10epochs/mnist/<same relative path>``; logs go to ``logs/mnist/10epochs/<same relative path>``, so nothing
from the 500-generation or 100-epoch runs is overwritten. W&B logging is on (project ``multimnist``). ``runner.py --dir`` runs one method folder. W&B names start with the method, then ``10epochs``.

Usage (from this folder)::

    python generate_10epochs_configs.py                      # seed 1 (testing)
    python generate_10epochs_configs.py --seeds 1 12 123 1234 12345 123456
"""

import argparse
import re
from pathlib import Path

GENERATIONS = 2344  # 10 * 60,000 / 256
EPOCHS, GD_LR = 10, "1.0e-3"
HEADER = "# 10-epoch protocol (PaMaL's budget): {what}.\n"

# ES base configs (seed 1) under no_scheduler/mnist/evomtl/, written without the _gen500 suffix
ES_BASES = [
    "openes-ws-full/pop_128/adam-lr_0.001/seed1.yaml",
    "guided_es-ws-full/pop_128/adam-lr_0.001/seed1_s0.05_a0.9_ema0.9.yaml",
    "mo_openes-upgrad-full/pop_128/adam-lr_0.001/seed1_gen500.yaml",
    "mo_guided_es-upgrad-full/pop_128/adam-lr_0.001/seed1_s0.05_a0.9_ema0.9.yaml",
    "mo_openes-mean-full/pop_128/adam-lr_0.001/seed1_gen500.yaml",
    "psl_guided_es-tch-full/pop_128/adam-lr_0.001/seed1_s0.05_a0.9_ema0.9.yaml",
    "psl_guided_es-tch-full/pop_128/adam-lr_0.001/seed1_s0.05_a0.9_ema0.9_taskguide.yaml",
    "psl_guided_es-tch-full/pop_128/adam-lr_0.001/seed1_s0.05_a0.9_ema0.9_center4_idealpop.yaml",
    "psl_guided_es-tch-full/pop_128/adam-lr_0.001/seed1_s0.05_a0.9_ema0.9_center4_idealpop_taskguide.yaml",
]


def sub_once(pattern, repl, text, name):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        raise ValueError(f"{name}: pattern {pattern!r} not found")
    return new


def protocol_paths(text, seed, name, method):
    """Logs under logs/mnist/10epochs/, seed in the path, W&B names <method>-10epochs-..."""
    def after_method(value):
        q = value[0] if value[:1] in "\"'" else ""
        value = value.strip("\"'")
        if not value.startswith(method):
            raise ValueError(f"{name}: {value!r} does not start with {method!r}")
        return q + method + "-10epochs" + value[len(method):] + q

    text = sub_once(r"^save_path: logs/mnist/(\S+)", lambda m: "save_path: logs/mnist/10epochs/"
                    + re.sub(r"seed_\d+", f"seed_{seed}", m.group(1)), text, name)
    text = re.sub(r"^(wandb_name: )(\S+)", lambda m: m.group(1) + after_method(re.sub(r"seed\d+", f"seed{seed}", m.group(2))),
                  text, flags=re.M)
    text = re.sub(r"^(wandb_group: )(\S+)", lambda m: m.group(1) + after_method(m.group(2)), text, flags=re.M)
    # Weights & Biases on, project multimnist
    return sub_once(r"^wandb_project: \S+.*$", "wandb_enable: true\nwandb_project: multimnist", text, name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    src, out = root / "no_scheduler" / "mnist", root / "10epochs" / "mnist"
    n_es = n_gd = 0
    for seed in args.seeds:
        for rel in ES_BASES:  # ES: seed 1 bases, other seeds by substitution
            text = (src / "evomtl" / rel).read_text()
            text = sub_once(r"^evo_iterations: \d+.*$", f"evo_iterations: {GENERATIONS}        "
                            "# 10 epochs: 2,344 generations x 256 images", text, rel)
            text = sub_once(r"^seed: \d+", f"seed: {seed}", text, rel)
            text = sub_once(r"^evo_seed: \d+", f"evo_seed: {seed}", text, rel)
            text = HEADER.format(what=f"{GENERATIONS} generations x 256 images = 10 epochs") + protocol_paths(text, seed, rel, Path(rel).parts[0])
            name = re.sub(r"^seed1", f"seed{seed}", Path(rel).name).replace("_gen500", "")
            dest = out / "evomtl" / Path(rel).parent / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            n_es += 1
        for method in sorted(d for d in src.iterdir() if d.is_dir() and d.name != "evomtl"):
            base = method / f"seed{seed}.yaml"
            if not base.exists():  # no config for this seed: the seed-1 config with the seed substituted
                base = method / "seed1.yaml"
                if not base.exists():
                    continue
            text = sub_once(r"^seed: \d+", f"seed: {seed}", base.read_text(), str(base))
            text = sub_once(r"^epochs: \d+", f"epochs: {EPOCHS}", text, str(base))
            text = sub_once(r"^lr: \S+", f"lr: {GD_LR}", text, str(base))
            text = HEADER.format(what=f"{EPOCHS} epochs, Adam with learning rate {GD_LR}") + protocol_paths(text, seed, str(base), method.name)
            dest = out / method.name / f"seed{seed}.yaml"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            n_gd += 1
    print(f"wrote {n_es} ES and {n_gd} gradient-descent configs under {out}")


if __name__ == "__main__":
    main()
