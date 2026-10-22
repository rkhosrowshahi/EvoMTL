"""How many generations until ES displacement points along the negative true gradient? (local experiment)

Runs an ES config exactly as ``main.py`` does (same model, data, seeds, Guided ES / MO-OpenES loop) with
W&B off, and records at every generation t:

* ``p_t``  the ES pseudo-gradient that drives the update (the aggregated direction)
* ``g_t``  the backprop gradient at the mean on the same mini-batch, aggregated by the same rule
           (sum of task gradients for ``ws``, UPGrad of the true Jacobian for ``upgrad``)
* ``m_t``  the search mean before the update (and the final mean)

Then, for windows of T consecutive generations starting at s (unit vectors ^, sums over t = s .. s+T-1,
Dm = m_{s+T} - m_s is the displacement Adam actually applied):

* ``est``        cos(sum p^, sum g^)       raw pseudo-gradients averaged over T vs true gradients averaged over T
* ``step_avg``   cos(-Dm, sum g^)          the applied displacement vs the window's average descent direction
* ``step_start`` cos(-Dm, g^_s)            the displacement vs the gradient where the window started
* ``drift``      cos(sum g^, g^_s)         the same for noiseless gradient steps (ceiling of step_start)
* ``coherence``  ||sum g^|| / T            how consistently the true gradient points one way over T (1 = fixed)
* ``theory``     c C / sqrt(c^2 C^2 + T (1 - c^2)), C = ||sum g^||, c = mean one-step cos(p, g):
                 ``est`` if each p_t were c * g^_t plus fresh isotropic noise

Writes to ``--out``: ``window_cos.csv`` (mean and quartiles per T, all windows and by phase of training),
``window_cos.png/.pdf``, ``metrics.csv`` and checkpoints of the run itself, and with ``--save_vectors``
``vectors.npz`` (about 0.6 GB for 2344 generations) so the analysis can be redone with ``--analyze``.

Usage (from examples/multimnist; extra flags go to main.py and override the YAML)::

    uv run python scripts/es_window_cosine.py --config configs/10epochs/mnist/evomtl/guided_es-ws/pop_128/adam-lr_0.001/seed1_s0.05_a0.9_ema0.9.yaml
    uv run python scripts/es_window_cosine.py --config <same> --evo_pop_size 2048 --out logs/mnist/window_cos/guided_es-ws/pop_2048
    uv run python scripts/es_window_cosine.py --analyze logs/mnist/window_cos/guided_es-ws/pop_128/vectors.npz
"""

import argparse
import csv
import os
import sys

import numpy as np

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASES = (("all", 0.0, 1.0), ("early", 0.0, 0.25), ("middle", 0.25, 0.5), ("late", 0.5, 1.0))
METRICS = ("est", "step_avg", "step_start", "drift", "coherence", "theory")


# --------------------------------------------------------------------------- recording

def run_and_record(cfg_path, out, forward):
    """Run main.py's training on ``cfg_path`` with hooks on the ES loop; return the recorded arrays."""
    import torch

    sys.path.insert(0, EXAMPLE_DIR)
    os.chdir(EXAMPLE_DIR)  # dataset_path '.' and relative save paths, as with main.py
    import main as mm
    from LibMTL.config import LibMTL_args
    from LibMTL.evomtl.moea import mo_openes as es
    from LibMTL.utils import set_device, set_random_seed

    sys.argv = ["main.py", "--config", cfg_path, "--save_path", out + "/", *forward]
    params = mm.parse_args(LibMTL_args)
    params.wandb_enable, params.wandb_project = False, None
    params.evo_es_grad_cos_freq = 1  # backprop gradient at every generation (captured below)
    if params.evo_moea not in ("mo_openes", "mo-openes", "openes"):
        raise SystemExit(f"needs an OpenES / Guided ES config (evo_moea mo_openes), got {params.evo_moea!r}")
    aggregator = params.evo_es_aggregator
    n, d = int(params.evo_iterations), None
    rec = {}

    orig_true_jacobian, orig_make_opt = es.true_jacobian, es.make_es_optimizer

    def true_jacobian(trainer, theta, batches):
        J = orig_true_jacobian(trainer, theta, batches)
        rec["J_true"] = J
        return J

    def make_es_optimizer(params_list, *a, **k):
        opt = orig_make_opt(params_list, *a, **k)
        mean = params_list[0]
        D = mean.numel()
        rec.update(t=0, p=np.zeros((n, D), np.float32), g=np.zeros((n, D), np.float32),
                   m=np.zeros((n + 1, D), np.float32))
        orig_step = opt.step

        def step(*sa, **sk):
            t = rec["t"]
            J = rec.pop("J_true")
            g = J.sum(0) if aggregator == "ws" else es.aggregate_jacobian(
                J, aggregator, params.upgrad_norm_eps, params.upgrad_reg_eps)[0]
            rec["m"][t] = mean.detach().numpy()
            rec["p"][t] = mean.grad.detach().numpy()
            rec["g"][t] = g.numpy()
            r = orig_step(*sa, **sk)
            rec["m"][t + 1] = mean.detach().numpy()
            rec["t"] = t + 1
            return r

        opt.step = step
        return opt

    es.true_jacobian, es.make_es_optimizer = true_jacobian, make_es_optimizer
    mm._log_to_save_path(params)  # stdout.log in --out, like main.py
    set_device(params.gpu_id)
    set_random_seed(params.seed)
    mm.main(params)
    t = rec["t"]
    meta = dict(aggregator=aggregator, pop_size=int(params.evo_pop_size), dims=rec["p"].shape[1],
                config=cfg_path, guided_k=int(params.evo_es_guided_k))
    return rec["p"][:t], rec["g"][:t], rec["m"][:t + 1], meta


# --------------------------------------------------------------------------- analysis

def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def rowwise_cos(a, b):
    return np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)


def window_sizes(n):
    return np.unique(np.round(np.logspace(0, np.log10(max(n // 2, 2)), 28)).astype(int))


def analyze(p, g, m):
    """Per window length T: arrays of per-window metrics and the window start fractions."""
    n = len(p)
    p, g = unit_rows(p.astype(np.float32)), unit_rows(g.astype(np.float32))
    c = float(np.mean(np.einsum("ij,ij->i", p, g)))  # one-step cos(p, g)
    P = np.zeros((n + 1, p.shape[1]), np.float32)
    np.cumsum(p, axis=0, out=P[1:])
    del p
    G = np.zeros_like(P)
    np.cumsum(g, axis=0, out=G[1:])
    out = {}
    for T in window_sizes(n):
        s = np.arange(0, n - T + 1, max(1, T // 4))
        Sp, Sg, Dm = P[s + T] - P[s], G[s + T] - G[s], m[s + T] - m[s]
        C = np.linalg.norm(Sg, axis=1)
        out[int(T)] = {
            "start": s / n,
            "est": rowwise_cos(Sp, Sg),
            "step_avg": rowwise_cos(-Dm, Sg),
            "step_start": rowwise_cos(-Dm, g[s]),
            "drift": rowwise_cos(Sg, g[s]),
            "coherence": C / T,
            "theory": c * C / np.sqrt(c * c * C * C + T * (1 - c * c)),
        }
    return out, c


def summarize(res, out_dir):
    rows = []
    for T, r in res.items():
        for phase, lo, hi in PHASES:
            sel = (r["start"] >= lo) & (r["start"] < hi)
            if not sel.any():
                continue
            row = {"T": T, "phase": phase, "n_windows": int(sel.sum())}
            for k in METRICS:
                q = np.percentile(r[k][sel], [25, 50, 75])
                row[k] = float(r[k][sel].mean())
                row[f"{k}_q25"], row[f"{k}_q75"] = float(q[0]), float(q[2])
            rows.append(row)
    with open(os.path.join(out_dir, "window_cos.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def report(rows, c):
    allr = [r for r in rows if r["phase"] == "all"]
    print(f"\none-step cos(pseudo-gradient, true gradient) c = {c:.4f}  ->  1/c^2 = {1 / c ** 2:.0f} generations")
    print(f"{'T':>6} {'est':>7} {'theory':>7} {'step_avg':>9} {'step_start':>11} {'drift':>7} {'coherence':>10}")
    for r in allr:
        print(f"{r['T']:>6} {r['est']:7.3f} {r['theory']:7.3f} {r['step_avg']:9.3f} {r['step_start']:11.3f} "
              f"{r['drift']:7.3f} {r['coherence']:10.3f}")
    for phase in ("all", "early", "middle", "late"):
        pr = [r for r in rows if r["phase"] == phase]
        if not pr:
            continue
        best = max(pr, key=lambda r: r["step_start"])
        half = next((r["T"] for r in pr if r["est"] >= 0.5), None)
        print(f"{phase:>6}: step_start peaks at T* = {best['T']} (cos {best['step_start']:.3f}); "
              f"est reaches 0.5 at T = {half if half else '> ' + str(pr[-1]['T'])}")


def plot(rows, c, meta, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"est": "#2a78d6", "step_avg": "#eb6834", "step_start": "#1baf7a", "coherence": "#eda100"}
    labels = {"est": "cos(Σ pseudo-grad, Σ true grad)", "step_avg": "cos(applied displacement, −Σ true grad)",
              "step_start": "cos(applied displacement, −true grad at window start)",
              "coherence": "true-gradient coherence ‖Σ ĝ‖ / T"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), layout="constrained")

    ax = axes[0]
    allr = [r for r in rows if r["phase"] == "all"]
    T = np.array([r["T"] for r in allr])
    for k in ("est", "step_avg", "step_start", "coherence"):
        ax.fill_between(T, [r[f"{k}_q25"] for r in allr], [r[f"{k}_q75"] for r in allr],
                        color=colors[k], alpha=0.15, linewidth=0)
        ax.plot(T, [r[k] for r in allr], color=colors[k], linewidth=2, label=labels[k])
    ax.plot(T, [r["theory"] for r in allr], color="#6b6a64", linestyle="--", linewidth=1.4,
            label="theory for Σ pseudo-grad (independent noise, measured coherence)")
    ax.plot(T, [r["drift"] for r in allr], color="#6b6a64", linestyle=":", linewidth=1.4,
            label="ceiling: cos(Σ true grad, true grad at window start)")
    ax.axvline(1 / c ** 2, color="#b5b4ad", linewidth=1)
    ax.annotate(f"1/c² = {1 / c ** 2:.0f}", (1 / c ** 2, 0.02), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color="#55544e")
    ax.set_title("All windows (line = mean, band = 25–75%)", loc="left", fontsize=11)

    ax = axes[1]
    for (phase, _, _), ls in zip(PHASES[1:], ("-", "--", ":")):
        pr = [r for r in rows if r["phase"] == phase]
        if pr:
            ax.plot([r["T"] for r in pr], [r["step_start"] for r in pr], color=colors["step_start"],
                    linestyle=ls, linewidth=2, label=f"{phase} windows: displacement vs start gradient")
            ax.plot([r["T"] for r in pr], [r["drift"] for r in pr], color="#6b6a64", linestyle=ls,
                    linewidth=1.2, label=f"{phase} windows: ceiling (gradient drift)")
    ax.set_title("By phase of training (start of window)", loc="left", fontsize=11)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Window length T (generations)")
        ax.grid(color="#e4e3dd", linewidth=0.8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    axes[0].set_ylabel("cosine / coherence")
    name = os.path.basename(str(meta.get("config", ""))).replace(".yaml", "")
    fig.suptitle(f"ES over windows of T generations: aggregator={meta.get('aggregator')}, "
                 f"N={meta.get('pop_size')}, d={meta.get('dims'):,}, one-step c={c:.3f}  ({name})",
                 x=0.01, ha="left", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"window_cos.{ext}"), dpi=200)
    print(f"wrote {os.path.join(out_dir, 'window_cos.png')} / .pdf and window_cos.csv")


# --------------------------------------------------------------------------- entry point

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="ES config YAML (relative to examples/multimnist or absolute)")
    ap.add_argument("--out", default=None,
                    help="output folder (default logs/mnist/window_cos/<method folder>/pop_<N>/seed_<s>)")
    ap.add_argument("--save_vectors", action="store_true", help="also save p, g, m to vectors.npz (~0.6 GB)")
    ap.add_argument("--analyze", default=None, help="vectors.npz from an earlier run: analysis only, no training")
    args, forward = ap.parse_known_args()

    if args.analyze:
        z = np.load(args.analyze, allow_pickle=True)
        p, g, m, meta = z["p"], z["g"], z["m"], z["meta"].item()
        out = args.out or os.path.dirname(os.path.abspath(args.analyze))
    else:
        if not args.config:
            ap.error("--config or --analyze is required")
        cfg = os.path.abspath(args.config) if os.path.exists(args.config) else os.path.join(EXAMPLE_DIR, args.config)
        if args.out:
            out = os.path.abspath(args.out)
        else:
            import yaml
            y = yaml.safe_load(open(cfg))
            pop = next((forward[i + 1] for i, a in enumerate(forward[:-1]) if a == "--evo_pop_size"), y["evo_pop_size"])
            seed = next((forward[i + 1] for i, a in enumerate(forward[:-1]) if a == "--seed"), y["seed"])
            method = os.path.normpath(cfg).split(os.sep)[-4]
            out = os.path.join(EXAMPLE_DIR, "logs/mnist/window_cos", method, f"pop_{pop}", f"seed_{seed}")
        os.makedirs(out, exist_ok=True)
        p, g, m, meta = run_and_record(cfg, out, forward)
        if args.save_vectors:
            np.savez(os.path.join(out, "vectors.npz"), p=p, g=g, m=m, meta=np.array(meta, dtype=object))
        sys.stdout = getattr(sys.stdout, "stream", sys.stdout)  # stop mirroring into stdout.log

    os.makedirs(out, exist_ok=True)
    print(f"\nanalyzing {len(p)} generations, d = {p.shape[1]} ...")
    res, c = analyze(p, g, m)
    rows = summarize(res, out)
    report(rows, c)
    plot(rows, c, meta, out)


if __name__ == "__main__":
    main()
