"""Guided graph search retaining the successful proposal's original reveal order."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .diffusion import Denoiser, sample
from .experiment import (
    BATCH,
    INITIAL,
    INTERVENTION_SEEDS,
    WINDOWS,
    fit_record,
    initial_pool,
    weights,
)
from .fitting import Scorer
from .grammar import encode
from .report import plot_trajectory
from .wave_data import coefficient_sign, make_blocks, synthetic

LEAVES = ("u", "u_t")
SCORE_ATOL = 1e-12


def prepare_benchmark():
    """Refit the known structure on the same data used to score candidates."""
    wave = synthetic(mixed=True)
    train, validation, test = (make_blocks(wave, split) for split in range(3))
    mild, strong = (
        [make_blocks(wave, 0, window, seed) for seed in INTERVENTION_SEEDS]
        for window in WINDOWS
    )
    scorer = Scorer(train, validation, test, mild, strong, coefficient_sign)
    known = scorer.evaluate(wave.truth)
    if not known.eligible:
        raise ValueError("The refitted known equation must qualify for the benchmark.")
    return wave, scorer, known


def search_until_quality(
    wave, scorer, target, seed=55, max_unique=None, progress=None, require_exact=False
):
    """Stop at an eligible quality hit, optionally requiring the ground-truth structure."""
    initial = initial_pool(LEAVES, seed, exclude=True)
    if max_unique is not None and max_unique < len(initial):
        raise ValueError("max_unique must accommodate the shared initialization.")
    if wave.truth in initial:
        raise ValueError("The known structure must be absent from the starting pool.")

    proposal_rng = np.random.default_rng(seed + 4000)
    training_rng = np.random.default_rng(seed + 5000)
    denoiser = Denoiser(seed)
    seen, tokens, curve, history = {}, [], [], []
    incumbent = None
    successful = None
    first_quality = None
    first_exact = None
    duplicates = 0
    start = perf_counter()

    def evaluate(graph, reveal_order, initial=False):
        nonlocal incumbent, successful, first_quality, first_exact
        result = seen[graph] = scorer.evaluate(graph)
        tokens.append(encode(graph, LEAVES))
        if graph == wave.truth and first_exact is None:
            first_exact = len(seen)
        if result.eligible and (
            incumbent is None or result.score() < incumbent.score()
        ):
            incumbent = result
        curve.append(None if incumbent is None else float(incumbent.score()))
        record = dict(
            evaluation=len(seen),
            initial=initial,
            reveal_order=list(reveal_order),
            **fit_record(result),
        )
        history.append(record)
        quality_hit = result.eligible and result.score() <= target + SCORE_ATOL
        if quality_hit and first_quality is None:
            first_quality = len(seen)
        reached = quality_hit and (not require_exact or graph == wave.truth)
        if reached:
            successful = record
        return reached

    reached = False
    for graph in initial:
        reached = evaluate(graph, encode(graph, LEAVES), initial=True)
        if reached:
            break

    round_number = 0
    while not reached and (max_unique is None or len(seen) < max_unique):
        round_number += 1
        denoiser.update(
            tokens,
            LEAVES,
            weights(list(seen.values()), 0.035),
            training_rng,
            steps=100 if round_number == 1 else 55,
        )
        goal = len(seen) + BATCH
        if max_unique is not None:
            goal = min(goal, max_unique)
        for _ in range(15_000):
            graph, reveal_order = sample(LEAVES, proposal_rng, denoiser)
            if graph in seen:
                duplicates += 1
                continue
            reached = evaluate(graph, reveal_order)
            if reached or len(seen) == goal:
                break
        else:
            raise RuntimeError("The sampler could not fill a batch of distinct graphs.")
        if progress is not None:
            progress(seed, len(seen), curve[-1], perf_counter() - start, reached)

    if incumbent is None:
        raise RuntimeError("No eligible graph was found within the evaluation budget.")
    selected = seen[successful["graph"]] if reached else incumbent
    return {
        "seed": seed,
        "curve": curve,
        "history": history,
        "first_quality": first_quality,
        "first_exact": first_exact,
        "reached": bool(reached),
        "successful": successful,
        "duplicates": duplicates,
        "final": fit_record(selected),
        "heldout_nmse": float(scorer.heldout(selected)[0]),
        "seconds": perf_counter() - start,
    }


def plot_search_progress(run, target):
    """One run's best eligible score and the known-equation reference."""
    curve = np.asarray(run["curve"][INITIAL - 1 :], dtype=float)
    x = np.arange(len(curve))
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ax.step(
        x, curve, where="post", color="#11635d", linewidth=2, label="Guided diffusion"
    )
    ax.axhline(
        target, color="#596579", linestyle="--", linewidth=1.2, label="Known equation"
    )
    ax.scatter(x[-1], curve[-1], color="#11635d", s=30, zorder=3)
    ax.set(
        title="Graph search to ground-truth structure",
        yscale="log",
        xlabel="New distinct graph evaluations after the shared pool",
        ylabel="Best eligible validation score",
        xlim=(0, max(1, x[-1]) * 1.03),
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="major", alpha=0.18)
    ax.legend(frameon=False)
    return fig


def plot_generation_history(run, count=5):
    """Show partial states from the successful candidate's actual sampled trajectory."""
    if count < 1:
        raise ValueError("count must be positive.")
    successful = run["successful"]
    if successful is None:
        raise ValueError("No successful candidate was sampled. Increase max_unique.")
    tokens = successful["reveal_order"]
    counts = (
        [len(tokens)]
        if count == 1
        else np.linspace(
            0,
            len(tokens),
            min(count, len(tokens) + 1),
            dtype=int,
        ).tolist()
    )
    fig = plot_trajectory(tokens, LEAVES, counts=counts)
    fig.suptitle(
        f"One successful candidate · evaluation {successful['evaluation']}", fontsize=12
    )
    return fig
