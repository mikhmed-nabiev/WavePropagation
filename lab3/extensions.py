"""Compare generated wave equations against the fixed Lab 1 model."""
import argparse
from itertools import combinations_with_replacement
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .diffusion import Denoiser, sample_rule
from .experiment import INTERVENTION_SEEDS, METHODS, WINDOWS, finalist, fit_record, weights
from .fitting import Scorer, normalized_error
from .grammar import AXES, LAPLACIAN, MIXED, Derivation, canonical_graph, encode
from .wave_data import WaveData, coefficient_sign, make_blocks


LEAVES = ('u', 'u_t')
LAB1_GRAPH = (LAPLACIAN,)
CORRECTIONS = tuple(branch for leaf in LEAVES for branch in (
    (leaf,), ('lap', (leaf,)), *((a, (leaf,)) for a in AXES),
    *((a, (b, (leaf,))) for a, b in combinations_with_replacement(AXES, 2)),
))
SPACING = (0.025, 2 / 24, 2 / 24, 2 / 24)


def benchmark(damping, field_seeds):
    """Independent periodic fields with the same PDE and different modes and phases."""
    wavevectors = np.array([(1, 0, 1), (1, 1, 0), (1, -1, 0), (0, 1, 1),
                            (2, 1, -1), (1, 2, 1), (0, 0, 1), (1, -1, -2),
                            (2, -1, 0), (1, 0, -2), (0, 2, -1), (2, -2, 1),
                            (3, 1, 0), (2, 2, -1), (1, 3, -1), (1, -2, 0)])
    fields = []
    splits = []
    for i, seed in enumerate(field_seeds):
        rng = np.random.default_rng(seed)
        frames = 96 if i == 0 else 48
        axes = [np.arange(n) * h for n, h in zip((frames, 24, 24, 24), SPACING, strict=True)]
        t, x, y, z = np.ix_(*axes)
        field = np.zeros((frames, 24, 24, 24), dtype=np.float32)
        k = np.pi * wavevectors[rng.choice(len(wavevectors), 10, replace=False)]
        omega = np.sqrt(0.85 * np.sum(k ** 2, axis=1) + 0.72 * k[:, 0] * k[:, 1] - damping ** 2 / 4)
        for (kx, ky, kz), w in zip(k, omega, strict=True):
            space = kx * x + ky * y + kz * z
            a, b = rng.uniform(0.25, 0.8, size=2)
            p, q = rng.uniform(0, 2 * np.pi, size=2)
            field += np.exp(-damping * t / 2) * (a * np.cos(space - w * t + p) + b * np.cos(space + w * t + q))
        start = splits[-1][0].stop if splits else 0
        splits.append((slice(start, start + frames), slice(None)))
        fields.append(field)
    search_start = (MIXED, LAPLACIAN) if damping else LAB1_GRAPH
    terms = {MIXED: 0.72, LAPLACIAN: 0.85}
    if damping:
        terms[('u_t',)] = -damping
    truth = canonical_graph(terms)
    data = WaveData('Damped mixed waves' if damping else 'Mixed waves', np.concatenate(fields), SPACING,
                    tuple(splits), truth, 'Periodic waves with independent mode mixtures in each split.')
    return data, search_start, [terms[branch] for branch in truth]


def choose_extension(paths, rng, denoiser):
    """Reveal a legal derivation among the remaining one-branch extensions."""
    state = Derivation(LEAVES)
    remaining = list(paths)
    while state.pending:
        i = len(state.tokens)
        legal = sorted({tokens[i] for tokens in remaining})
        rule = sample_rule(state, legal, rng, denoiser)
        state.add(rule)
        remaining = [tokens for tokens in remaining if tokens[i] == rule]
    return tuple(state.tokens)


def search_extensions(data, scorer, base, method, seed, budget):
    rng = np.random.default_rng(seed)
    denoiser = None if method == METHODS[2] else Denoiser(seed)
    base_fit = scorer.evaluate(base)
    history = [base_fit]
    training = [encode(base, LEAVES)]
    paths = {tuple(encode((*base, branch), LEAVES)): canonical_graph((*base, branch))
             for branch in CORRECTIONS if branch not in base}
    proposals = []
    for i in range(budget - 1):
        if denoiser is not None and i % 4 == 0:
            denoiser.update(training, LEAVES, weights(history, 0.035 if method == METHODS[0] else 0.0),
                            rng, steps=50)
        tokens = choose_extension(paths, rng, denoiser)
        evaluation = scorer.evaluate(paths.pop(tokens))
        history.append(evaluation)
        training.append(tokens)
        proposals.append(dict(reveal_order=tokens, **fit_record(evaluation)))
    return dict(method=method, seed=seed, selected=finalist(scorer, history, data.truth), proposals=proposals)


def spatial_symbols(shape, spacing, graph, spectral=False):
    freq = np.ix_(*(2 * np.pi * np.fft.fftfreq(n) for n in shape))
    if spectral:
        first = {axis: 1j * np.where(np.isclose(np.abs(k), np.pi), 0, k) / h
                 for axis, k, h in zip(AXES, freq, spacing, strict=True)}
        second = {axis: -(k / h) ** 2 for axis, k, h in zip(AXES, freq, spacing, strict=True)}
    else:
        first = {axis: 1j * np.sin(k) / h for axis, k, h in zip(AXES, freq, spacing, strict=True)}
        second = {axis: -4 * np.sin(k / 2) ** 2 / h ** 2 for axis, k, h in zip(AXES, freq, spacing, strict=True)}

    def symbol(expr):
        op, *args = expr
        if not args:
            return op, 1
        if op == 'lap':
            return args[0][0], sum(second.values())
        child, *rest = args[0]
        if not rest:
            return child, first[op]
        return rest[0][0], second[op] if child == op else first[op] * first[child]

    return [(leaf == 'u_t', np.broadcast_to(value, shape)) for leaf, value in map(symbol, graph)]


def modal_frames(initial, dt, symbols, theta, steps):
    a = sum(coef * value for (velocity, value), coef in zip(symbols, theta, strict=True) if not velocity)
    b = sum(coef * value for (velocity, value), coef in zip(symbols, theta, strict=True) if velocity)
    root = np.sqrt(b ** 2 + 4 * a + 0j)
    with np.errstate(over='ignore', invalid='ignore'):
        r1, r2 = np.exp(dt * (b + root) / 2), np.exp(dt * (b - root) / 2)
    prev, current = initial
    for _ in range(steps):
        with np.errstate(over='ignore', invalid='ignore'):
            prev, current = current, (r1 + r2) * current - r1 * r2 * prev
        yield current


def forecast_frames(initial, spacing, graph, theta, steps, *, spectral=False):
    """Yield future fields using only the fitted PDE and two initial snapshots."""
    symbols = spatial_symbols(initial.shape[1:], spacing[1:], graph, spectral)
    modes = np.fft.fftn(initial, axes=(1, 2, 3))
    for forecast in modal_frames(modes, spacing[0], symbols, theta, steps):
        yield np.fft.ifftn(forecast).real


def rollout(field, spacing, graph, theta, *, spectral=False):
    predictions = forecast_frames(field[:2], spacing, graph, theta, len(field) - 2, spectral=spectral)
    return [normalized_error(actual, predicted)
            for actual, predicted in zip(field[2:], predictions, strict=True)]


def test_forecasts(data, tests, model):
    estimates = []
    for split, rows in enumerate(tests, 2):
        graph, theta = model['graph'], model['theta']
        time, space = data.splits[split]
        forecast = rollout(data.u[time, space], data.spacing, graph, theta)
        estimates.append(dict(weak_nmse=normalized_error(rows.target, rows.matrix(graph) @ theta),
                              rollout_nmse=float(np.mean(forecast)), rollout_curve=forecast))
    return estimates


def run_benchmarks(budget):
    max_budget = len(CORRECTIONS) - 1
    if not 2 <= budget <= max_budget:
        raise ValueError(f'Budget must lie between 2 and {max_budget}, including the starting model.')
    torch.set_num_threads(1)
    results = []
    search_seeds = (11, 22, 33, 44, 55)
    field_seeds = (101, 202, 303, 404, 505)
    with threadpool_limits(limits=1):
        for damping in (0.0, 2.2):
            data, search_start, coefficients = benchmark(damping, field_seeds)
            train, val, *tests = (make_blocks(data, split) for split in range(len(field_seeds)))
            mild, strong = ([make_blocks(data, 0, w, seed) for seed in INTERVENTION_SEEDS] for w in WINDOWS)
            scorer = Scorer(train, val, tests[0], mild, strong, coefficient_sign)
            baseline = finalist(scorer, [scorer.evaluate(LAB1_GRAPH)], data.truth)
            oracle = finalist(scorer, [scorer.evaluate(data.truth)], data.truth)
            runs = [search_extensions(data, scorer, search_start, method, seed, budget)
                    for seed in search_seeds for method in METHODS]
            for run in runs:
                run['held_out'] = test_forecasts(data, tests, run['selected'])
            result = dict(name=data.name, truth=data.truth, true_coefficients=coefficients,
                          baseline=baseline, baseline_tests=test_forecasts(data, tests, baseline),
                          oracle=oracle, oracle_tests=test_forecasts(data, tests, oracle), runs=runs)
            if search_start != LAB1_GRAPH:
                start = finalist(scorer, [scorer.evaluate(search_start)], data.truth)
                result.update(search_start=start, search_start_tests=test_forecasts(data, tests, start))
            results.append(result)
            print(data.name, [(m, sum(r['selected']['exact_recovery'] for r in runs if r['method'] == m))
                              for m in METHODS], flush=True)
    return dict(budget=budget, search_seeds=search_seeds, field_seeds=field_seeds, benchmarks=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=18)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results' / 'extensions.json')
    args = parser.parse_args()
    report = run_benchmarks(args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
