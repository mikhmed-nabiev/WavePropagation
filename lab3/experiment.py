"""Run paired graph searches with a fixed budget of unique evaluations."""
import argparse
from dataclasses import replace
import gzip
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .diffusion import Denoiser, sample
from .fitting import Scorer, fit_coefficients, normalized_error
from .grammar import LAPLACIAN, MIXED, PURE, canonical_graph, encode, format_expr, equation_dag
from .wave_data import coefficient_sign, load_meshrir, make_blocks, sanity_data, synthetic


METHODS = ('guided diffusion', 'unguided diffusion', 'random grammar')
WINDOWS = (2, 12)
AUDIT_WINDOWS = (1, 2, 4, 8, 12, 24)
INTERVENTION_SEEDS = (811, 812, 813)
AUDIT_SEEDS = (*INTERVENTION_SEEDS, 814, 815, 816, 817, 818, 819)
INITIAL = 12
BATCH = 20


def contains(expr, target):
    return expr == target or any(contains(child, target) for child in expr[1:])


def initial_pool(leaves, seed, exclude):
    rng = np.random.default_rng(seed)
    initial = list(dict.fromkeys([(LAPLACIAN,), canonical_graph([LAPLACIAN, ('f',)])]
                                 if 'f' in leaves else [(LAPLACIAN,)]))
    while len(initial) < INITIAL:
        graph, _ = sample(leaves, rng)
        if graph not in initial and not (exclude and any(contains(branch, MIXED) for branch in graph)):
            initial.append(graph)
    return initial


def weights(evaluations, stability):
    scores = np.array([e.score(stability) + 0.05 * (1 - e.fit.kept.mean()) for e in evaluations])
    return 0.075 + np.exp(-(scores - scores.min()) / 0.12)


def safe(array):
    return [float(value) if np.isfinite(value) else None for value in array]


def fit_record(e):
    return dict(graph=e.graph, dag=equation_dag(e.graph), terms=[format_expr(x) for x in e.graph],
                theta=safe(e.fit.theta), sd=safe(e.fit.sd), z=safe(e.fit.z), kept=e.fit.kept.tolist(),
                rank=e.fit.rank, scaled_condition=float(e.fit.condition) if np.isfinite(e.fit.condition) else None,
                validation=e.validation, score=e.score(), stability=e.mild_support.tolist(),
                strong_support=e.strong_support.tolist(), complexity=e.complexity, eligible=bool(e.eligible))


def finalist(scorer, candidates, truth):
    eligible = [e for e in candidates if e.eligible]
    best = min(eligible or candidates, key=lambda e: e.score())
    error, _ = scorer.heldout(best)
    return dict(**fit_record(best), test_nmse=error,
                exact_recovery=None if truth is None else bool(best.eligible and best.graph == truth),
                ablations=[dict(term=format_expr(term), refitted_theta=safe(theta), delta_validation=delta)
                           for term, theta, delta in scorer.ablate(best)])


def search(data, scorer, leaves, initial, seed, method, budget, logs):
    rng = np.random.default_rng(seed + 4000)
    train_rng = np.random.default_rng(seed + 5000)
    denoiser = None if method == METHODS[2] else Denoiser(seed)
    penalty = 0.035 if method == METHODS[0] else 0.0
    seen = {}
    duplicates = 0
    first_truth = None
    first_mixed_branch = None
    incumbent = None
    curve = []

    def record(graph, prefix, round_id, initial=False):
        nonlocal duplicates, incumbent, first_truth, first_mixed_branch
        duplicate = graph in seen
        if duplicate:
            duplicates += 1
        else:
            e = seen[graph] = scorer.evaluate(graph)
            if graph == data.truth and first_truth is None:
                first_truth = len(seen)
            if MIXED in graph and first_mixed_branch is None:
                first_mixed_branch = len(seen)
            if e.eligible and (incumbent is None or e.score() < incumbent):
                incumbent = e.score()
            curve.append(incumbent)
        logs.write(json.dumps(dict(dataset=data.name, method=method, seed=seed, round=round_id,
                                   initial=initial, duplicate=duplicate, unique_evaluations=len(seen),
                                   reveal_order=prefix,
                                   **(dict(graph=graph) if duplicate else fit_record(seen[graph]))),
                              allow_nan=False) + '\n')

    for graph in initial:
        record(graph, encode(graph, leaves), 0, initial=True)
    round_id = 0
    while len(seen) < budget:
        round_id += 1
        if denoiser is not None:
            denoiser.update([encode(graph, leaves) for graph in seen], leaves,
                            weights(list(seen.values()), penalty), train_rng, steps=100 if round_id == 1 else 55)
        goal = min(len(seen) + BATCH, budget)
        attempts = 0
        while len(seen) < goal:
            graph, trajectory = sample(leaves, rng, denoiser)
            record(graph, trajectory, round_id)
            attempts += 1
            if attempts >= 15000 and len(seen) < goal:
                raise RuntimeError('Candidate budget could not be filled with distinct graphs.')
    result = finalist(scorer, list(seen.values()), data.truth)
    return dict(seed=seed, method=method, curve=curve, duplicates=duplicates, unique=len(seen),
                first_truth=first_truth, first_mixed_branch=first_mixed_branch, finalist=result)


def fixed_mask(scorer, leaves, truth):
    # Prune directional edges while preserving the shared Lab 1 speed coefficient.
    graphs = []
    for count in range(1, len(PURE) + 1):
        for terms in combinations(PURE, count):
            branch = terms[0]
            for term in terms[1:]:
                branch = ('add', branch, term)
            graphs.append(canonical_graph([branch]))
            if 'f' in leaves:
                graphs.append(canonical_graph([branch, ('f',)]))
    return dict(unique=len(graphs), finalist=finalist(scorer, [scorer.evaluate(g) for g in graphs], truth))


def audit_finalists(data, scorer, selected_graph):
    """Stress-test frozen graph structures without changing search or selection."""
    graphs = {'Selected graph': selected_graph}
    if data.truth is not None and data.truth != selected_graph:
        graphs['Known equation'] = data.truth
    shuffle = {label: [] for label in graphs}
    for window in AUDIT_WINDOWS:
        fits = {label: [] for label in graphs}
        for draw, seed in enumerate(AUDIT_SEEDS if window > 1 else (0,)):
            if window in WINDOWS and draw < len(INTERVENTION_SEEDS):
                rows = (scorer.mild if window == WINDOWS[0] else scorer.strong)[draw]
            else:
                rows = scorer.train if window == 1 else make_blocks(data, 0, window, seed)
            for label, graph in graphs.items():
                fit = scorer.fit(graph, rows)
                clean = scorer.evaluate(graph).fit
                validation = scorer.validation
                fits[label].append(dict(theta=safe(fit.theta), z=safe(fit.z),
                                        retained=(clean.kept & fit.kept & (clean.theta * fit.theta > 0)).tolist(),
                                        clean_validation_nmse=normalized_error(validation.target,
                                                                               validation.matrix(graph) @ fit.theta)))
        for label, draws in fits.items():
            shuffle[label].append(dict(window=window, draws=draws))

    reversal = None
    if data.truth is not None:
        time, _ = data.splits[2]
        reversed_field = data.u.copy()
        reversed_field[time] = reversed_field[time][::-1]
        reversed_test = make_blocks(replace(data, u=reversed_field), 2)
        reversal = {}
        for label, graph in graphs.items():
            evaluation = scorer.evaluate(graph)
            prediction = reversed_test.matrix(graph) @ evaluation.fit.theta
            reversal[label] = dict(forward_nmse=scorer.heldout(evaluation)[0],
                                   reversed_nmse=normalized_error(reversed_test.target, prediction))
    return dict(terms={label: [format_expr(branch) for branch in graph] for label, graph in graphs.items()},
                shuffle=shuffle, time_reversal=reversal, shuffle_draws=len(AUDIT_SEEDS))


def run_experiments(output, meshrir=None, seeds=(11, 22, 33, 44, 55), budget=180, progress=print):
    if budget < INITIAL:
        raise ValueError(f'The budget must include at least the {INITIAL} shared initial graphs.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    real_path = Path(meshrir) if meshrir is not None else Path(__file__).parent / 'data' / 'S1-M3969_npy'
    real = load_meshrir(real_path) if meshrir is not None or real_path.is_dir() else None
    mesh_access = ('Measured data were processed from the verified local grid.' if real is not None else
                   'Measured run pending. The official Zenodo record was reachable, but the archive '
                   'was unavailable to this execution environment and no local copy was present.')
    report = dict(budget=budget, seeds=list(seeds), windows=list(WINDOWS), mesh_access=mesh_access, datasets=[])
    with threadpool_limits(limits=1), gzip.open(output / 'proposals.jsonl.gz', 'wt', encoding='utf-8') as logs:
        original = sanity_data()
        fit = fit_coefficients(original.matrix((LAPLACIAN,)), original.target, original.groups, [1])
        directional = fit_coefficients(original.matrix(PURE), original.target, original.groups, [1] * 3)
        report['lab1_check'] = dict(speed=float(np.sqrt(fit.theta[0])), directional_rank=directional.rank)
        datasets = (synthetic(), synthetic(mixed=True)) if real is None else (synthetic(), synthetic(mixed=True), real)
        for data in datasets:
            progress(f'{data.name}')
            leaves = ('u', 'u_t', *(data.inputs or {}))
            train, val, test = (make_blocks(data, split) for split in range(3))
            mild, strong = ([make_blocks(data, 0, window, seed) for seed in INTERVENTION_SEEDS] for window in WINDOWS)
            scorer = Scorer(train, val, test, mild, strong, coefficient_sign)
            result = dict(name=data.name, description=data.description, spacing=data.spacing,
                          truth=None if data.truth is None else [format_expr(x) for x in data.truth],
                          sample_counts=[len(d.target) for d in (train, val, test)],
                          fixed_mask=fixed_mask(scorer, leaves, data.truth), runs=[])
            if 'f' not in leaves:
                result['lab1'] = finalist(scorer, [scorer.evaluate((LAPLACIAN,))], data.truth)
            pure_fit = scorer.fit(PURE, train)
            result['directional_diagnostics'] = dict(rank=pure_fit.rank, condition=safe([pure_fit.condition])[0])
            for seed in seeds:
                pool = initial_pool(leaves, seed, exclude=data.truth is not None and MIXED in data.truth)
                for method in METHODS:
                    run = search(data, scorer, leaves, pool, seed, method, budget, logs)
                    result['runs'].append(run)
                    progress(f"  {seed} {method}: {', '.join(run['finalist']['terms'])}; "
                             f"score {run['finalist']['score']:.6g}, duplicates {run['duplicates']}")
            best = min((r['finalist'] for r in result['runs']), key=lambda r: (not r['eligible'], r['score']))
            graph = best['graph']
            _, prediction = scorer.heldout(scorer.cache[graph])
            mask = ((test.centers[:, 1:] == test.centers[0, 1:]).all(axis=1) & (test.groups == 0))
            indices = np.linspace(0, len(prediction) - 1, min(len(prediction), 350), dtype=int)
            result['prediction'] = dict(time_index=test.centers[mask, 0].tolist(),
                                        trace_observed=safe(test.target[mask]), trace_predicted=safe(prediction[mask]),
                                        observed=safe(test.target[indices]), predicted=safe(prediction[indices]),
                                        terms=best['terms'])
            result['best'] = best
            if data.truth is not None:
                result['oracle'] = finalist(scorer, [scorer.evaluate(data.truth)], data.truth)
            result['audit'] = audit_finalists(data, scorer, graph)
            report['datasets'].append(result)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    from .report import create_figures

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--meshrir', type=Path, default=os.getenv('WAVE_MESHRIR'))
    parser.add_argument('--budget', type=int, default=180)
    parser.add_argument('--seeds', nargs='+', type=int, default=(11, 22, 33, 44, 55))
    args = parser.parse_args()
    report = run_experiments(args.output, args.meshrir, args.seeds, args.budget)
    create_figures(report, args.output)


if __name__ == '__main__':
    main()
