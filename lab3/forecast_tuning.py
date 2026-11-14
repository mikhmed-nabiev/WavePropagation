"""Fit periodic forecasts to multi-step training trajectories and select on validation."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from .experiment import METHODS, safe
from .extensions import CORRECTIONS, benchmark, modal_frames, rollout, spatial_symbols
from .fitting import fit_coefficients
from .grammar import canonical_graph, equation_dag
from .wave_data import coefficient_sign, make_blocks, synthetic


def graph_tuple(expr):
    return (expr[0], *(graph_tuple(child) for child in expr[1:]))


def blocks(data, split):
    time, space = data.splits[split]
    return data.u[time, space]


def fit_rollout(training, spacing, graph, initial, horizon, spectral):
    """By Parseval, modal residuals give the spatial forecast objective."""
    windows = np.stack([training[i:i + horizon] for i in range(0, len(training) - horizon + 1, horizon)])
    transformed = np.fft.fftn(windows, axes=(2, 3, 4))
    energy = np.sum(np.abs(transformed) ** 2, axis=(0, 1))
    active = energy > energy.max() * 1e-12
    target = transformed[:, :, active]
    scale = np.linalg.norm(target[:, 2:], axis=-1, keepdims=True)
    symbols = [(is_velocity, value[active]) for is_velocity, value in
               spatial_symbols(training.shape[1:], spacing[1:], graph, spectral)]
    seed = target[:, :2].swapaxes(0, 1)

    def residual(theta):
        prediction = np.stack(list(modal_frames(seed, spacing[0], symbols, theta, horizon - 2)), axis=1)
        with np.errstate(over='ignore', invalid='ignore'):
            error = (prediction - target[:, 2:]) / scale
        real_error = np.ascontiguousarray(error).view(float).ravel()
        return np.nan_to_num(real_error, nan=1e6, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)

    signs = np.array([coefficient_sign(branch) for branch in graph])
    optimum = least_squares(residual, initial, bounds=(np.where(signs > 0, 0., -np.inf), np.inf), max_nfev=80)
    return optimum.x


def model(graph, theta, spectral=False):
    return dict(graph=graph, theta=list(map(float, theta)), dag=equation_dag(graph), spectral=spectral)


def scores(field, spacing, fitted):
    errors = rollout(field, spacing, fitted['graph'], fitted['theta'], spectral=fitted['spectral'])
    return dict(mean=float(np.mean(errors)), final=errors[-1], curve=errors)


def proposal_branches(output):
    branches = {}
    with gzip.open(output / 'proposals.jsonl.gz', 'rt', encoding='utf-8') as log:
        for line in log:
            record = json.loads(line)
            if record['method'] == METHODS[0] and not record['initial'] and not record['duplicate']:
                branches.setdefault(record['dataset'], set()).update(map(graph_tuple, record['graph']))
    return branches


def tune_benchmark(data, baseline, selected, discovered=(), extra_test=()):
    train, val = (blocks(data, split) for split in (0, 1))
    train_rows = make_blocks(data, 0)
    horizon = len(val)
    base_graph, selected_graph = (tuple(map(graph_tuple, record['graph'])) for record in (baseline, selected))

    def tune(graph, theta, spectral):
        return model(graph, fit_rollout(train, data.spacing, graph, theta, horizon, spectral), spectral)

    original = model(selected_graph, selected['theta'])
    same_fd = tune(selected_graph, original['theta'], False)
    same_spectral = tune(selected_graph, original['theta'], True)
    candidates = [original, same_fd, same_spectral]
    # Recombine generated branches by replacing one term in the selected graph.
    neighbors = {canonical_graph((*selected_graph[:i], *selected_graph[i+1:], branch))
                 for i in range(len(selected_graph)) for branch in set(discovered).intersection(CORRECTIONS)}
    for graph in sorted(neighbors - {selected_graph}):
        fit = fit_coefficients(train_rows.matrix(graph), train_rows.target, train_rows.groups,
                               [coefficient_sign(branch) for branch in graph])
        if fit.rank == len(graph) and fit.kept.all():
            candidates.append(tune(graph, fit.theta, True))
    validation = [scores(val, data.spacing, candidate)['mean'] for candidate in candidates]
    chosen = candidates[int(np.argmin(validation))]
    models = {'Lab 1, rollout-tuned': tune(base_graph, baseline['theta'], True),
              'Original selected forecast': original, 'Same graph, rollout-tuned (finite differences)': same_fd,
              'Same graph, rollout-tuned (spectral)': same_spectral, 'Validation-selected forecast': chosen}
    tests = [blocks(data, split) for split in range(2, len(data.splits))]
    result = dict(name=data.name, models=models, candidate_count=len(candidates),
                  validation={label: scores(val, data.spacing, fitted)['mean'] for label, fitted in models.items()},
                  test={label: [scores(field, data.spacing, fitted) for field in tests] for label, fitted in models.items()},
                  exact_structure=chosen['graph'] == data.truth)
    if extra_test:
        result['extra_test'] = {label: [scores(field, spacing, fitted)['mean'] for field, spacing in extra_test]
                                for label, fitted in models.items()}
    return result


def run(output):
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    extensions = json.loads((output / 'extensions.json').read_text(encoding='utf-8'))
    branches = proposal_branches(output)
    results = []
    fresh_seeds = (707, 808, 909)
    with threadpool_limits(limits=1):
        fresh, _, _ = benchmark(0., fresh_seeds)
        fresh_fields = tuple((blocks(fresh, i), fresh.spacing) for i in range(len(fresh_seeds)))
        for result in summary['datasets']:
            if result['name'] in ('Isotropic waves', 'Mixed derivative'):
                data = synthetic(mixed=result['name'] == 'Mixed derivative', periodic=True)
                selected = result['best']
                if all(graph_tuple(branch) in CORRECTIONS for branch in selected['graph']):
                    extra = fresh_fields if result['name'] == 'Mixed derivative' else ()
                    results.append(tune_benchmark(data, result['lab1'], selected, branches.get(data.name, ()), extra))
        for result in extensions['benchmarks']:
            damping = -result['true_coefficients'][result['truth'].index(['u_t'])] if ['u_t'] in result['truth'] else 0.
            data, _, _ = benchmark(damping, extensions['field_seeds'])
            guided = min((r for r in result['runs'] if r['method'] == METHODS[0]), key=lambda r: r['selected']['score'])
            results.append(tune_benchmark(data, result['baseline'], guided['selected']))
    for result in results:
        print(result['name'], {label: float(np.mean([test['mean'] for test in tests]))
                              for label, tests in result['test'].items()}, flush=True)
    return dict(benchmarks=results, field_seeds=extensions['field_seeds'], fresh_seeds=fresh_seeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results')
    output = parser.parse_args().results
    results = run(output)
    (output / 'forecast_tuning.json').write_text(json.dumps(results, allow_nan=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
