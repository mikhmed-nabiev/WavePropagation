"""Evaluate the same forecast post-processing on the guided and random proposal logs."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .experiment import METHODS
from .extensions import CORRECTIONS, LAB1_GRAPH, benchmark
from .fitting import fit_coefficients
from .forecast_tuning import blocks, fit_rollout, graph_tuple, model, scores
from .grammar import canonical_graph
from .wave_data import coefficient_sign, make_blocks, synthetic


def compare(output):
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    mixed = next(d for d in summary['datasets'] if d['name'] == 'Mixed derivative')
    tuning = json.loads((output / 'forecast_tuning.json').read_text(encoding='utf-8'))
    methods = (METHODS[0], METHODS[2])
    branches = {(run['method'], run['seed']): set() for run in mixed['runs'] if run['method'] in methods}
    legal = set(CORRECTIONS) - set(LAB1_GRAPH)
    with gzip.open(output / 'proposals.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            key = record['method'], record['seed']
            if record['dataset'] == mixed['name'] and key in branches and not record['initial'] and not record['duplicate']:
                branches[key].update(legal.intersection(map(graph_tuple, record['graph'])))

    data = synthetic(mixed=True, periodic=True)
    train, val, test = (blocks(data, split) for split in range(3))
    train_rows = make_blocks(data, 0)
    fresh, _, _ = benchmark(0., tuning['fresh_seeds'])
    fresh_fields = [blocks(fresh, i) for i in range(len(fresh.splits))]

    # Use a common Laplacian anchor, and fit each proposed correction once.
    fits = {}
    with threadpool_limits(limits=1):
        for branch in [None, *sorted(set().union(*branches.values()))]:
            graph = LAB1_GRAPH if branch is None else canonical_graph((*LAB1_GRAPH, branch))
            fit = fit_coefficients(train_rows.matrix(graph), train_rows.target, train_rows.groups,
                                   [coefficient_sign(term) for term in graph])
            if fit.rank == len(graph) and fit.kept.all():
                theta = fit_rollout(train, data.spacing, graph, fit.theta, len(val), True)
                fitted = model(graph, theta, spectral=True)
                fits[branch] = (scores(val, data.spacing, fitted)['mean'], fitted)

    rows = []
    for (method, seed), proposed in branches.items():
        candidates = [fits[branch] for branch in [None, *sorted(proposed)] if branch in fits]
        validation, chosen = min(candidates, key=lambda entry: entry[0])
        rows.append(dict(method=method, seed=seed, candidates=len(candidates), exact=chosen['graph'] == data.truth,
                         graph=chosen['graph'], theta=chosen['theta'], validation=validation,
                         held_out=scores(test, data.spacing, chosen)['mean'],
                         fresh=[scores(field, fresh.spacing, chosen)['mean'] for field in fresh_fields]))
    return dict(budget=summary['budget'], fresh_seeds=tuning['fresh_seeds'], runs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results')
    output = parser.parse_args().results
    result = compare(output)
    (output / 'sampler_comparison.json').write_text(json.dumps(result, allow_nan=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
