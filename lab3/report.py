"""Generate the notebook's tables and plots from a reproducible experiment run."""
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from wave_equation_tree import NodeColor, WaveEquationDAG, draw_dag

from .experiment import INITIAL, METHODS
from .grammar import Derivation


COLORS = ('#146a63', '#30569f', '#bd542e')
REFERENCES = '''- Austin et al. (2021), [Structured Denoising Diffusion Models in Discrete State-Spaces](https://papers.neurips.cc/paper_files/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html).
- Kusner et al. (2017), [Grammar Variational Autoencoder](https://proceedings.mlr.press/v70/kusner17a.html).
- Messenger and Bortz (2021), [Weak SINDy for Partial Differential Equations](https://doi.org/10.1016/j.jcp.2021.110525).
- Koyama et al. (2021), [MeshRIR](https://doi.org/10.5281/zenodo.10852693).
- [Membrane and temperature dissipation, Lab 2](https://github.com/mikhmed-nabiev/Membrane-and-temperature-dissipation/tree/main/lab2-heat), coefficient support and temporal interventions.'''


def method_rows(dataset):
    for method in METHODS:
        runs = [r for r in dataset['runs'] if r['method'] == method]
        finals = [r['finalist'] for r in runs]
        yield (method, sum(f['exact_recovery'] is True for f in finals), len(runs),
               np.median([f['score'] for f in finals]), np.median([f['test_nmse'] for f in finals]),
               np.median([r['duplicates'] for r in runs]), sum(r['first_truth'] is not None for r in runs))


def plot_scores(dataset):
    fig, (ax, test) = plt.subplots(1, 2, figsize=(10.6, 3.7), layout='constrained',
                                   gridspec_kw={'width_ratios': [3, 2]})
    for method, color, style, width in zip(METHODS, COLORS, ('-', '--', ':'), (3.0, 2.0, 1.3), strict=True):
        runs = [r for r in dataset['runs'] if r['method'] == method]
        curves = np.array([r['curve'][INITIAL - 1:] for r in runs], dtype=float)
        x = np.arange(curves.shape[1])
        # No eligible model is represented by a gap before all seeds have one.
        valid = np.isfinite(curves).all(axis=0)
        ax.step(x[valid], np.median(curves[:, valid], axis=0), where='post', color=color, label=method,
                linestyle=style, linewidth=width)
        ax.fill_between(x[valid], *np.quantile(curves[:, valid], [0.25, 0.75], axis=0),
                        step='post', color=color, alpha=0.13)
        seeds = np.arange(len(runs))
        offset = (METHODS.index(method) - 1) * 0.16
        test.scatter(seeds + offset, [r['finalist']['test_nmse'] for r in runs], color=color, s=35, zorder=3)
    oracle = dataset.get('oracle')
    if oracle and oracle['eligible']:
        ax.axhline(oracle['score'], color='#596579', ls='--', lw=1.1, label='Known equation')
        test.axhline(oracle['test_nmse'], color='#596579', ls='--', lw=1.1)
    ax.set(xlabel='New unique evaluations after shared pool', ylabel='Best eligible validation score', yscale='log')
    test.set(xlabel='Paired search seed', ylabel='Finalist held-out NMSE', yscale='log',
             xticks=seeds, xticklabels=[r['seed'] for r in runs])
    for panel in (ax, test):
        panel.grid(axis='y', alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc='outside lower center', fontsize=9)
    return fig


def plot_interventions(dataset):
    audit = dataset['audit']
    fig, (coef, error) = plt.subplots(1, 2, figsize=(10.6, 3.7), layout='constrained')
    for (label, levels), color, style in zip(audit['shuffle'].items(), COLORS, ('-', '--')):
        windows = [level['window'] for level in levels]
        positions = np.arange(len(windows))
        for j, term in enumerate(audit['terms'][label]):
            quantiles = np.array([np.quantile([d['theta'][j] for d in level['draws']], [0.25, 0.5, 0.75])
                                  for level in levels])
            marker = ('o', 's', 'D', '^')[j]
            coef.plot(positions, quantiles[:, 1], style, marker=marker, color=color, ms=4,
                      label=f"{label} · {term.replace('D𝓏', 'D_z')}")
            coef.fill_between(positions, quantiles[:, 0], quantiles[:, 2], color=color, alpha=0.12)
        quantiles = np.array([np.quantile([d['clean_validation_nmse'] for d in level['draws']], [0.25, 0.5, 0.75])
                              for level in levels])
        error.plot(positions, quantiles[:, 1], style, marker='o', color=color, ms=4, label=label)
        error.fill_between(positions, quantiles[:, 0], quantiles[:, 2], color=color, alpha=0.12)
    coef.set_ylabel('Refitted coefficient')
    coef.set_yscale('symlog', linthresh=1)
    coef.axhline(0, color='#596579', lw=0.8)
    error.set(ylabel='NMSE on clean validation data', yscale='log')
    error.axhline(1, color='#596579', ls=':', lw=1.0, label='Zero prediction')
    for ax in (coef, error):
        ax.set(xlabel='Shuffle window (samples)', xticks=positions, xticklabels=['1 · clean', *windows[1:]])
        ax.grid(axis='y', alpha=0.2)
        ax.legend(frameon=False, loc='upper left' if ax is coef else 'lower right', fontsize=8)
    return fig


def plot_predictions(dataset):
    pred = dataset['prediction']
    fig, (trace, scatter) = plt.subplots(1, 2, figsize=(8.9, 3.15), layout='constrained')
    times = np.array(pred['time_index']) * dataset['spacing'][0]
    trace.plot(times, pred['trace_observed'], 'o-', color='#253349', label='Field-derived', ms=4)
    trace.plot(times, pred['trace_predicted'], 's--', color=COLORS[0], label='Model', ms=4)
    trace.set(xlabel='Time from block start (model units)', ylabel='Weak acceleration')
    trace.legend(frameon=False)
    observed, predicted = pred['observed'], pred['predicted']
    limits = [min(*observed, *predicted), max(*observed, *predicted)]
    scatter.plot(limits, limits, ':', color='#253349')
    scatter.scatter(observed, predicted, color=COLORS[0], alpha=0.42, s=11)
    scatter.set(xlabel='Field-derived weak acceleration', ylabel='Predicted weak acceleration')
    for ax in (trace, scatter):
        ax.grid(alpha=0.2)
    return fig


def symbolic_branch(expr):
    """Render grammar nodes using the differential notation of the notebook."""
    op, *args = expr
    if not args:
        return {'u_t': r'\partial_t u'}.get(op, op)
    if op.startswith('D_'):
        indices = [op[-1]]
        child = args[0]
        while child[0].startswith('D_'):
            indices.append(child[0][-1])
            child = child[1]
        if child[0] == 'u_t':
            indices.append('t')
            child = ['u']
        inner = symbolic_branch(child)
        if len(child) > 1:
            inner = r'\left(' + inner + r'\right)'
        return r'\partial_{' + ''.join(indices) + '} ' + inner
    values = [symbolic_branch(arg) for arg in args]
    if op == 'lap':
        return r'\Delta u' if args[0][0] == 'u' else r'\Delta\left(' + values[0] + r'\right)'
    if op in ('add', 'mul', 'dot', 'scale'):
        separator = ' + ' if op == 'add' else r'\,\cdot\,'
        return r'\left(' + separator.join(values) + r'\right)'
    if op in ('square', 'cube'):
        return r'\left(' + values[0] + r'\right)^' + ('2' if op == 'square' else '3')
    if op == 'grad':
        return r'\nabla\left(' + values[0] + r'\right)'
    if op == 'div':
        return r'\nabla\!\cdot\!\left(' + values[0] + r'\right)'
    if op == 'vec':
        return r'\left(' + ', '.join(values) + r'\right)'
    return '\\' + op + r'\left(' + values[0] + r'\right)'


def symbolic_equations(records):
    sections = []
    for label, model in records:
        terms = []
        for branch, value in zip(model['graph'], model['theta'], strict=True):
            expression = symbolic_branch(branch)
            if value is None:
                coefficient = r'\theta_{' + str(len(terms) + 1) + '}'
            else:
                coefficient = str(abs(value))
                if 'e' in coefficient:
                    mantissa, exponent = coefficient.split('e')
                    coefficient = mantissa + r'\times 10^{' + str(int(exponent)) + '}'
            prefix = ('- ' if value is not None and value < 0 else '+ ') if terms else (
                '-' if value is not None and value < 0 else '')
            terms.append(prefix + coefficient + r'\,' + expression)
        sections.append(f'**{label}**\n\n$$u_{{tt}} = {" ".join(terms)}$$')
    return '\n\n'.join(sections)


def display_forecast_views(coords, fields, animation, *, appendix=None, title='Forecast'):
    if appendix is not None:
        appendix.append((title, coords, fields, animation))
        return
    from IPython.display import Image, display
    from .forecast_plots import field_snapshots, forecast_diagnostics, wavefront_maps, wavefront_surfaces

    if animation.is_file():
        display(Image(filename=str(animation), embed=True, width=720,
                      alt='Animated reference field, Lab 1 forecast and selected forecast'))
    for plot in (field_snapshots, wavefront_maps, wavefront_surfaces, forecast_diagnostics):
        fig = plot(coords, fields)
        display(fig)
        plt.close(fig)


def report_tuned_forecasts(report, output, appendix=None):
    from IPython.display import Markdown, display
    from .forecast_plots import forecast_comparison, tuned_residuals
    from .wave_data import synthetic

    names = ('Lab 1, rollout-tuned', 'Original selected forecast', 'Same graph, rollout-tuned (finite differences)',
             'Same graph, rollout-tuned (spectral)', 'Validation-selected forecast')
    lines = ['<a id="forecast-tuning"></a>\n\n## 5. C — Forecast tuning',
             'Coefficients minimize multi-step field NMSE on training windows. Every window starts with two '
             'snapshots and then rolls forward freely. For the unrestricted finalists, a small structural search '
             'also tries one-term replacements from branches pooled across the guided search seeds. This is '
             'additional post-search optimization, not the original diffusion finalist. The Lab 2 '
             'support rule screens those replacements using training data. The validation rollout selects the equation '
             'and discretization. All subsequent forecasts freeze those choices.',
             'The spectral solver differentiates the periodic wave field in space using Fourier wavenumbers. '
             'This reduces phase error on smooth simulated waves. Lab 1 receives the same spectral solver and rollout '
             'fit for its one coefficient. Table entries are mean held-out field NMSE.',
             '| Benchmark | Tuned Lab 1 | Original selected | Same graph, tuned FD | Same graph, tuned spectral | Validation-selected |\n'
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    rows = []
    for dataset in report['benchmarks']:
        values = [np.mean([test['mean'] for test in dataset['test'][label]]) for label in names]
        rows.append('| ' + dataset['name'] + ' | ' + ' | '.join(f'{value:.6g}' for value in values) + ' |')
    lines[-1] += '\n' + '\n'.join(rows)
    display(Markdown('\n\n'.join(lines)))
    mixed = next((r for r in report['benchmarks'] if r['name'] == 'Mixed derivative'), None)
    if mixed is None:
        return
    display(Markdown('### Fine-tuned mixed-wave forecast'))
    chosen = mixed['models'][names[-1]]
    display(Markdown(symbolic_equations([('Rollout-tuned Lab 1', mixed['models'][names[0]]),
                                         ('Previous finalist', mixed['models'][names[1]]),
                                         ('Validation-selected forecast', chosen)])))
    fig = plot_equation_records([('Previous finalist', mixed['models'][names[1]]), ('Validation-selected forecast', chosen)])
    display(fig)
    plt.close(fig)

    if 'extra_test' in mixed:
        rows = ['| Fresh field seed | Tuned Lab 1 | Previous finalist | Validation-selected |', '| --- | ---: | ---: | ---: |']
        for i, seed in enumerate(report['fresh_seeds']):
            errors = [mixed['extra_test'][label][i] for label in (names[0], names[1], names[-1])]
            rows.append(f'| {seed} | ' + ' | '.join(f'{v:.6g}' for v in errors) + ' |')
        display(Markdown(f'The tuned equation was selected from {mixed["candidate_count"]} forecast variants. '
                         'The same frozen coefficients were then scored on three additional fields with new modes, '
                         'amplitudes, and phases. Errors near $10^{-13}$ are numerical-scale residuals '
                         'on these noise-free synthetic waves; they do not establish this accuracy on measurements.\n\n'
                         + '\n'.join(rows)))

    coords, fields = forecast_comparison(synthetic(mixed=True, periodic=True), (mixed['models'][names[0]], chosen))
    fig, tolerances, crossings, horizon = tuned_residuals(mixed, coords, fields[0])
    headers = '| Model | ' + ' | '.join(f'First above {value}%' for value in tolerances) + ' |'
    rows = [headers, '| --- | ---: | ---: |']
    for label, points in crossings.items():
        times = [f'{t:.3f}' if t is not None else f'Below through {horizon:.3f}' for t in points]
        rows.append(f'| {label} | ' + ' | '.join(times) + ' |')
    display(Markdown('#### Residuals versus forecast lead time\n\n'
                     'Each point summarizes the residual over the full 3D field. Relative RMS divides this '
                     'residual by the reference RMS at the same time. Dotted lines mark 5% and 10%. '
                     'Lead time starts at the last input frame, in simulation time units. '
                     'The table gives the first sampled frame above each threshold.\n\n' + '\n'.join(rows)))
    display(fig)
    plt.close(fig)
    peak_percent = 100 * np.sqrt(max(mixed['test'][names[-1]][0]['curve']))
    from .phase_drift import phase_drift_estimate, plot_phase_drift
    display(Markdown('#### Forecast horizon and conditional phase drift\n\n'
                     f'The directly evaluated rollout reached **{horizon:.3f} simulation time units** after '
                     f'the last input frame, with a peak relative RMS error of **{peak_percent:.6f}%**. '
                     'No divergence threshold was observed for the tuned model within that interval.'))
    if chosen.get('spectral', False):
        try:
            drift = phase_drift_estimate(chosen)
        except ValueError as error:
            display(Markdown(f'An analytic phase estimate is unavailable: {error}'))
        else:
            rows = ['| Relative RMS threshold | Approximate lead time (simulation units) |', '| --- | ---: |']
            for level, time in zip(drift['thresholds'], drift['times'], strict=True):
                value = f'{time:,.3f}' if np.isfinite(time) else 'No phase drift in the ideal model'
                rows.append(f'| {100 * level:g}% | {value} |')
            display(Markdown(r'''For the original nine modes, let $A_j$ be the amplitude and
$\delta\omega_j=\widehat\omega_j-\omega_j$ the frequency error computed from the fitted coefficients.
Spatial orthogonality gives the small-phase approximation

$$E_{\mathrm{RMS}}(\tau)\approx \tau D,\qquad
D=\sqrt{\frac{\sum_j A_j^2(\delta\omega_j)^2}{\sum_j A_j^2}},\qquad
\tau_\varepsilon\approx\frac{\varepsilon}{D}.$$

Here $\widehat\omega_j=\pi\sqrt{\widehat\theta_\Delta\|\mathbf{k}_j\|^2+
\widehat\theta_{xy}k_{jx}k_{jy}}$. The table and plot are recalculated from the saved model at each execution.
They assume unchanged modes and amplitudes, continuous spectral dynamics, and no noise. They omit
two-frame initialization transients and future numerical accumulation. These are conditional extrapolations,
**not evaluated forecast horizons or statistical confidence bounds**.

''' + f'Computed $D={drift["slope"]:.12g}$.\n\n' + '\n'.join(rows)))
            fig = plot_phase_drift(drift, horizon)
            display(fig)
            plt.close(fig)
    display_forecast_views(coords, fields, output / 'figures' / 'tuned_mixed_forecast.gif',
                           appendix=appendix, title='C — Forecast-tuned mixed waves')


def report_sampler_comparison(result, mixed):
    from IPython.display import Markdown, display

    methods = (METHODS[0], METHODS[2])
    rows = ['| Metric | Guided diffusion | Random grammar |', '| --- | ---: | ---: |']
    original = [[r for r in mixed['runs'] if r['method'] == method] for method in methods]
    tuned = [[r for r in result['runs'] if r['method'] == method] for method in methods]
    for label, values in (
        ('Median held-out weak NMSE, original finalists',
         [f"{np.median([r['finalist']['test_nmse'] for r in runs]):.6g}" for runs in original]),
        ('Median duplicate proposals', [f"{np.median([r['duplicates'] for r in runs]):g}" for runs in original]),
        ('Exact recovery after identical forecast tuning', [f"{sum(r['exact'] for r in runs)}/{len(runs)}" for runs in tuned]),
        ('Median held-out forecast NMSE after tuning', [f"{np.median([r['held_out'] for r in runs]):.6g}" for runs in tuned]),
    ):
        rows.append('| ' + ' | '.join([label, *values]) + ' |')
    display(Markdown('<a id="sampler-comparison"></a>\n\n## 6. Guided diffusion versus random search\n\n'
                     f'The unrestricted mixed-wave search used {result["budget"]} unique graph evaluations per seed. '
                     'The original finalists were chosen by the Lab 3 validation score. For the forecast comparison, '
                     'each sampler and seed supplies its own proposed branches. Each eligible branch is paired with '
                     'the Lab 1 Laplacian, fitted on training trajectories using the same spectral solver, and selected '
                     'by validation forecast error. The extra fitting step is a separate comparison after search.\n\n'
                     + '\n'.join(rows)))

    seeds = sorted({r['seed'] for r in result['runs']})
    fig, ax = plt.subplots(figsize=(7.6, 3.15), layout='constrained')
    for runs, label, color, offset in zip(tuned, ('Guided diffusion', 'Random grammar'),
                                          (COLORS[0], COLORS[2]), (-0.11, 0.11), strict=True):
        by_seed = {r['seed']: r for r in runs}
        ax.scatter(np.arange(len(seeds)) + offset, [by_seed[s]['held_out'] for s in seeds],
                   s=43, color=color, label=label)
    ax.set(xlabel='Paired search seed', ylabel='Held-out field NMSE after tuning', yscale='log',
           xticks=np.arange(len(seeds)), xticklabels=seeds)
    ax.grid(axis='y', alpha=0.22)
    ax.legend(frameon=False)
    display(fig)
    plt.close(fig)


def report_extensions(report, output, appendix=None):
    """Report the separate restricted search against the refitted Lab 1 equation."""
    from IPython.display import Markdown, display
    from .forecast_plots import heldout_fields, representative_run, sampler_forecast_errors

    display(Markdown(rf'''<a id="extensions"></a>

## 4. B — One-branch extensions of the baseline

The baseline on both datasets is the Lab 1 equation $u_{{tt}}=c^2\Delta u+f$. These wave fields have no applied source, so $f=0$ and the baseline fits one shared $c^2$. The mixed-wave search starts from the Lab 1 graph. The damped-wave search starts from the undamped mixed-wave graph, which is also reported as an intermediate comparison. Each search adds one branch from the same library of fields and spatial derivatives. All three samplers use {report['budget']} evaluations each, including their search start. The analytic data-generating equation is fitted separately as a reference.

Each benchmark uses one training field, one validation field, and three held-out fields with separately drawn modes, amplitudes and phases. Every equation is fitted by the same weak least-squares procedure. Validation uses the Lab 3 score. For each held-out field, the forecast sees the first two snapshots and propagates for 46 frames with shared periodic finite differences and a Fourier time propagator. Lab 1’s original solver assumes fixed boundaries and zero initial velocity. The shared fit and forecast protocol isolates the effect of equation structure on these periodic fields. The search seeds measure sampler variability on the same data.'''))
    for index, benchmark in enumerate(report['benchmarks'], 1):
        base = benchmark['baseline_tests']
        oracle = benchmark['oracle_tests']
        base_rollout = np.mean([t['rollout_nmse'] for t in base])
        rows = ['| Model | Exact recovery | Mean weak NMSE | Mean forecast NMSE | Beats Lab 1 on every test field |',
                '| --- | ---: | ---: | ---: | ---: |']
        references = [('Lab 1 equation, refitted', base)]
        if 'search_start_tests' in benchmark:
            references.append(('Undamped mixed-wave equation, refitted', benchmark['search_start_tests']))
        references.append(('Data-generating equation, refitted', oracle))
        for label, tests in references:
            rows.append(f"| {label} | — | {np.mean([t['weak_nmse'] for t in tests]):.6g} | "
                        f"{np.mean([t['rollout_nmse'] for t in tests]):.6g} | — |")
        representative = representative_run(benchmark)
        for method in METHODS:
            runs = [r for r in benchmark['runs'] if r['method'] == method]
            exact = sum(r['selected']['exact_recovery'] for r in runs)
            forecasts = np.array([[t['rollout_nmse'] for t in r['held_out']] for r in runs])
            weak = np.array([[t['weak_nmse'] for t in r['held_out']] for r in runs])
            count = int(np.sum(np.all(forecasts < [t['rollout_nmse'] for t in base], axis=1)))
            rows.append(f'| {method} | {exact}/{len(runs)} | {np.median(weak.mean(axis=1)):.6g} | '
                        f'{np.median(forecasts.mean(axis=1)):.6g} | {count}/{len(runs)} |')
            if method == METHODS[0]:
                ratio = base_rollout / np.median(forecasts.mean(axis=1))
        selected = representative['selected']
        change = (f'a {ratio:.1f}× lower mean forecast NMSE for the median guided run' if ratio > 1 else
                  f'a forecast NMSE ratio of {ratio:.3g} (Lab 1 divided by guided)')
        display(Markdown('\n\n'.join([
            f"### {benchmark['name']}", '\n'.join(rows),
            'Each sampler row reports the median across search seeds of the mean across the three test fields.',
            f'The benchmark gives {change}. The forecast figures use the guided finalist from seed '
            f'{representative["seed"]}, which has the lowest guided validation score.',
        ])))
        models = [('Lab 1 baseline', benchmark['baseline']), ('Selected guided equation', selected)]
        equations = [*models, ('Data-generating equation, refitted', benchmark['oracle'])]
        if 'search_start' in benchmark:
            equations.insert(1, ('Undamped mixed-wave equation, refitted', benchmark['search_start']))
        equations.append(('Analytic data-generating equation', dict(graph=benchmark['truth'], theta=benchmark['true_coefficients'])))
        display(Markdown(symbolic_equations(equations)))
        fig = plot_equation_records(models)
        display(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 3.1), layout='constrained')
        for label, tests, color in (('Lab 1 baseline', base, COLORS[2]),
                                    ('Selected guided equation', representative['held_out'], COLORS[0]),
                                    ('Data-generating equation, refitted', oracle, '#596579')):
            curves = np.array([test['rollout_curve'] for test in tests])
            frames = np.arange(1, curves.shape[1] + 1)
            ax.plot(frames, curves.mean(axis=0), ':' if tests is oracle else '-', color=color, label=label)
            ax.fill_between(frames, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.08)
        ax.set(xlabel='Forecast steps after initialization', ylabel='Field NMSE', yscale='log')
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        display(Markdown('Forecast curves average the three test fields. Shading spans their minimum and maximum errors.'))
        display(fig)
        plt.close(fig)

        seed, coords, fields = heldout_fields(report, benchmark, representative)
        display(Markdown(f'#### Held-out fields and supplementary views\n\n'
                         f'The figures use the first prespecified test field and guided run {representative["seed"]}, '
                         'chosen by validation score. The animations and snapshots share a displacement scale within '
                         'each figure, with values outside the reference range shown in saturated colors. '
                         'The full field, residual, and wavefront views are in Section 8. '
                         'The two residual maps in each plane use the same scale.'))
        animation = output / 'figures' / f'forecast_{index:02d}.gif'
        display_forecast_views(coords, fields, animation, appendix=appendix,
                               title=f'B — {benchmark["name"]} · test field {seed}')
        display(Markdown('Each dot below is one search run evaluated on one '
                         'held-out field. Small horizontal offsets reveal overlapping runs.'))
        fig = sampler_forecast_errors(benchmark, report['field_seeds'][2:])
        display(fig)
        plt.close(fig)
    display(Markdown('The generated equations improve the Lab 1 baseline on the synthetic benchmarks. '
                     'The random comparator is at least as successful at exact recovery here. '
                     'The unrestricted graph search in Section 3 misses the exact mixed-wave equation.'))


def add_visual_node(dag, rule):
    label = {'lap': 'Δ', 'add': '+', 'mul': '×', 'u_t': 'uₜ', 'u_tt': 'uₜₜ'}.get(rule, rule)
    if rule.startswith('theta_'):
        label = 'θ' + rule[-1].translate(str.maketrans('1234', '₁₂₃₄'))
        color = NodeColor.PARAMETER
    elif rule in ('u', 'u_t', 'q', 'f', 'u_tt'):
        color = NodeColor.OBSERVED
    else:
        color = None if rule == '?' else NodeColor.LATENT
    return dag._add_node(rule, color, label)


def plot_equation_records(records, columns=2):
    columns = min(columns, len(records))
    rows = (len(records) + columns - 1) // columns
    fig, panels = plt.subplots(rows, columns, figsize=(5.6 * columns, 3.3 * rows), layout='constrained')
    axes = np.atleast_1d(panels).ravel()
    for ax, (title, record) in zip(axes, records):
        dag = WaveEquationDAG()
        repeated = {}
        for op, operands in record['dag']['nodes']:
            node = add_visual_node(dag, op)
            for operand, count in Counter(operands).items():
                dag._add_edge(operand, node)
                if count > 1:
                    repeated[operand, node] = f'×{count}'
        result = add_visual_node(dag, record['dag']['target'])
        dag._add_edge(record['dag']['rhs'], result)
        # Put inputs at the left and the equation output at the right.
        for layer, nodes in enumerate(reversed(list(nx.topological_generations(dag.graph.reverse(copy=False))))):
            for node in nodes:
                dag.graph.nodes[node]['layer'] = layer
        pos = nx.multipartite_layout(dag.graph, subset_key='layer')
        draw_dag(dag, ax, pos, node_size=760, font_size=10)
        if repeated:
            nx.draw_networkx_edge_labels(dag.graph, pos, edge_labels=repeated, font_size=8, ax=ax)
        ax.set_title(title)
    for ax in axes[len(records):]:
        ax.set_axis_off()
    return fig


def plot_equation_graphs(dataset):
    records = [('Selected equation', dataset['best'])]
    known = dataset.get('oracle')
    if known and known['graph'] != dataset['best']['graph']:
        records.append(('Known equation', known))
    return plot_equation_records(records)


def proposal_variety(output):
    """Count canonical models once per dataset and method across search seeds."""
    proposals = Counter()
    distinct = defaultdict(dict)
    with gzip.open(output / 'proposals.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if record['initial']:
                continue
            key = record['dataset'], record['method']
            proposals[key] += 1
            if not record['duplicate']:
                distinct[key].setdefault(json.dumps(record['graph']), record)

    lines = ['### Generated model variety',
             'Graph counts deduplicate newly evaluated canonical equation structures across search seeds, '
             'separately for each sampler. Each run’s starting pool is excluded. Proposal totals include repeated draws. '
             'All generated structures count, regardless of their fitted coefficients or scores.',
             '| Dataset | Sampler | Proposals | Distinct graphs | 1 branch | 2 branches | 3 branches | 4 branches |\n'
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    table = []
    for key, total in proposals.items():
        records = distinct[key].values()
        sizes = Counter(len(record['graph']) for record in records)
        values = [*key, total, len(records), *(sizes[n] for n in range(1, 5))]
        table.append('| ' + ' | '.join(map(str, values)) + ' |')
    if not table:
        return lines[0] + '\n\nThe run ended at the shared starting pool.', []
    lines[-1] += '\n' + '\n'.join(table)

    mixed = distinct['Mixed derivative', METHODS[0]].values()
    candidates = sorted(mixed, key=lambda r: len(r['dag']['nodes']))
    profiles = (
        ('Pointwise product', lambda r: any(term[0] == 'mul' for term in r['graph'])),
        ('Mixed spatial derivative', lambda r: ['D_x', ['D_y', ['u']]] in r['graph']),
        ('Nonlinear transformation', lambda r: any(term[0] in ('square', 'cube', 'sin', 'tanh') for term in r['graph'])),
        ('Vector calculus', lambda r: any(op in ('grad', 'div', 'dot', 'vec', 'scale') for op, _ in r['dag']['nodes'])),
        ('Three branches', lambda r: len(r['graph']) == 3),
        ('Four branches', lambda r: len(r['graph']) == 4),
    )
    examples = []
    for title, matches in profiles:
        record = next((r for r in candidates if matches(r) and all(r is not shown for _, shown in examples)), None)
        if record is not None:
            examples.append((title, record))
    if examples:
        lines.append('These examples come from guided diffusion on the mixed-derivative dataset. '
                     'For each listed structure, the gallery uses the smallest available DAG, with ties by log order. '
                     'An edge marked ×2 supplies the same input twice.')
    return '\n\n'.join(lines), examples


def plot_trajectory(tokens, leaves=('u', 'u_t'), *, counts=None):
    # Use the complete derivation to keep node positions fixed as the mask shrinks.
    complete = Derivation(leaves)
    for token in tokens:
        complete.add(token)
    full = nx.DiGraph()
    full.add_nodes_from(range(len(tokens)))
    for parent, children in enumerate(complete.children):
        full.add_edges_from((child, parent) for child in children)
    positions = {}
    leaf_x = 0

    def place(node, depth):
        nonlocal leaf_x
        children = complete.children[node]
        if children:
            x = np.mean([place(child, depth + 1) for child in children])
        else:
            x = leaf_x
            leaf_x += 1
        positions[node] = x, -depth
        return x

    place(0, 0)
    positions = {node: (x / max(leaf_x - 1, 1), y) for node, (x, y) in positions.items()}
    width = 11.4 if counts is None else 3.2 * len(counts)
    if counts is None:
        counts = (0, min(3, len(tokens)-1), len(tokens))
    if not counts or any(count < 0 or count > len(tokens) for count in counts):
        raise ValueError('Reveal counts must lie between zero and the full derivation length.')
    fig, panels = plt.subplots(1, len(counts), figsize=(width, 3.6), layout='constrained')
    for ax, count in zip(np.atleast_1d(panels), counts, strict=True):
        dag = WaveEquationDAG()
        visible = list(range(count))
        if count == 0:
            visible = [0]
        else:
            visible.extend(node for node in range(count, len(tokens)) if next(full.successors(node)) < count)
        indices = {}
        for node in visible:
            rule = tokens[node] if node < count else '?'
            indices[node] = add_visual_node(dag, 'Σ' if node == 0 and count else rule)
        for child, parent in full.edges:
            if child in indices and parent in indices:
                dag._add_edge(indices[child], indices[parent])
        draw_dag(dag, ax, {indices[node]: positions[node] for node in visible}, node_size=640, font_size=9)
        ax.set_xlim(min(x for x, _ in positions.values()) - 0.24, max(x for x, _ in positions.values()) + 0.24)
        ax.set_ylim(min(y for _, y in positions.values()) - 0.24, max(y for _, y in positions.values()) + 0.24)
        ax.set_title(f'{count} of {len(tokens)} rules revealed', fontsize=11)
    return fig


def example_trajectory(output):
    fallback = None
    with gzip.open(output / 'proposals.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if (record['dataset'] != 'Mixed derivative' or record['method'] != METHODS[0]
                    or record['initial'] or record['duplicate'] or len(record['reveal_order']) > 11):
                continue
            example = (record['reveal_order'], record['unique_evaluations'], record['seed'],
                       'Dₓ(Dᵧ(u))' in record['terms'])
            if example[-1]:
                return example
            if fallback is None:
                fallback = example
    return fallback


def scores_are_flat(dataset):
    for run in dataset['runs']:
        start = run['curve'][INITIAL - 1]
        if start is None or any(value != start for value in run['curve'][INITIAL:]):
            return False
    return True


def audit_text(dataset):
    audit = dataset['audit']
    lines = ['#### Shuffle stress test',
             'Each draw refits the displayed graph on shuffled **training** data, then evaluates it on the same '
             f"untouched validation data. Curves show medians with interquartile bands over {audit['shuffle_draws']} shuffles per "
             'nontrivial window. NMSE = 1 corresponds to predicting zero.']
    levels = audit['shuffle']['Selected graph']
    clean = levels[0]['draws'][0]['clean_validation_nmse']
    strong = next(level['draws'] for level in levels if level['window'] == 12)
    retained = np.sum([draw['retained'] for draw in strong], axis=0)
    support = ', '.join(f'{term} {n}/{len(strong)}' for term, n in zip(audit['terms']['Selected graph'], retained))
    error = np.median([draw['clean_validation_nmse'] for draw in strong])
    lines.append(f'The selected graph’s median clean-validation NMSE changes from {clean:.4g} to {error:.4g} at '
                 f'12 samples. The branch support counts at that window are {support}. Passing the threshold can '
                 'coexist with large coefficient drift and poor predictive accuracy.')
    if audit['time_reversal']:
        lines.extend(['#### Time reversal on the held-out field',
                      'The synthetic equations contain only even time derivatives. Reversing the held-out snapshots '
                      'preserves the equation and flips the sign of features linear in $u_t$. The coefficients are frozen.',
                      '| Equation | Original test NMSE | Reversed test NMSE |\n| --- | ---: | ---: |\n' +
                      '\n'.join(f"| {label} | {v['forward_nmse']:.6g} | {v['reversed_nmse']:.6g} |"
                                for label, v in audit['time_reversal'].items())])
    return '\n\n'.join(lines)


def create_figures(report, output):
    output = Path(output)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    _, examples = proposal_variety(output)
    if examples:
        fig = plot_equation_records(examples)
        fig.savefig(figures / 'model_variety.png', dpi=150)
        plt.close(fig)
    for i, data in enumerate(report['datasets'], 1):
        figures_to_draw = [('equation', plot_equation_graphs), ('shuffle', plot_interventions),
                           ('prediction', plot_predictions)]
        if not scores_are_flat(data):
            figures_to_draw.insert(0, ('scores', plot_scores))
        for name, plot in figures_to_draw:
            fig = plot(data)
            fig.savefig(figures / f'{i:02d}_{name}.png', dpi=150)
            plt.close(fig)
    example = example_trajectory(output)
    if example:
        fig = plot_trajectory(example[0])
        fig.savefig(figures / 'reverse_trajectory.png', dpi=150)
        plt.close(fig)


def report_overview(report):
    from IPython.display import Markdown, display

    lab1 = report['lab1_check']
    display(Markdown(rf'''# Lab 3 — Generating wave-equation graphs

## 1. Question and experimental design

Can a graph generator improve the wave equation from Lab 1, and does feedback from interventions help it search?
We start from $u_{{tt}}=c^2\Delta u+f$ and use source-free synthetic fields ($f=0$). Each generated branch receives
one fitted coefficient. Removing that coefficient removes the corresponding contribution to the equation.

The report distinguishes three experiments; their finalists and errors are not interchangeable.

| Protocol | Search and fitting | Selection rule | What it tests |
| --- | --- | --- | --- |
| A. Unrestricted grammar | {len(report['seeds'])} seeds × {report['budget']} unique graphs per method and dataset; weak fit | Common score $J$ | Discovery of new structure |
| B. One-branch extensions | 5 seeds × 18 graphs per method; weak fit | Common score $J$ | Improvement of the Lab 1 baseline in a restricted library |
| C. Forecast tuning | Training rollout fit, branch recombination and solver choice | Validation field NMSE | Prediction after additional optimization |

**Reading order:** [method](#method) → [A: unrestricted search](#unrestricted) → [B: extensions](#extensions)
→ [C: forecast tuning and divergence](#forecast-tuning) → [sampler comparison](#sampler-comparison)
→ [conclusions](#conclusions). Detailed field and wavefront views are in the [appendix](#appendix);
commands and references are in [Section 9](#reproduction).

**Scope:** {report['mesh_access']} The real-data requirement remains open for the supplied results.

<a id="method"></a>

## 2. Method

### 2.1. Typed equation graphs

$$u_{{tt}}=\sum_{{j=1}}^m\theta_j\phi_j(u,u_t),\qquad 1\le m\le4.$$

Branches combine scalar algebra, spatial derivatives, and typed vector operations. The synthetic inputs are $u$
and $u_t$; prescribed fields $q$ and $f$ are available only when a dataset supplies them. Each branch has at most
eight operators, and each path has spatial derivative order at most two. Sine and tanh operate in the dataset's
dimensionless model units. Common subexpressions are shared. Canonicalization merges algebraic duplicates and
commuting spatial derivatives while preserving distinctions between shared and independent coefficients.

### 2.2. Ordered absorbing masks and the reverse sampler

A graph is encoded as breadth-first grammar rules $\mathbf{{z}}=(z_1,\ldots,z_L)$.
Forward corruption replaces a growing suffix with an absorbing mask:

$$q_t(\mathbf{{z}})= (z_1,\ldots,z_{{L-t}},\underbrace{{\mathrm{{MASK}},\ldots,\mathrm{{MASK}}}}_{{t}}).$$

The reverse process reveals one legal rule at a time, in parent-before-child order. With $A_i$ the legal set,

$$p(z_i\mid z_{{<i}})=0.8\,\operatorname{{softmax}}_{{A_i}}
\!\left(\ell_\lambda(z_{{<i}})/1.15\right)+0.2\,p_{{\mathrm{{prior}}}}(z_i\mid A_i).$$

This is an **ordered absorbing-mask process with autoregressive reverse transitions**. It adapts the masking
idea to a grammar; it does not reproduce a general simultaneous graph-denoising diffusion model.
The random comparator uses the same grammar prior. Unguided diffusion removes only the intervention penalty
from the training weights; all methods use the same final selection score.

### 2.3. Fit, intervene and feed back

Compact weak-form weights move outer derivatives onto test functions. Inner derivatives in nonlinear expressions
use centered differences. Joint least squares estimates $\boldsymbol{{\theta}}$; a column-scaled rank check rejects
unresolved fits. Marginal uncertainty uses a block-clustered sandwich with an independent-residual variance floor.
The Lab 2 support rule requires $|\widehat\theta_j|/s_j\ge2$, or the appropriate one-sided value when the sign has
independent physical justification. Rank-deficient graphs and graphs with unsupported clean coefficients are ineligible.

A shuffle permutes whole snapshots inside a window. Mild ({report['windows'][0]} samples) and strong
({report['windows'][1]} samples) interventions each use three shared draws. A retained branch must stay supported
with the same sign on clean and shuffled data. Let $R(G)$ be the mean mild-shuffle retention. Selection minimizes

$$J(G)=\mathrm{{NMSE}}_{{\mathrm{{val}}}}(G)+0.0015\,C(G)+0.035\,[1-R(G)],$$

where $C(G)$ counts shared internal operators, fitted coefficients, and assembly multiplications and sums.
Weak NMSE is $\|\mathbf{{y}}-\mathbf{{X}}\widehat{{\boldsymbol{{\theta}}}}\|^2/\|\mathbf{{y}}\|^2$.
Field NMSE instead compares freely rolled-out displacement against the reference field; plots identify which is used.

After each batch, evaluated graphs receive weights

$$w(G)=0.075+\exp\!\left[-\frac{{s(G)-\min_Hs(H)}}{{0.12}}\right],\qquad
s(G)=J(G)+0.05\,[1-K(G)],$$

with $K(G)$ the supported fraction on clean data. Unguided weights omit $0.035[1-R(G)]$.
The denoiser minimizes weighted next-rule cross-entropy; each graph's weight is divided among its prefix examples
so long derivations do not receive extra weight. It uses 12-dimensional rule embeddings, an 80-unit GELU layer,
AdamW with learning rate 0.003, batches of 96, and 100 update steps initially / 55 subsequently.

**Algorithm and validity.** Start from 12 unique graphs shared across methods, including the Laplacian. On mixed
waves, the initial pool excludes the mixed-derivative branch. Repeat until the evaluation budget is exhausted:

1. Train the reverse sampler on the weighted evaluated pool (skip for random search).
2. Reveal legal productions until no slots remain; canonicalize and compile the graph.
3. Fit every new graph on training data, refit under interventions, and compute its validation score.
4. Save every proposal and trajectory, count duplicates separately, and update after 20 new graphs.
5. Select the lowest-score eligible graph; evaluate its frozen fit on the reserved test split.

Types and arities hold by induction because only legal productions fill each slot. Every scalar slot can terminate
at an input; the grammar reserves capacity to finish vector slots. Eight operators with at most three children
create at most 25 slots per branch, so the four-branch derivation terminates within 101 rules. Compilation inserts
children before consumers; every edge therefore respects a topological order. These are syntactic validity
guarantees, not proofs of physical correctness, causality, or global optimization.

### 2.4. Data and controls

Protocol A uses nine analytic modes on a $24^3$ grid, 240 frames with $\Delta t=0.021$, and disjoint time splits
of 120 / 60 / 60 frames. Different search seeds reuse these fields. Protocol B instead uses separate training,
validation and three test realizations; its section specifies the restricted library and forecast setup.
Every forward forecast starts from two observed frames and then evolves without future observations.

The original single-mode Lab 1 check recovers speed {lab1['speed']:.5f} (truth 1), but its three directional
features have rank {lab1['directional_rank']}/3. Multimode data are needed to distinguish their coefficients.
The isotropic benchmark includes its true Laplacian in the initial pool and is therefore a fitting sanity check.
'''))


def show_lab():
    from IPython.display import Markdown, display
    from .forecast_plots import selected_forecast_errors, unrestricted_fields

    output = Path(os.getenv('WAVE_LAB3_RESULTS', Path(__file__).parent / 'results'))
    report = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    appendix = []
    report_overview(report)
    trajectory = example_trajectory(output)
    if trajectory:
        tokens, evaluation, seed, has_mixed = trajectory
        proposal = 'A guided proposal containing a mixed derivative' if has_mixed else 'A guided proposal'
        display(Markdown(f'### 2.5. Example reverse trajectory\n\n{proposal} from seed {seed} at unique evaluation {evaluation}. '
                         'Gray nodes are masked. Blue nodes are field variables, green nodes are operators, '
                         'and orange nodes in the compiled equations are fitted coefficients. Arrows point toward the output.'))
        fig = plot_trajectory(tokens)
        display(fig)
        plt.close(fig)
    display(Markdown('<a id="unrestricted"></a>\n\n## 3. A — Unrestricted graph search\n\n'
                     'The following finalists minimize the common validation score within the bounded grammar. '
                     'Weak NMSE measures a local equation residual using observed fields; it is not forecast error.'))
    for index, dataset in enumerate(report['datasets'], 1):
        baseline = dataset['fixed_mask']['finalist']
        winner = dataset['best']
        directional = dataset['directional_diagnostics']
        condition = f"{directional['condition']:.3g}" if directional['condition'] is not None else '∞'
        rows = ['| Method | Truth proposed | Exact finalist | Median score | Median test NMSE | Median duplicates |',
                '| --- | ---: | ---: | ---: | ---: | ---: |']
        metrics = list(method_rows(dataset))
        for method, count, n, score, test, duplicates, proposed in metrics:
            recovered = f'{proposed}/{n} | {count}/{n}' if dataset['truth'] else '— | —'
            rows.append(f'| {method} | {recovered} | {score:.6g} | {test:.6g} | {duplicates:g} |')
        if dataset['name'] == 'Isotropic waves' and all(row[1] == row[2] for row in metrics):
            interpretation = (f'Every method selected the seeded Laplacian. No search improved the '
                              f'initial validation score of {metrics[0][3]:.6g}. This is a sanity check '
                              'of fitting and selection. It provides no evidence about discovery.')
        elif dataset['truth']:
            guided, unguided, random = metrics
            interpretation = (f'Guidance recovered the exact structure in {guided[1]} of {guided[2]} seeds, '
                              f'compared with {unguided[1]} for unguided diffusion and {random[1]} for random grammar search. '
                              'The runs reuse the same field, so this comparison measures search variability on one benchmark.')
            mixed_counts = [sum(run.get('first_mixed_branch') is not None for run in dataset['runs']
                                if run['method'] == method) for method in METHODS]
            if all('first_mixed_branch' in run for run in dataset['runs']):
                interpretation += (f' The mixed derivative appeared as a standalone branch in {mixed_counts[0]} guided, '
                                   f'{mixed_counts[1]} unguided and {mixed_counts[2]} random runs.')
        else:
            interpretation = 'The measured field has no labeled graph. The table compares validation scores and held-out local prediction.'
        oracle = dataset.get('oracle')
        if dataset['name'] == 'Mixed derivative' and oracle:
            rule = 'passes' if oracle['eligible'] else 'fails'
            interpretation += (f" A separate post-search fit of the known graph {rule} the support rule "
                               f"and gives validation score {oracle['score']:.6g} and test NMSE {oracle['test_nmse']:.6g}.")
        coeffs = ['| Branch | Coefficient | Marginal SD | z | Supported |', '| --- | ---: | ---: | ---: | --- |']
        for vals in zip(winner['terms'], winner['theta'], winner['sd'], winner['z'], winner['kept'], strict=True):
            term, *values, kept = vals
            nums = ' | '.join('Unresolved' if value is None else f'{value:.6g}' for value in values)
            coeffs.append(f'| {term} | {nums} | {"Yes" if kept else "No"} |')
        ablations = '\n'.join(f"- {a['term']} → 0, refitted coefficients {a['refitted_theta']}, "
                              f"validation NMSE change {a['delta_validation']:+.5g}." for a in winner['ablations'])
        outcome = 'Best eligible graph' if winner['eligible'] else 'Best candidate (all proposals failed the clean-data rule)'
        lab1_terms = ' + '.join(baseline['terms'])
        truth_text = ' + '.join(dataset['truth'] or [])
        text = [f"### A{index}. {dataset['name']}", dataset['description'],
                f"The known branch structure is {truth_text}." if truth_text else "The measured reference is the pressure field.",
                f"Weak rows in train / validation / test are {dataset['sample_counts']}. Directional training features have rank {directional['rank']}/3 and scaled condition {condition}.",
                '\n'.join(rows), interpretation,
                f"The fixed-term mask evaluated {dataset['fixed_mask']['unique']} subsets and chose {lab1_terms} with test NMSE {baseline['test_nmse']:.6g}.",
                f"{outcome} has validation NMSE {winner['validation']:.6g} and held-out NMSE {winner['test_nmse']:.6g}. "
                f"Its fitted design has rank {winner['rank']} of {len(winner['terms'])}.",
                '\n'.join(coeffs), '#### Branch removal and refitting', ablations or 'No branch qualified for ablation.']
        display(Markdown('\n\n'.join(text)))
        models = [('A — Unrestricted finalist', winner), ('Fixed-term baseline, refitted', baseline)]
        if oracle:
            models.append(('Known equation, refitted', oracle))
        display(Markdown(symbolic_equations(models)))
        if not scores_are_flat(dataset):
            display(Markdown('The left plot starts after the shared pool. Lines show median best eligible '
                             'validation scores and bands show the middle half of the paired seeds. The right '
                             'plot shows the held-out error for every finalist. The known equation was fit separately.'))
            fig = plot_scores(dataset)
            display(fig)
            plt.close(fig)
        fig = plot_equation_graphs(dataset)
        display(fig)
        plt.close(fig)
        display(Markdown(audit_text(dataset)))
        for plot in (plot_interventions, plot_predictions):
            fig = plot(dataset)
            display(fig)
            plt.close(fig)
        forecast = unrestricted_fields(dataset)
        if forecast is not None:
            selected_run = next(run for run in dataset['runs'] if run['finalist'] == winner)
            coords, fields = forecast
            fig, (base_nmse, selected_nmse) = selected_forecast_errors(coords, fields)
            display(Markdown('#### Forward forecast of the unrestricted-search finalist\n\n'
                             f'This is the selected graph from {selected_run["method"]}, seed {selected_run["seed"]}. '
                             'The forecast uses the original analytic modes on a full periodic spatial grid over '
                             'the reserved time interval. Both models retain their original fitted coefficients '
                             'and initialize with the same two snapshots. Time is relative to the first snapshot. '
                             'The contours and 3D surfaces show where $u=0$. '
                             f'Mean field NMSE is {base_nmse:.6g} for Lab 1 and {selected_nmse:.6g} for the finalist.'))
            animation = output / 'figures' / f'unrestricted_forecast_{index:02d}.gif'
            display_forecast_views(coords, fields, animation, appendix=appendix,
                                   title=f'A — {dataset["name"]}')
            display(fig)
            plt.close(fig)
        elif dataset['truth']:
            display(Markdown('The periodic Fourier forecast is available when the selected graph uses linear '
                             'constant-coefficient branches. The weak-acceleration plots above use the full selected graph.'))
    extension_results = output / 'extensions.json'
    if extension_results.is_file():
        report_extensions(json.loads(extension_results.read_text(encoding='utf-8')), output, appendix)
    tuning_results = output / 'forecast_tuning.json'
    if tuning_results.is_file():
        report_tuned_forecasts(json.loads(tuning_results.read_text(encoding='utf-8')), output, appendix)
    comparison = output / 'sampler_comparison.json'
    if comparison.is_file():
        mixed = next(d for d in report['datasets'] if d['name'] == 'Mixed derivative')
        report_sampler_comparison(json.loads(comparison.read_text(encoding='utf-8')), mixed)
    display(Markdown('''<a id="conclusions"></a>

## 7. Conclusions and limitations

The evidence answers three different questions. The unrestricted generator explores new graphs but fails to recover
the exact mixed-wave equation in the supplied runs. Restricted one-branch searches improve the refitted Lab 1 equation
on both synthetic benchmarks. Additional rollout fitting and branch recombination recover the mixed-wave dynamics
to numerical accuracy on the tested noise-free fields. That last result includes extra optimization after graph search.

**These experiments do not establish a consistent advantage for diffusion over random search.** Guided search has
lower median weak error on the unrestricted mixed benchmark, while random search also finds the useful extensions.
Under identical forecast postprocessing, both recover the exact structure in four of five runs. Search seeds reuse
the same training fields, so they do not measure variation across independent training datasets.

Temporal shuffling probes sensitivity to time scrambling; it does not identify a physical causal mechanism.
Supported coefficients can survive a shuffle even when predictive error increases. The support threshold is a
post-search screening heuristic, not a calibrated significance test. Dependence among weak rows and only a few
time clusters limit the uncertainty estimates. Discretization can bias fitted coefficients even without noise.

**Requirement still open:** the supplied results contain no measured-data experiment. The MeshRIR loader and
instructions are available, but synthetic performance does not fulfill the handout's real-data requirement.

<a id="appendix"></a>

## 8. Supplementary graph and forecast diagnostics

The views below retain the detailed evidence for every selected model. Each group identifies its protocol.
Animations compare the reference, refitted Lab 1 equation, and selected equation. Snapshots and residual maps use
matched scales within a comparison. Zero-displacement contours and surfaces visualize the superposed field;
they are not a uniquely identified physical arrival front. Time traces shade the two initialization frames.
'''))
    variety, examples = proposal_variety(output)
    display(Markdown(variety))
    if examples:
        fig = plot_equation_records(examples)
        display(fig)
        plt.close(fig)
    for title, coords, fields, animation in appendix:
        display(Markdown(f'### {title}'))
        display_forecast_views(coords, fields, animation)
    display(Markdown('''<a id="reproduction"></a>

## 9. Reproduction and sources

This notebook regenerates tables, graphs and forecasts from the frozen result files; executing it does not retrain
the sampler. All fitting, grammar and forecasting code remains reusable in `lab3/`. See `lab3/README.md` for the
global Pixi environment, complete experiment commands and measured-data setup. From the repository root:

```bash
python -m unittest lab3.test_lab3
python -m lab3.experiment
python -m lab3.extensions
python -m lab3.forecast_tuning
python -m lab3.sampler_comparison
python -m lab3.forecast_plots
```

Saved evidence: `results/summary.json`, `extensions.json`, `forecast_tuning.json`, `sampler_comparison.json`,
and `proposals.jsonl.gz`. The presentation is `presentation/Lab3_presentation.pdf`; its editable LaTeX source,
bundled figures and speaking notes are in `presentation/latex/`.

### References

''' + REFERENCES))
