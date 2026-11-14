"""Field, wavefront, and error figures for the held-out forecasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from skimage.measure import marching_cubes

from .extensions import CORRECTIONS, LAB1_GRAPH, benchmark, forecast_frames
from .experiment import METHODS
from .fitting import normalized_error
from .wave_data import synthetic


FIELD_LABELS = ('Reference field', 'Lab 1 forecast', 'Selected forecast')
FIELD_COLORS = ('#202b38', '#9c4a26', '#126e64')
SAMPLER_COLORS = ('#146a63', '#30569f', '#bd542e')


def representative_run(result):
    guided = [run for run in result['runs'] if run['method'] == METHODS[0]]
    return min(guided, key=lambda run: run['selected']['score'])


def heldout_fields(report, result, run):
    """Regenerate the first prespecified test field and both frozen forecasts."""
    seeds = report['field_seeds']
    damping = -result['true_coefficients'][result['truth'].index(['u_t'])] if ['u_t'] in result['truth'] else 0.
    data, _, _ = benchmark(damping, seeds[:3])
    time, space = data.splits[2]
    observed = data.u[time, space]
    fields = [observed]
    for model, stored in ((result['baseline'], result['baseline_tests'][0]),
                          (run['selected'], run['held_out'][0])):
        predictions = np.stack(list(forecast_frames(observed[:2], data.spacing, model['graph'],
                                                    model['theta'], len(observed) - 2)))
        errors = [normalized_error(actual, predicted) for actual, predicted in zip(observed[2:], predictions, strict=True)]
        np.testing.assert_allclose(errors, stored['rollout_curve'], rtol=1e-8, atol=1e-12)
        fields.append(np.concatenate((observed[:2], predictions)))
    coordinates = tuple(np.arange(n) * h for n, h in zip(observed.shape, data.spacing, strict=True))
    return seeds[2], coordinates, fields


def unrestricted_fields(result):
    """Use the original analytic waves on a full periodic grid at held-out times."""
    if result['name'] not in ('Isotropic waves', 'Mixed derivative'):
        return None

    def expression_tuple(expr):
        return (expr[0], *(expression_tuple(child) for child in expr[1:]))

    baseline = result.get('lab1', result['fixed_mask']['finalist'])
    if tuple(map(expression_tuple, baseline['graph'])) != LAB1_GRAPH:
        return None
    if any(expression_tuple(branch) not in CORRECTIONS for branch in result['best']['graph']):
        return None
    data = synthetic(mixed=result['name'] == 'Mixed derivative', periodic=True)

    return forecast_comparison(data, (baseline, result['best']))


def forecast_comparison(data, models):
    time, space = data.splits[2]
    observed = data.u[time, space]
    fields = [observed]
    for model in models:
        future = forecast_frames(observed[:2], data.spacing, model['graph'], model['theta'], len(observed) - 2,
                                 spectral=model.get('spectral', False))
        fields.append(np.concatenate((observed[:2], np.stack(list(future)))))
    coordinates = tuple(np.arange(n) * h for n, h in zip(observed.shape, data.spacing, strict=True))
    return coordinates, fields


def field_snapshots(coords, fields):
    times, x, y, z = coords
    middle = len(z) // 2
    indices = (2, len(times) // 2, len(times) - 1)
    fig, axes = plt.subplots(3, 3, figsize=(8.8, 7.8), layout='constrained', sharex=True, sharey=True)
    vmax = max(np.abs(fields[0][list(indices), :, :, middle]).max(), np.finfo(float).eps)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for row, (label, field) in enumerate(zip(FIELD_LABELS, fields, strict=True)):
        for col, index in enumerate(indices):
            ax = axes[row, col]
            image = ax.pcolormesh(x, y, field[index, :, :, middle].T, shading='nearest', cmap='RdBu_r', norm=norm)
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(f't = {times[index]:.3f}')
            if col == 0:
                ax.set_ylabel(f'{label}\ny')
            if row == 2:
                ax.set_xlabel('x')
    fig.colorbar(image, ax=axes, label='Displacement u', shrink=0.75, extend='both')
    fig.suptitle(f'Wave displacement in the central xy plane (z = {z[middle]:g})', fontsize=14)
    return fig


def wavefront_maps(coords, fields):
    times, *xyz = coords
    cross_sections = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.1), layout='constrained')
    axes_names = 'xyz'
    index = len(times) - 1
    for col, (a, b, fixed) in enumerate(cross_sections):
        mid = len(xyz[fixed]) // 2
        slices = [np.take(field[index], mid, axis=fixed).T for field in fields]
        residuals = [estimate - slices[0] for estimate in slices[1:]]
        span = max(*(np.abs(residual).max() for residual in residuals), np.finfo(float).eps)
        norm = TwoSlopeNorm(vmin=-span, vcenter=0, vmax=span)
        for row, residual in enumerate(residuals):
            ax = axes[row, col]
            image = ax.pcolormesh(xyz[a], xyz[b], residual, cmap='RdBu_r', shading='nearest', norm=norm, alpha=0.8)
            for wave, color, style in ((slices[0], '#181d26', '-'), (slices[row + 1], '#137b70', '--')):
                if wave.min() < 0 < wave.max():
                    ax.contour(xyz[a], xyz[b], wave, levels=[0], colors=[color], linestyles=[style], linewidths=1.2)
            ax.set(aspect='equal', xlabel=axes_names[a], ylabel=axes_names[b])
            ax.set_title(f'{FIELD_LABELS[row + 1]} · {axes_names[a]}{axes_names[b]}', fontsize=10)
        fig.colorbar(image, ax=axes[:, col], label='Forecast − reference', shrink=0.82)
    handles = [Line2D([0], [0], color='#181d26', label='Reference u = 0'),
               Line2D([0], [0], color='#137b70', ls='--', label='Forecast u = 0')]
    fig.legend(handles=handles, loc='outside lower center', frameon=False, ncol=2)
    fig.suptitle(f'Wavefront contours and signed residuals at t = {times[index]:.3f}', fontsize=14)
    return fig


def forecast_diagnostics(coords, fields):
    times, *xyz = coords
    midpoint = tuple(len(axis) // 2 for axis in xyz)
    fig, (point, amplitude, scatter) = plt.subplots(1, 3, figsize=(11, 3.25), layout='constrained')
    for label, color, field in zip(FIELD_LABELS, FIELD_COLORS, fields, strict=True):
        point.plot(times, field[(slice(None), *midpoint)], color=color, label=label)
        amplitude.plot(times, np.sqrt(np.mean(field ** 2, axis=(1, 2, 3))), color=color)
    actual = fields[0][-1, ::2, ::2, ::2].ravel()
    span = max(np.abs(field[-1]).max() for field in fields)
    scatter.plot([-span, span], [-span, span], color='#536077', ls=':', lw=1)
    for color, field in zip(FIELD_COLORS[1:], fields[1:], strict=True):
        scatter.scatter(actual, field[-1, ::2, ::2, ::2].ravel(), s=8, alpha=0.25, color=color, linewidths=0)
    scatter.set(xlabel='Reference u', ylabel='Forecast u', title='Final frame', aspect='equal')
    point.set(xlabel='Time', ylabel='Displacement u', title='Center-point trace')
    amplitude.set(xlabel='Time', ylabel='Spatial RMS displacement', title='Amplitude over time')
    for ax in (point, amplitude):
        ax.axvspan(times[0], times[1], color='#98a5b6', alpha=0.22)
        ax.grid(alpha=0.18)
    scatter.grid(alpha=0.18)
    handles, labels = point.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc='outside lower center', fontsize=9)
    return fig


def selected_forecast_errors(coords, fields):
    times = coords[0]
    fig, ax = plt.subplots(figsize=(7.6, 3.1), layout='constrained')
    _, *forecasts = fields
    means = []
    for label, color, estimate in zip(FIELD_LABELS[1:], FIELD_COLORS[1:], forecasts, strict=True):
        errors = [normalized_error(reference, frame) for reference, frame in zip(fields[0][2:], estimate[2:], strict=True)]
        means.append(float(np.mean(errors)))
        ax.plot(times[2:], errors, color=color, label=label)
    ax.set(xlabel='Time from start of held-out interval', ylabel='Field NMSE', yscale='log')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return fig, means


def tuned_residuals(result, coords, reference):
    """Show the spatial RMS residual and the first sampled tolerance crossings."""
    times = coords[0][2:] - coords[0][1]
    reference_rms = np.sqrt(np.mean(reference[2:].astype(float) ** 2, axis=(1, 2, 3)))
    comparisons = (('Lab 1, rollout-tuned', 'Tuned Lab 1', FIELD_COLORS[1]),
                   ('Original selected forecast', 'Previous finalist', '#476591'),
                   ('Validation-selected forecast', 'Fine-tuned finalist', FIELD_COLORS[2]))
    tolerances = (5, 10)
    fig, (relative, absolute) = plt.subplots(1, 2, figsize=(10.5, 3.85), layout='constrained', sharex=True)
    crossings = {}
    for key, label, color in comparisons:
        rms_ratio = np.sqrt(result['test'][key][0]['curve'])
        percent = 100 * rms_ratio
        relative.plot(times, percent, color=color, label=label, linewidth=2)
        absolute.plot(times, rms_ratio * reference_rms, color=color, linewidth=2)
        crossings[label] = []
        for threshold in tolerances:
            above = np.flatnonzero(percent > threshold)
            crossings[label].append(float(times[above[0]]) if above.size else None)
            if above.size:
                idx = above[0]
                relative.scatter(times[idx], percent[idx], color=color, s=24, zorder=3)
    for threshold in tolerances:
        relative.axhline(threshold, color='#6c7786', linestyle=':', linewidth=0.9)
    relative.set(title='Relative residual', ylabel='RMS residual / reference RMS (%)')
    absolute.set(title='Absolute residual', ylabel='Spatial RMS displacement error')
    for ax in (relative, absolute):
        ax.set(xlabel='Lead time after last input frame', yscale='log', xlim=(0, times[-1]))
        ax.grid(alpha=0.2)
    handles, labels = relative.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc='outside lower center')
    fig.suptitle('Mixed waves · held-out forecast residuals', fontsize=14)
    return fig, tolerances, crossings, float(times[-1])


def sampler_forecast_errors(result, field_seeds):
    fig, ax = plt.subplots(figsize=(7.2, 3.25), layout='constrained')
    x = np.arange(len(field_seeds))
    baseline = [test['rollout_nmse'] for test in result['baseline_tests']]
    ax.plot(x, baseline, 'D--', color='#27374b', label='Lab 1', ms=5)
    for i, (method, color) in enumerate(zip(METHODS, SAMPLER_COLORS, strict=True)):
        errors = np.array([[test['rollout_nmse'] for test in run['held_out']]
                           for run in result['runs'] if run['method'] == method])
        offsets = np.linspace(-0.055, 0.055, len(errors))
        ax.scatter((x + (i - 1) * 0.22) + offsets[:, None], errors, color=color, s=27, alpha=0.72, label=method)
    ax.set(xlabel='Held-out field seed', ylabel='Mean forecast NMSE', yscale='log', xticks=x, xticklabels=field_seeds)
    ax.grid(axis='y', alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='outside lower center', ncol=4, fontsize=8)
    return fig


def wavefront_surfaces(coords, fields):
    times, *xyz = coords
    spacing = tuple(axis[1] - axis[0] for axis in xyz)
    index = len(times) - 1
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.85), layout='constrained',
                             subplot_kw={'projection': '3d'})
    for ax, label, color, field in zip(axes, FIELD_LABELS, FIELD_COLORS, fields, strict=True):
        if field[index].min() < 0 < field[index].max():
            vertices, faces, _, _ = marching_cubes(field[index], level=0, spacing=spacing)
            ax.plot_trisurf(*vertices.T, triangles=faces, color=color, linewidth=0, shade=True, alpha=0.82)
        ax.set(xlabel='x', ylabel='y', zlabel='z', title=label, box_aspect=(1, 1, 1))
        ax.set_xlim(xyz[0][0], xyz[0][-1])
        ax.set_ylim(xyz[1][0], xyz[1][-1])
        ax.set_zlim(xyz[2][0], xyz[2][-1])
        ax.view_init(elev=22, azim=-53)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticks([0, 1])
    fig.suptitle(f'3D zero-displacement surfaces at t = {times[index]:.3f}', fontsize=14)
    return fig


def save_wave_animation(coords, fields, path):
    times, x, y, z = coords
    middle = len(z) // 2
    vmax = np.abs(fields[0][:, :, :, middle]).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.85), layout='constrained', sharey=True)
    layers = []
    for ax, label, field in zip(axes, FIELD_LABELS, fields, strict=True):
        layers.append(ax.pcolormesh(x, y, field[2, :, :, middle].T, norm=norm, cmap='RdBu_r', shading='nearest'))
        ax.set(title=label, xlabel='x', aspect='equal')
    axes[0].set_ylabel('y')
    fig.colorbar(layers[0], ax=axes, label='Displacement u', shrink=0.8, extend='both')
    title = fig.suptitle('')

    def draw(index):
        title.set_text(f'Central xy plane · z = {z[middle]:g} · t = {times[index]:.3f}')
        for layer, field in zip(layers, fields, strict=True):
            layer.set_array(field[index, :, :, middle].T)

    animation = FuncAnimation(fig, draw, frames=range(2, len(times)), repeat=True)
    animation.save(path, writer=PillowWriter(fps=9), dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results')
    output = parser.parse_args().results
    report = json.loads((output / 'extensions.json').read_text(encoding='utf-8'))
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    for i, result in enumerate(report['benchmarks'], 1):
        _, coords, fields = heldout_fields(report, result, representative_run(result))
        save_wave_animation(coords, fields, figures / f'forecast_{i:02d}.gif')
    summary_file = output / 'summary.json'
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        for i, result in enumerate(summary['datasets'], 1):
            forecast = unrestricted_fields(result)
            if forecast is not None:
                coords, fields = forecast
                save_wave_animation(coords, fields, figures / f'unrestricted_forecast_{i:02d}.gif')
    tuned_file = output / 'forecast_tuning.json'
    if tuned_file.is_file():
        tuned = json.loads(tuned_file.read_text(encoding='utf-8'))
        mixed = next((r for r in tuned['benchmarks'] if r['name'] == 'Mixed derivative'), None)
        if mixed is not None:
            models = mixed['models']
            coords, fields = forecast_comparison(synthetic(mixed=True, periodic=True),
                                                 (models['Lab 1, rollout-tuned'], models['Validation-selected forecast']))
            save_wave_animation(coords, fields, figures / 'tuned_mixed_forecast.gif')
            fig, _, _, horizon = tuned_residuals(mixed, coords, fields[0])
            fig.savefig(figures / 'tuned_mixed_residuals.png', dpi=150)
            plt.close(fig)
            from .phase_drift import phase_drift_estimate, plot_phase_drift
            model = models['Validation-selected forecast']
            if model['spectral']:
                try:
                    drift = phase_drift_estimate(model)
                except ValueError:
                    pass  # The estimate applies only to the two-branch mixed-wave model.
                else:
                    fig = plot_phase_drift(drift, horizon)
                    fig.savefig(figures / 'tuned_mixed_phase_drift.png', dpi=150)
                    plt.close(fig)


if __name__ == '__main__':
    main()
