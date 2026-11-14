"""Conditional small-phase drift estimates for the synthetic mixed-wave model."""
import numpy as np

from .grammar import LAPLACIAN, MIXED
from .wave_data import synthetic_modes


def phase_drift_estimate(model, thresholds=(0.01, 0.05, 0.10)):
    """Estimate RMS phase drift; this is not a simulated or statistical horizon.

    Assumes the exact nine traveling modes, unchanged amplitudes, continuous
    spectral dynamics and no noise. Two-frame initialization transients and
    future numerical accumulation are omitted. Only lap(u) + D_x D_y u is valid.
    """
    def freeze(value):
        return tuple(freeze(v) for v in value) if isinstance(value, (list, tuple)) else value

    graph = [freeze(branch) for branch in model['graph']]
    if len(graph) != 2 or set(graph) != {LAPLACIAN, MIXED}:
        raise ValueError('Phase estimate requires precisely the Laplacian and mixed derivative.')
    coefficients = dict(zip(graph, model['theta'], strict=True))
    k, amplitudes, truth, _ = synthetic_modes(mixed=True)
    squared = coefficients[LAPLACIAN] * (k**2).sum(axis=1) + coefficients[MIXED] * k[:, 0] * k[:, 1]
    if not np.isfinite(squared).all() or np.any(squared <= 0):
        raise ValueError('Phase estimate requires finite, stable oscillatory modes.')
    delta = np.pi * np.sqrt(squared) - truth
    slope = float(np.sqrt(np.average(delta**2, weights=amplitudes**2)))
    thresholds = np.asarray(thresholds, dtype=float)
    if np.any(~np.isfinite(thresholds)) or np.any(thresholds <= 0):
        raise ValueError('RMS thresholds must be positive and finite.')
    times = thresholds / slope if slope else np.full_like(thresholds, np.inf)
    return dict(slope=slope, thresholds=thresholds, times=times, delta_omega=delta)


def plot_phase_drift(estimate, horizon):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 3.6), layout='constrained')
    times, thresholds = estimate['times'], estimate['thresholds']
    if estimate['slope'] == 0:
        ax.text(0.5, 0.5, 'Zero frequency mismatch: no phase drift in this idealized model.',
                ha='center', va='center', transform=ax.transAxes)
    else:
        lead = np.linspace(0, 1.1 * max(times), 300)
        ax.plot(lead, 100 * estimate['slope'] * lead, '--', color='#146a63',
                label='Small-phase approximation (not a rollout)')
        ax.scatter(times, 100 * thresholds, color='#146a63')
        for index, (t, level) in enumerate(zip(times, thresholds, strict=True)):
            last = index == len(times) - 1
            ax.annotate(f'{100 * level:g}% at {t:,.0f}', (t, 100 * level),
                        xytext=(-5 if last else 5, 7), ha='right' if last else 'left',
                        textcoords='offset points', fontsize=9)
        ax.axvline(horizon, color='#9c4a26', label=f'Evaluated horizon: {horizon:.3f}')
        ax.set_ylim(0, 125 * max(thresholds))
        ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.set(xlabel='Forecast lead time (simulation units)', ylabel='Estimated relative RMS error (%)',
           title='Conditional phase drift on the original nine synthetic modes')
    ax.grid(alpha=0.2)
    return fig
