"""Refresh presentation figures from the Lab 3 frozen results; run from the repo root."""
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from lab3.forecast_plots import forecast_comparison
from lab3.phase_drift import phase_drift_estimate, plot_phase_drift
from lab3.wave_data import synthetic

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / 'results'
FIGURES = HERE / 'figures'
FIGURES.mkdir(exist_ok=True)
for name in ('02_equation.png', 'reverse_trajectory.png'):
    shutil.copyfile(RESULTS / 'figures' / name, FIGURES / name)
report = json.loads((RESULTS / 'forecast_tuning.json').read_text())
mixed = next(b for b in report['benchmarks'] if b['name'] == 'Mixed derivative')
models = mixed['models']
coords, fields = forecast_comparison(synthetic(mixed=True, periodic=True),
                                     (models['Lab 1, rollout-tuned'], models['Validation-selected forecast']))
times, x, y, z = coords
vmax = np.abs(fields[0][-1, :, :, len(z)//2]).max()
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.55), layout='constrained', sharey=True)
for ax, field, label in zip(axes, fields, ('Reference', 'Lab 1, rollout-tuned', 'Selected spectral model'), strict=True):
    mesh = ax.pcolormesh(x, y, field[-1, :, :, len(z)//2].T, norm=norm, cmap='RdBu_r', shading='nearest')
    ax.set(title=label, xlabel='x', aspect='equal', xticks=(0,1), yticks=(0,1))
axes[0].set_ylabel('y')
fig.colorbar(mesh, ax=axes, label='Displacement u', shrink=0.85, extend='both')
fig.savefig(FIGURES / 'tuned_field.pdf', bbox_inches='tight')
plt.close(fig)

lead = times[2:] - times[1]
fig, ax = plt.subplots(figsize=(5.3, 2.9), layout='constrained')
for key, label, color in [('Lab 1, rollout-tuned','Lab 1, tuned','#9c4a26'),
                          ('Original selected forecast','Original finalist','#476591'),
                          ('Validation-selected forecast','Forecast-tuned','#146a63')]:
    error = 100 * np.sqrt(mixed['test'][key][0]['curve'])
    assert len(error) == len(lead)
    ax.plot(lead,error,label=label,color=color,lw=1.8)
ax.axhline(5,color='#6c7786',ls=':',lw=1)
ax.set(xlabel='Lead time after last input frame',ylabel='Relative RMS error (%)',yscale='log',xlim=(0,lead[-1]))
ax.grid(alpha=0.2)
ax.legend(fontsize=8,frameon=False,loc='center right')
fig.savefig(FIGURES / 'residuals.pdf', bbox_inches='tight')
plt.close(fig)

drift = phase_drift_estimate(models['Validation-selected forecast'])
fig = plot_phase_drift(drift, float(lead[-1]))
fig.savefig(RESULTS / 'figures' / 'tuned_mixed_phase_drift.png', dpi=150)
plt.close(fig)
print('Refreshed figure assets; evaluated horizon:', float(lead[-1]))
