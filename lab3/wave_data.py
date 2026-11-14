"""Wave fields, space-time splits and compact weak-form measurements."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import correlate1d
from scipy.signal import butter, sosfilt

from wave_data_loader import generate_homogeneous_wave

from .grammar import AXES, LAPLACIAN, MIXED, PURE


@dataclass
class WaveData:
    name: str
    u: np.ndarray
    spacing: tuple
    splits: tuple
    truth: tuple | None
    description: str
    inputs: dict | None = None


def synthetic_modes(mixed=False):
    """Return wavevectors, amplitudes, frequencies and phases of the nine-mode field."""
    # Oblique, signed wavevectors and different magnitudes break the single-mode aliases.
    k = np.array([[1, 0, 1], [1, 1, 0], [1, -1, 0], [0, 1, 1], [2, 1, -1],
                  [1, 2, 1], [0, 0, 1], [1, -1, -2], [2, -1, 0]], dtype=float)
    theta = 0.85 if mixed else 1.0
    cross = 0.72 if mixed else 0.0
    omega = np.pi * np.sqrt(theta * (k**2).sum(axis=1) + cross * k[:, 0] * k[:, 1])
    j = np.arange(len(k))
    return k, 0.46 + 0.08 * (j % 4), omega, 0.64 * j


def synthetic(mixed=False, *, periodic=False):
    k, amplitudes, omega, phases = synthetic_modes(mixed)
    theta = 0.85 if mixed else 1.0
    cross = 0.72 if mixed else 0.0
    # The analytic modes have period 2. Use that full domain for FFT rollouts.
    spatial = np.linspace(0, 2, 24, endpoint=False) if periodic else np.linspace(0, 1.75, 24)
    time = np.arange(240) * 0.021
    t, x, y, z = np.ix_(time, spatial, spatial, spatial)
    u = np.zeros((len(time), len(spatial), len(spatial), len(spatial)), dtype=np.float32)
    for (kx, ky, kz), amplitude, w, phase in zip(k, amplitudes, omega, phases, strict=True):
        u += amplitude * np.cos(np.pi * (kx*x + ky*y + kz*z) - w*t + phase)
    full = slice(None)
    splits = ((slice(0, 120), full), (slice(120, 180), full), (slice(180, 240), full))
    return WaveData('Mixed derivative' if mixed else 'Isotropic waves', u,
                    (time[1], *(float(spatial[1]),) * 3), splits,
                    (MIXED, LAPLACIAN) if mixed else (LAPLACIAN,),
                    (f'Nine analytic modes with c²={theta:g} and mixed coefficient {cross:g}. '
                     'Train, validation and test use disjoint time ranges.'))


def sanity_data():
    data = generate_homogeneous_wave(nx=19, ny=19, nz=19, nt=120, c=1.0, T=2.5)
    return WeakRows([data['u'][i:i+24] for i in range(0, 120, 24)],
                    tuple(data[a][1] - data[a][0] for a in ('t', 'x', 'y', 'z')))


def _impulse_offset(files, mic, source, reference_speed, frequency):
    # Align training microphones by expected travel time only to locate the shared delay.
    # Keep the original relative channel timing when constructing the pressure field.
    distances = np.linalg.norm(mic - source, axis=1)
    training = np.flatnonzero(mic[:, 0] <= np.unique(mic[:, 0])[6])[::80]
    travel = np.rint(frequency * distances[training] / reference_speed).astype(int)
    traces = [np.load(files[i], mmap_mode='r')[0] for i in training]
    common_length = min(len(trace) - lag for trace, lag in zip(traces, travel, strict=True))
    aligned = np.stack([trace[lag:lag + common_length]
                        for trace, lag in zip(traces, travel, strict=True)])
    power = np.mean(aligned**2, axis=0)
    if not np.isfinite(power).all() or power.max() <= 0:
        raise ValueError('Cannot locate the impulse response in the training microphones.')
    return int(np.argmax(power))


def load_meshrir(directory):
    directory = Path(directory)
    if not (directory / 'pos_mic.npy').is_file():
        raise FileNotFoundError(f'MeshRIR pos_mic.npy was not found in {directory}.')
    mic = np.round(np.load(directory / 'pos_mic.npy'), 4)
    source = np.load(directory / 'pos_src.npy').reshape(-1, 3)
    if mic.shape != (3969, 3) or source.shape != (1, 3) or not np.isfinite(source).all():
        raise ValueError('Expected MeshRIR S1-M3969 microphone and source positions.')
    coords = tuple(np.unique(axis) for axis in mic.T)
    if tuple(map(len, coords)) != (21, 21, 9) or not all(np.allclose(np.diff(a), 0.05) for a in coords):
        raise ValueError('Expected MeshRIR’s 21 × 21 × 9 grid with 0.05 m spacing.')
    nearest = np.clip(source[0], [a.min() for a in coords], [a.max() for a in coords])
    if np.linalg.norm(nearest - source[0]) < 0.10:
        raise ValueError('The source is too close to the analysis grid for a source-free fit.')
    index = tuple(np.searchsorted(axis, mic[:, j]) for j, axis in enumerate(coords))
    if len(set(zip(*index))) != len(mic):
        raise ValueError('MeshRIR microphone grid contains repeated positions.')
    files = {int(p.stem.rsplit('_', 1)[-1]): p for p in directory.glob('ir_*.npy')}
    if set(files) != set(range(len(mic))):
        raise ValueError('Expected one numbered impulse response for each microphone.')
    # Set the band below the 3.43 kHz spatial Nyquist limit. Derivatives use
    # dimensionless time t * 343 and coordinates divided by one metre.
    frequency, decimation = 48_000, 8
    reference_speed = 343
    sos = butter(6, [180, 900], btype='bandpass', fs=frequency, output='sos')
    nt = 576
    field = np.empty((nt, 21, 21, 9), dtype=np.float32)
    offset = _impulse_offset(files, mic, source[0], reference_speed, frequency)
    distance = np.linalg.norm(nearest - source[0])
    start = max(0, offset + int(distance / reference_speed * frequency) - 480)
    stop = start + nt * decimation
    if stop > 32768:
        raise ValueError('The impulse window extends beyond the available response.')
    for i, location in enumerate(zip(*index)):
        ir = np.load(files[i], mmap_mode='r')
        if ir.shape != (1, 32768):
            raise ValueError(f'Unexpected impulse-response shape for microphone {i}.')
        # Warm up the causal filter before the crop so split edges do not reset it.
        filtered = sosfilt(sos, ir[0, :stop])
        field[(slice(None), *location)] = filtered[start:stop:decimation]
    scale = np.sqrt(np.mean(field[:288, :7].astype(float)**2))
    if not np.isfinite(field).all() or scale <= 0:
        raise ValueError('Impulse responses are non-finite or the training segment is silent.')
    field /= scale
    splits = ((slice(0, 288), slice(0, 7)), (slice(288, 432), slice(7, 14)),
              (slice(432, 576), slice(14, 21)))
    return WaveData('MeshRIR S1-M3969', field, (reference_speed * decimation / frequency, 0.05, 0.05, 0.05),
                    splits, None, '180–900 Hz band, 6 kHz effective sampling, and disjoint time and x blocks. '
                    f'Selected volume lies at least {distance:.3f} m from the source. '
                    f'Training-derived shared delay: {1e3 * offset / frequency:.2f} ms. '
                    f'Window starts at sample {start} of the original responses. '
                    'For a Laplacian-only fit, multiply √θ by 343 to express the fitted speed in m/s.')


def window_shuffle(values, width, rng):
    """Permute snapshots within each time window while preserving spatial structure."""
    if width < 1:
        raise ValueError('Shuffle windows require a positive width.')
    permutation = np.concatenate([rng.permutation(np.arange(lo, min(lo + width, len(values))))
                                  for lo in range(0, len(values), width)])
    return values[permutation]


class WeakRows:
    """Sample separable compact test functions entirely inside each data block."""

    def __init__(self, blocks, spacing, inputs=None):
        self.blocks = tuple(blocks)
        self.inputs = {'u': self.blocks, **(inputs or {})}
        self.spacing = spacing
        self.cols = {}
        self.strong = lru_cache(maxsize=36)(self._strong)
        self.geometry = []
        groups = []
        centers = []
        # One group per independent time block; each stencil has a two-point halo.
        for i, block in enumerate(self.blocks):
            axes = [np.arange(3, n - 3, step) for n, step in zip(block.shape, (3, 4, 5, 4))]
            if any(len(axis) == 0 for axis in axes):
                raise ValueError('Each space–time block needs at least seven samples along each axis.')
            self.geometry.append(axes)
            subindex = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 4)
            groups.extend([i] * len(subindex))
            centers.append(subindex)
        self.groups = np.array(groups)
        self.centers = np.concatenate(centers)
        self.target = self._column(('u',), (2, 0, 0, 0))

    def _project(self, value, index, orders):
        # Correlating the smoothing weights with centered derivative stencils is
        # the discrete adjoint identity: field derivatives are never formed here.
        for axis, (order, h) in enumerate(zip(orders, self.spacing, strict=True)):
            kernel = np.array([1., 2., 1.]) / 4
            if order == 1:
                kernel = np.convolve(kernel, [-0.5, 0, 0.5]) / h
            elif order == 2:
                kernel = np.convolve(kernel, [1, -2, 1]) / h**2
            value = correlate1d(value, kernel, axis=axis, mode='constant')
            value = np.take(value, self.geometry[index][axis], axis=axis)
        return value.ravel().astype(float)

    def _column(self, expr, orders):
        op, *args = expr
        if op in AXES or op in ('lap', 'u_t'):
            axes = tuple(range(1, 4)) if op == 'lap' else (0 if op == 'u_t' else AXES.index(op) + 1,)
            order = 2 if op == 'lap' else 1
            child = ('u',) if op == 'u_t' else args[0]
            return sum(self._column(child, tuple(a + order * (i == axis) for i, a in enumerate(orders)))
                       for axis in axes)
        if op == 'div':
            return sum(np.concatenate([self._project(self.strong(args[0], j)[axis-1], j,
                                                      tuple(a + (i == axis) for i, a in enumerate(orders)))
                                       for j in range(len(self.blocks))]) for axis in range(1, 4))
        return np.concatenate([self._project(self.strong(expr, j), j, orders)
                               for j in range(len(self.blocks))])

    def _strong(self, expr, j):
        # Inner derivatives of nonlinear expressions cannot in general be moved
        # onto the test function. Use centered differences on the filtered field.
        op, *args = expr
        if op in self.inputs:
            return self.inputs[op][j].astype(np.float64, copy=False)
        if op == 'u_t':
            return self._difference(self.strong(('u',), j), 0)
        if op in AXES:
            axis = AXES.index(op) + 1
            if args[0][0] == op:
                return self._difference(self.strong(args[0][1], j), axis, order=2)
            return self._difference(self.strong(args[0], j), axis)
        a, *rest = [self.strong(arg, j) for arg in args]
        if op == 'lap':
            return sum(self._difference(a, axis, order=2) for axis in (1, 2, 3))
        if op == 'grad':
            return tuple(self._difference(a, axis) for axis in (1, 2, 3))
        if op == 'div':
            return sum(self._difference(v, axis) for axis, v in enumerate(a, 1))
        if op == 'vec':
            return (a, *rest)
        if op == 'scale':
            return tuple(a * value for value in rest[0])
        if op == 'dot':
            return sum(p * q for p, q in zip(a, rest[0], strict=True))
        if op == 'add':
            return a + rest[0]
        if op == 'mul':
            return a * rest[0]
        return {'square': np.square, 'cube': lambda v: v**3, 'sin': np.sin, 'tanh': np.tanh}[op](a)

    def _difference(self, values, axis, order=1):
        stencil = [-0.5, 0, 0.5] if order == 1 else [1, -2, 1]
        return correlate1d(values, np.array(stencil) / self.spacing[axis]**order, axis, mode='nearest')

    def matrix(self, graph):
        for branch in graph:
            if branch not in self.cols:
                with np.errstate(over='ignore', invalid='ignore'):
                    self.cols[branch] = self._column(branch, (0, 0, 0, 0))
                self.strong.cache_clear()
        columns = [self.cols[branch] for branch in graph]
        return np.column_stack(columns) if columns else np.empty((len(self.target), 0))


def coefficient_sign(branch):
    return int(branch == LAPLACIAN or branch in PURE)


def make_blocks(data, split, window=1, seed=0):
    time, space = data.splits[split]
    views = [data.u[lo:min(lo + 24, time.stop), space] for lo in range(time.start, time.stop, 24)]
    rng = np.random.default_rng(seed)
    blocks = [v if window == 1 else window_shuffle(v, window, rng) for v in views]
    inputs = {}
    for name, values in (data.inputs or {}).items():
        if values.ndim == 3:
            values = np.broadcast_to(values, data.u.shape)
        inputs[name] = [values[lo:min(lo + 24, time.stop), space]
                        for lo in range(time.start, time.stop, 24)]
    return WeakRows(blocks, data.spacing, inputs)
