"""Short, boundary-independent forecasts for the measured Laplacian baseline."""

import numpy as np
from scipy.ndimage import laplace


def interior_forecasts(field, dt, dx, speeds, first_test_x):
    """Reuse two measured snapshots at each origin and score a common interior.

    The spatial stencil can reach one grid point farther per substep. The
    returned crop excludes every point whose stencil can reach outside the grid.
    """
    speeds = np.asarray(speeds)
    substeps = max(1, int(np.ceil(np.sqrt(3) * speeds.max() * dt / dx)))
    horizon = min(3, (min(field.shape[1:]) - 1) // (2 * substeps),
                  (field.shape[1] - first_test_x - 1) // substeps)
    if horizon < 1:
        raise ValueError("The measured grid is too shallow for this speed and sample interval.")

    radius = substeps * horizon
    interior = (slice(max(radius, first_test_x), -radius),
                slice(radius, -radius), slice(radius, -radius))
    origins = range(0, len(field) - horizon - 1, 4)
    actual, predictions = [], [[] for _ in range(len(speeds) + 1)]
    for origin in origins:
        first, second = field[origin:origin + 2].astype(float)
        actual.append(field[(slice(origin + 2, origin + 2 + horizon), *interior)])
        predictions[0].append(np.stack([
            (second + step * (second - first))[interior] for step in range(1, horizon + 1)
        ]))
        for speed, output in zip(speeds, predictions[1:], strict=True):
            h = dt / substeps
            factor = (speed / dx)**2
            # Recover the previous substep from the two samples and the acceleration at the second.
            previous = second - (h / dt) * (second - first)
            previous += 0.5 * h * (h - dt) * factor * laplace(second, mode='constant')
            current = second
            forecast = []
            for step in range(1, horizon * substeps + 1):
                previous, current = current, 2 * current - previous + h**2 * factor * laplace(current, mode='constant')
                if step % substeps == 0:
                    forecast.append(current[interior])
            output.append(np.stack(forecast))

    return np.arange(1, horizon + 1) * dt, np.stack(actual), np.asarray(predictions)
