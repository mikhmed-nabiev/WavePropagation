"""Coefficient estimation, support tests and intervention-aware model scores."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .grammar import complexity


@dataclass
class Fit:
    theta: np.ndarray
    sd: np.ndarray
    z: np.ndarray
    kept: np.ndarray
    rank: int
    condition: float


def lab2_support(theta, sd, signs):
    """Lab 2's standardized distance to zero, extended to either known sign."""
    z = np.divide(theta, sd, out=np.full_like(theta, np.nan), where=sd > 0)
    evidence = np.where(signs == 0, np.abs(z), signs * z)
    return z, evidence >= 2.0


def fit_coefficients(x, y, groups, signs):
    """Joint least squares with scale-invariant rank and clustered uncertainty."""
    n, p = x.shape
    signs = np.asarray(signs)
    if p == 0:
        blank = np.array([])
        return Fit(blank, blank, blank, blank.astype(bool), 0, 1.0)
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1e140:
        unresolved = np.full(p, np.nan)
        return Fit(np.zeros(p), unresolved, unresolved, np.zeros(p, dtype=bool), 0, np.inf)
    scales = np.linalg.norm(x, axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    a = x / scales
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    good = s > max(s[0] * 1e-9, np.finfo(float).eps)
    rank = int(good.sum())
    pinv = (vt[good].T / s[good]) @ u[:, good].T
    beta = pinv @ y
    if np.any(signs * beta < 0):
        beta = lsq_linear(a, y, bounds=(np.where(signs > 0, 0.0, -np.inf),
                                        np.where(signs < 0, 0.0, np.inf)), method='bvls').x
    errors = y - a @ beta
    _, group_id = np.unique(groups, return_inverse=True)
    n_groups = int(group_id.max()) + 1
    # Sum each block's influence before forming the cluster sandwich diagonal.
    influence = np.zeros((n_groups, p))
    np.add.at(influence, group_id, pinv.T * errors[:, None])
    variance = np.sum(influence**2, axis=0) * n_groups / max(n_groups - 1, 1)
    iid_variance = np.sum(pinv**2, axis=1) * (errors @ errors) / max(n - rank, 1)
    sd = np.sqrt(np.maximum(variance, iid_variance)) / scales
    sd = np.maximum(sd, np.finfo(float).eps * max(np.linalg.norm(y), 1.0) / scales)
    identified = np.sum(vt[good] ** 2, axis=0) > 1 - 1e-8
    sd[~identified] = np.nan
    theta = beta / scales
    z, kept = lab2_support(theta, sd, signs)
    condition = float(s[0] / s[-1]) if rank == p else np.inf
    return Fit(theta, sd, z, kept, rank, condition)


def intervention_rate(clean, intervened):
    return np.mean([clean.kept & fit.kept & (clean.theta * fit.theta > 0)
                    for fit in intervened], axis=0)


def normalized_error(y, prediction):
    with np.errstate(over='ignore', invalid='ignore'):
        error = np.mean((y - prediction) ** 2) / max(np.mean(y ** 2), np.finfo(float).tiny)
    return float(min(error, 1e12)) if np.isfinite(error) else 1e12


@dataclass
class Evaluation:
    graph: tuple
    fit: Fit
    validation: float
    complexity: int
    mild_support: np.ndarray
    strong_support: np.ndarray

    @property
    def eligible(self):
        return self.fit.rank == len(self.graph) and self.fit.kept.all()

    def score(self, stability_weight=0.035):
        return self.validation + 0.0015 * self.complexity + stability_weight * (1 - self.mild_support.mean())


class Scorer:
    def __init__(self, train, validation, test, mild, strong, constraint):
        self.train, self.validation, self.test = train, validation, test
        self.mild, self.strong = mild, strong
        self.constraint = constraint
        self.cache = {}

    def fit(self, graph, data):
        return fit_coefficients(data.matrix(graph), data.target, data.groups,
                                [self.constraint(branch) for branch in graph])

    def evaluate(self, graph):
        if graph not in self.cache:
            fitted = self.fit(graph, self.train)
            with np.errstate(over='ignore', invalid='ignore'):
                validation = normalized_error(self.validation.target, self.validation.matrix(graph) @ fitted.theta)
            self.cache[graph] = Evaluation(graph, fitted, validation, complexity(graph),
                                           intervention_rate(fitted, [self.fit(graph, d) for d in self.mild]),
                                           intervention_rate(fitted, [self.fit(graph, d) for d in self.strong]))
        return self.cache[graph]

    def heldout(self, evaluation):
        prediction = self.test.matrix(evaluation.graph) @ evaluation.fit.theta
        return normalized_error(self.test.target, prediction), prediction

    def ablate(self, evaluation):
        results = []
        for idx, branch in enumerate(evaluation.graph):
            if evaluation.fit.kept[idx]:
                reduced = evaluation.graph[:idx] + evaluation.graph[idx + 1:]
                fitted = self.fit(reduced, self.train)
                error = normalized_error(self.validation.target, self.validation.matrix(reduced) @ fitted.theta)
                results.append((branch, fitted.theta, error - evaluation.validation))
        return results
