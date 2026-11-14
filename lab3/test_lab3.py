import unittest

import numpy as np

from .diffusion import sample
from .experiment import initial_pool, contains
from .extensions import CORRECTIONS, LEAVES, choose_extension, rollout
from .fitting import fit_coefficients, lab2_support
from .grammar import (LAPLACIAN, MAX_OPS, MIXED, PURE, Derivation, canonical_graph,
                      compile_dag, encode, equation_dag)
from .wave_data import WeakRows, sanity_data, window_shuffle


class Lab3Tests(unittest.TestCase):
    def test_grammar_roundtrip_and_dag(self):
        rng = np.random.default_rng(7)
        leaves = ('u', 'u_t', 'q', 'f')
        for _ in range(200):
            graph, tokens = sample(leaves, rng)
            state = Derivation(leaves)
            for token in tokens:
                state.add(token)
            self.assertEqual(state.equation(), graph)
            self.assertLessEqual(max(state.used), MAX_OPS)
            self.assertTrue(all(not count for count in state.reserved))
            encoded = Derivation(leaves)
            for token in encode(graph, leaves):
                encoded.add(token)
            self.assertEqual(encoded.equation(), graph)
            nodes, roots = compile_dag(graph)
            self.assertTrue(all(child < i for i, (_, children) in enumerate(nodes) for child in children))
            self.assertEqual(len(roots), len(graph))
            assembled = equation_dag(graph)
            self.assertEqual(assembled['rhs'], len(assembled['nodes']) - 1)
            self.assertEqual(sum(op.startswith('theta_') for op, _ in assembled['nodes']), len(graph))

    def test_canonicalization_preserves_coefficients(self):
        u = ('u',)
        lap = ('add', ('add', PURE[2], PURE[0]), PURE[1])
        self.assertEqual(canonical_graph([lap, ('div', ('grad', u))]), (LAPLACIAN,))
        self.assertEqual(canonical_graph(PURE), PURE)
        self.assertEqual(canonical_graph([('D_y', ('D_x', u))]), (MIXED,))
        self.assertEqual(canonical_graph([('mul', u, ('mul', u, u))]), (('cube', u),))
        samples = [('lap', ('square', u)), ('mul', ('square', u), ('u_t',)),
                   ('div', ('scale', ('q',), ('grad', u)))]
        self.assertTrue(encode(tuple(samples), ('u', 'q', 'u_t')))
        with self.assertRaises(ValueError):
            encode((('lap', ('D_x', u)),), ('u',))
        with self.assertRaises(ValueError):
            encode((('q',),), ('u',))
        for seed in (11, 22, 33, 44, 55):
            pool = initial_pool(('u', 'u_t'), seed, exclude=True)
            self.assertFalse(any(contains(branch, MIXED) for graph in pool for branch in graph))

    def test_fit_and_rank(self):
        rows = sanity_data()
        c2 = fit_coefficients(rows.matrix((LAPLACIAN,)), rows.target, rows.groups, [1])
        self.assertAlmostEqual(c2.theta[0], 1.0, places=2)
        self.assertTrue(c2.kept[0])
        directional = fit_coefficients(rows.matrix(PURE), rows.target, rows.groups, [1] * 3)
        self.assertEqual(directional.rank, 1)
        self.assertFalse(directional.kept.any())
        _, keeps = lab2_support(np.array([-3., -3., -3., 3.]), np.ones(4), np.array([0, 1, -1, -1]))
        np.testing.assert_array_equal(keeps, [True, False, True, False])

    def test_weak_polynomial_and_shuffle(self):
        t, x, y, z = np.ix_(*(np.arange(12, dtype=float),)*4)
        field = t**2 + x*y + np.zeros_like(z)
        rows = WeakRows([field], (1., 1., 1., 1.))
        np.testing.assert_allclose(rows.target, 2.)
        np.testing.assert_allclose(rows.matrix((MIXED,)), 1.)
        np.testing.assert_allclose(rows.matrix((LAPLACIAN,)), 0.)
        shuffled = window_shuffle(field, 3, np.random.default_rng(5))
        self.assertFalse(np.array_equal(field, shuffled))
        np.testing.assert_array_equal(field.reshape(4, 3, 12, 12, 12).sum(axis=1),
                                      shuffled.reshape(4, 3, 12, 12, 12).sum(axis=1))

    def test_conditional_search_and_forecast(self):
        base = (MIXED, LAPLACIAN)
        paths = {tuple(encode((*base, branch), LEAVES)): canonical_graph((*base, branch))
                 for branch in CORRECTIONS if branch not in base}
        self.assertEqual(len(paths), len(CORRECTIONS) - len(base))
        rng = np.random.default_rng(11)
        while paths:
            tokens = choose_extension(paths, rng, None)
            state = Derivation(LEAVES)
            for token in tokens:
                state.add(token)
            graph = paths.pop(tokens)
            self.assertEqual(state.equation(), graph)
            self.assertTrue(set(base) < set(graph))

        # On one periodic discrete mode, the recurrence has an analytic solution.
        n, dt, dx = 12, 0.025, 2 / 12
        x = 2 * np.pi * np.arange(n) / n
        t = np.arange(25) * dt
        damping = 2.2
        omega = np.sqrt(4 * np.sin(np.pi / n)**2 / dx**2 - damping**2 / 4)
        field = (np.exp(-damping * t[:, None] / 2) * np.cos(x[None, :] - omega * t[:, None]))[:, :, None, None]
        errors = rollout(field, (dt, dx, dx, dx), (LAPLACIAN, ('u_t',)), (1, -damping))
        self.assertLess(max(errors), 1e-22)

    def test_rollout_tuning_recovers_spectral_wave_speed(self):
        from .forecast_tuning import fit_rollout

        n, dt, dx = 12, 0.025, 2 / 12
        x = 2 * np.pi * np.arange(n) / n
        t = np.arange(48) * dt
        field = np.cos(x[None, :] - 0.92 * np.pi * t[:, None])[:, :, None, None]
        theta = fit_rollout(field[:32], (dt, dx, dx, dx), (LAPLACIAN,), (1.,), 16, True)
        np.testing.assert_allclose(theta, [0.92**2], atol=1e-9)
        errors = rollout(field[32:], (dt, dx, dx, dx), (LAPLACIAN,), theta, spectral=True)
        self.assertLess(max(errors), 1e-16)

    def test_phase_drift_limits_and_small_phase_error(self):
        from .phase_drift import phase_drift_estimate
        from .wave_data import synthetic_modes

        exact = dict(graph=(LAPLACIAN, MIXED), theta=(0.85, 0.72))
        self.assertTrue(np.isinf(phase_drift_estimate(exact)['times']).all())
        perturbed = dict(graph=(MIXED, LAPLACIAN), theta=(0.72000002, 0.84999998))
        drift = phase_drift_estimate(perturbed)
        _, amplitude, _, _ = synthetic_modes(mixed=True)
        # Independent exact spatial RMS for equal-amplitude, orthogonal traveling modes.
        phase = drift['times'][:, None] * drift['delta_omega'][None, :]
        exact_rms = np.sqrt(np.average(4 * np.sin(phase / 2)**2, axis=1, weights=amplitude**2))
        np.testing.assert_allclose(exact_rms, drift['thresholds'], rtol=0.002)
        reversed_model = dict(graph=perturbed['graph'][::-1], theta=perturbed['theta'][::-1])
        np.testing.assert_array_equal(phase_drift_estimate(reversed_model)['times'], drift['times'])
        with self.assertRaises(ValueError):
            phase_drift_estimate(dict(graph=(LAPLACIAN,), theta=(0.85,)))
        with self.assertRaises(ValueError):
            phase_drift_estimate(dict(graph=(LAPLACIAN, MIXED), theta=(-1., 0.)))


if __name__ == '__main__':
    unittest.main()
