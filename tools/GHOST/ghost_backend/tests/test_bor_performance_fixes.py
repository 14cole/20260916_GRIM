"""BoR runtime fixes: BLAS thread bounds, parallel near preparation, sparse constraints."""
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import kernels
from ghost_backend.bor import solver as bor

C0 = bor.C0
FREQUENCY_HZ = 1.0e9


def _sphere(radius, elements):
    theta = np.linspace(0.0, math.pi, int(elements) + 1)
    return np.column_stack((radius * np.sin(theta), radius * np.cos(theta)))


def _hemisphere(radius, elements, upper=True):
    start, stop = (0.0, 0.5 * math.pi) if upper else (0.5 * math.pi, math.pi)
    theta = np.linspace(start, stop, int(elements) + 1)
    return np.column_stack((radius * np.sin(theta), radius * np.cos(theta)))


class BoundedBlasThreadTests(unittest.TestCase):
    def limits_requested(self, cpus, workers, current_threads):
        info = [dict(user_api="blas", num_threads=current_threads)]
        with mock.patch.object(bor.os, "cpu_count", return_value=cpus), \
                mock.patch("ghost_backend.execution.thread_control.threadpool_info",
                           return_value=info), \
                mock.patch("ghost_backend.execution.thread_control.threadpool_limits") as limits:
            with bor._bounded_blas_threads(workers):
                pass
        return [call.kwargs["limits"] for call in limits.call_args_list]

    def test_concurrent_workers_share_the_cores(self):
        self.assertEqual(self.limits_requested(16, 4, 16), [4])
        self.assertEqual(self.limits_requested(16, 15, 16), [1])

    def test_single_worker_and_existing_lower_limits_are_left_alone(self):
        self.assertEqual(self.limits_requested(16, 1, 16), [])
        self.assertEqual(self.limits_requested(16, 4, 2), [])

    def test_mode_sweep_applies_the_bound_for_its_worker_count(self):
        seen = []
        original = bor._bounded_blas_threads

        def spy(workers):
            seen.append(workers)
            return original(workers)

        with mock.patch.object(bor, "_bounded_blas_threads", side_effect=spy):
            bor._mode_sweep(
                1, [90.0], ("VV",), 0, 1e-6,
                lambda m: (np.ones((1, 1), complex), None),
                lambda m, th, pol: np.ones(1, complex),
                lambda m, x, th, pol: complex(x[0]),
                workers=3,
            )
        self.assertEqual(seen, [3])


class NearKernelRefinementTests(unittest.TestCase):
    def test_only_unconverged_points_are_refined_and_values_are_per_point(self):
        k = 2.0 * math.pi * FREQUENCY_HZ / C0
        radius = 3.0 / k
        gen = kernels.Generatrix(_sphere(radius, 30))
        s, sp, _ = bor._cell_points("diag", depth=4)
        rp, zp, trp, tzp, *_ = bor._points_on_element(gen, 10, s)
        rq, zq, trq, tzq, *_ = bor._points_on_element(gen, 10, sp)
        n = len(rp)
        args = (rp, zp, np.full(n, trp), np.full(n, tzp),
                rq, zq, np.full(n, trq), np.full(n, tzq))
        sizes = []
        rule = kernels._mfie_kernels_near_rule

        def spy(*a, **kw):
            sizes.append(len(np.atleast_1d(a[0])))
            return rule(*a, **kw)

        spy.__name__ = rule.__name__
        values = kernels._checked_near_kernels(spy, args, k, 15, 48, 0, True)
        chunk_first = max(sizes)
        self.assertLess(min(sizes), chunk_first,
                        "later refinement rounds must evaluate a subset")

        # A point's kernel does not depend on which other points share its batch.
        for index in (0, n // 2, n - 1):
            alone = kernels._checked_near_kernels(
                rule, tuple(a[index:index + 1] for a in args), k, 15, 48, 0, True)
            for batch_value, single_value in zip(values, alone):
                scale = np.max(np.abs(single_value))
                np.testing.assert_allclose(batch_value[index], single_value[0],
                                           rtol=0, atol=1e-7 * scale)


class ParallelNearPreparationTests(unittest.TestCase):
    def test_parallel_preparation_matches_serial_bit_for_bit(self):
        radius = 2.0 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        serial = bor.BorPecSolver(_sphere(radius, 14), FREQUENCY_HZ)
        parallel = bor.BorPecSolver(_sphere(radius, 14), FREQUENCY_HZ)
        serial.prepare_operators(4, efie=True, mfie=True, workers=1)
        parallel.prepare_operators(4, efie=True, mfie=True, workers=4)
        for key, prepared in serial._near_contractions.items():
            other = parallel._near_contractions[key]
            for field in ("rows", "cols", "source_elems", "values"):
                np.testing.assert_array_equal(other[field], prepared[field])
        self.assertEqual(parallel.near_quadrature_order_max,
                         serial.near_quadrature_order_max)

    def test_map_preserves_order_and_propagates_failures(self):
        self.assertEqual(bor._map_near_pairs(lambda p: p * 2, range(40), 4),
                         [2 * p for p in range(40)])

        def failing(pair):
            if pair == 7:
                raise RuntimeError("abort requested")
            return pair

        with self.assertRaisesRegex(RuntimeError, "abort requested"):
            bor._map_near_pairs(failing, range(40), 4)

    def test_near_worker_count_is_bounded_by_scratch_budget(self):
        self.assertEqual(bor._near_preparation_workers(1), 1)
        cap = bor.NEAR_PREPARATION_SCRATCH_BYTES // bor._NEAR_TASK_SCRATCH_BYTES
        self.assertEqual(bor._near_preparation_workers(1000), cap)


class MultiRegionConstraintTests(unittest.TestCase):
    def banded_system(self):
        radius = C0 / (2.0 * math.pi * FREQUENCY_HZ)
        outer = 1.12 * radius
        surfaces = [
            (_hemisphere(outer, 4, upper=True), False),
            (_hemisphere(outer, 4, upper=False), False),
            (np.asarray([[outer, 0.0], [radius, 0.0]]), False),
            (_hemisphere(radius, 4, upper=True), True),
            (_hemisphere(radius, 4, upper=False), True),
        ]
        regions = [
            {"medium": None, "bounds": [(0, +1), (1, +1)], "exterior": True},
            {"medium": (2.4 - 0.06j, 1.0 - 0.01j), "bounds": [(0, -1), (2, +1), (3, +1)]},
            {"medium": (3.1 - 0.10j, 1.0 - 0.02j), "bounds": [(1, -1), (2, -1), (4, +1)]},
        ]
        return bor._MultiRegionBor(surfaces, regions, FREQUENCY_HZ,
                                   near_factor=2.0, near_order=12)

    def test_sparse_constraints_are_built_once_and_match_dense(self):
        system = self.banded_system()
        conversions = []
        original = bor.csr_matrix

        def counting(*args, **kwargs):
            conversions.append(1)
            return original(*args, **kwargs)

        with mock.patch.object(bor, "csr_matrix", side_effect=counting):
            first = system.rhs(1, 37.0, "VV")
            count = len(conversions)
            for theta in (0.0, 37.0, 90.0):
                for pol in ("VV", "HH"):
                    system.rhs(1, theta, pol)
                    system.farfield(1, first, theta, pol)
        self.assertEqual(len(conversions), count)

        full = np.zeros(system.n_full, complex)
        for (si, _) in system.regions[system.ext_region]["bounds"]:
            solver = system.solv[(si, system.ext_region)]
            full[system.off_J[si]:system.off_J[si] + 2 * system.Nn[si]] = \
                solver.rhs_mode(1, 37.0, "VV")
            if system.off_M[si] is not None:
                full[system.off_M[si]:system.off_M[si] + 2 * system.Nn[si]] = \
                    bor.ETA0 * solver.rhs_h_mode(1, 37.0, "VV")
        dense_q = np.asarray(system.build_Q(1))
        np.testing.assert_allclose(first, dense_q.conj().T @ full, rtol=1e-14, atol=0)

    def test_multiregion_signed_mode_symmetry_matches_full_signed_sweep(self):
        common = dict(freq_hz=FREQUENCY_HZ, thetas_deg=[0.0, 37.0, 90.0, 143.0],
                      n_modes=8, mode_tol=1.0e-6, workers=2, progress=None,
                      check_abort=None, formulation="banded", extra={},
                      table_precision="double", assembly="tables")
        reduced = bor._solve_multiregion(self.banded_system(), **common)
        self.assertTrue(reduced["signed_mode_symmetry_used"])

        impl = bor._mode_sweep_impl

        def without_symmetry(*args, **kwargs):
            kwargs["signed_mode_symmetry"] = False
            return impl(*args, **kwargs)

        with mock.patch.object(bor, "_mode_sweep_impl", side_effect=without_symmetry):
            full = bor._solve_multiregion(self.banded_system(), **common)
        self.assertFalse(full["signed_mode_symmetry_used"])
        for key in ("amp_vv", "amp_hh"):
            np.testing.assert_allclose(np.asarray(reduced[key]), np.asarray(full[key]),
                                       rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
