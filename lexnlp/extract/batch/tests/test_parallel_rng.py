__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"

import importlib
from unittest import TestCase

import numpy as np

from lexnlp.extract.batch.parallel_rng import spawn_child_generators


class TestSpawnChildGenerators(TestCase):
    def test_public_reexport_is_callable(self):
        """``spawn_child_generators`` is part of the package's public API
        surface and must remain importable from ``lexnlp.extract.batch``."""
        module = importlib.import_module("lexnlp.extract.batch")
        reexport = getattr(module, "spawn_child_generators")
        self.assertTrue(callable(reexport))
        # Sanity-check that the re-exported symbol is the same object as
        # the one importable from the submodule.
        self.assertIs(reexport, spawn_child_generators)
        # Smoke-call the public path so we exercise it end-to-end.
        children = reexport(seed=5, n=1)
        self.assertEqual(len(children), 1)

    def test_returns_n_generators(self):
        children = spawn_child_generators(seed=42, n=4)
        self.assertEqual(len(children), 4)
        for c in children:
            self.assertIsInstance(c, np.random.Generator)

    def test_zero_children_returns_empty_list(self):
        children = spawn_child_generators(seed=0, n=0)
        self.assertEqual(children, [])

    def test_negative_n_raises(self):
        with self.assertRaises(ValueError):
            spawn_child_generators(seed=0, n=-1)

    def test_children_streams_are_independent(self):
        """Sibling streams must not produce identical samples."""
        children = spawn_child_generators(seed=42, n=3)
        samples = [c.random(16) for c in children]
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                self.assertFalse(
                    np.allclose(samples[i], samples[j]),
                    msg=f"child {i} and child {j} produced identical samples",
                )

    def test_deterministic_for_same_seed(self):
        a = spawn_child_generators(seed=123, n=5)
        b = spawn_child_generators(seed=123, n=5)
        for ga, gb in zip(a, b, strict=True):
            self.assertTrue(np.array_equal(ga.random(8), gb.random(8)))

    def test_different_seeds_differ(self):
        a = spawn_child_generators(seed=1, n=2)
        b = spawn_child_generators(seed=2, n=2)
        self.assertFalse(np.allclose(a[0].random(8), b[0].random(8)))

    def test_accepts_parent_generator(self):
        parent = np.random.default_rng(99)
        children = spawn_child_generators(seed=parent, n=2)
        self.assertEqual(len(children), 2)

    def test_child_stream_reproducible_across_calls(self):
        """Spawning twice from the same integer seed yields identical streams."""
        first = spawn_child_generators(seed=7, n=1)[0].integers(0, 1000, size=10)
        second = spawn_child_generators(seed=7, n=1)[0].integers(0, 1000, size=10)
        self.assertTrue(np.array_equal(first, second))


class TestAdaptiveMaxWorkers(TestCase):
    """Tests for ``adaptive_max_workers``, which was added to ``__all__`` in this PR."""

    def test_in_async_extract_all(self):
        """adaptive_max_workers must be listed in async_extract.__all__."""
        from lexnlp.extract.batch.async_extract import __all__ as async_all

        self.assertIn("adaptive_max_workers", async_all)

    def test_importable_from_package(self):
        """adaptive_max_workers must be re-exported from lexnlp.extract.batch."""
        module = importlib.import_module("lexnlp.extract.batch")
        self.assertTrue(hasattr(module, "adaptive_max_workers"))
        self.assertTrue(callable(module.adaptive_max_workers))

    def test_package_all_contains_adaptive_max_workers(self):
        """__all__ in lexnlp.extract.batch must include adaptive_max_workers."""
        module = importlib.import_module("lexnlp.extract.batch")
        all_names = getattr(module, "__all__", [])
        self.assertIn("adaptive_max_workers", all_names)

    def test_returns_positive_integer(self):
        """adaptive_max_workers must always return a positive integer."""
        from lexnlp.extract.batch.async_extract import adaptive_max_workers

        result = adaptive_max_workers()
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_returns_at_least_one(self):
        """Even on a single-core machine the result must be >= 1."""
        from unittest.mock import patch

        from lexnlp.extract.batch.async_extract import adaptive_max_workers

        class FakeMemory:
            available = 0  # 0 GiB available

        with patch("psutil.cpu_count", return_value=1):
            with patch("psutil.virtual_memory", return_value=FakeMemory()):
                result = adaptive_max_workers()
        self.assertGreaterEqual(result, 1)

    def test_capped_by_physical_cores(self):
        """Result must not exceed the number of physical CPU cores."""
        from unittest.mock import patch

        from lexnlp.extract.batch.async_extract import adaptive_max_workers

        class FakeMemory:
            available = 1024**3 * 1000  # 1000 GiB — unlimited RAM

        with patch("psutil.cpu_count", return_value=4):
            with patch("psutil.virtual_memory", return_value=FakeMemory()):
                result = adaptive_max_workers()
        self.assertLessEqual(result, 4)

    def test_fallback_when_psutil_missing(self):
        """When psutil is not importable the function must return 8."""
        import sys
        from unittest.mock import patch

        from lexnlp.extract.batch.async_extract import adaptive_max_workers

        with patch.dict(sys.modules, {"psutil": None}):
            result = adaptive_max_workers()
        self.assertEqual(result, 8)
