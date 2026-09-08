"""Coverage tests for :mod:`lexnlp.utils.decorators` exception paths."""

from __future__ import annotations

import types
from unittest import TestCase

from lexnlp.utils.decorators import handle_invalid_text, safe_failure


class TestSafeFailureGeneratorPaths(TestCase):
    def test_mid_iteration_error_is_suppressed(self) -> None:
        @safe_failure
        def gen() -> types.GeneratorType:
            yield 1
            yield 2
            raise RuntimeError("late boom")

        result = list(gen())
        self.assertEqual([1, 2], result)

    def test_mid_iteration_error_reraises_with_flag(self) -> None:
        @safe_failure
        def gen() -> types.GeneratorType:
            yield 1
            yield 2
            raise RuntimeError("late boom")

        with self.assertRaises(RuntimeError):
            list(gen(safe_failure=False))

    def test_successful_generator_yields_all_items(self) -> None:
        @safe_failure
        def gen() -> types.GeneratorType:
            yield 1
            yield 2
            yield 3

        self.assertEqual([1, 2, 3], list(gen()))


class TestSafeFailurePlainFunctionPaths(TestCase):
    def test_immediate_error_is_suppressed(self) -> None:
        @safe_failure
        def boom(value: int) -> int:
            raise ValueError("boom")

        wrapped = boom(1)
        self.assertIsInstance(wrapped, types.GeneratorType)
        self.assertEqual([], list(wrapped))

    def test_immediate_error_reraises_with_flag(self) -> None:
        @safe_failure
        def boom(value: int) -> int:
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            list(boom(1, safe_failure=False))

    def test_plain_success_yields_no_items(self) -> None:
        # The wrapper is always a generator: a non-generator return value
        # is computed but never yielded, so iteration is empty.
        @safe_failure
        def add(a: int, b: int) -> int:
            return a + b

        wrapped = add(1, 2)
        self.assertIsInstance(wrapped, types.GeneratorType)
        self.assertEqual([], list(wrapped))


class TestHandleInvalidTextDirectDecoration(TestCase):
    def test_direct_decoration_uses_default_return_value(self) -> None:
        def upper(text: str) -> str:
            return text.upper()

        wrapped = handle_invalid_text(upper)
        self.assertIsNone(wrapped(""))
        self.assertEqual("HI", wrapped("hi"))

    def test_direct_decoration_with_custom_return_value(self) -> None:
        def upper(text: str) -> str:
            return text.upper()

        wrapped = handle_invalid_text(upper, return_value="EMPTY")
        self.assertEqual("EMPTY", wrapped(""))
        self.assertEqual("HI", wrapped("hi"))
