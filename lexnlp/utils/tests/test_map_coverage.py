"""Coverage tests for Map attribute fallback and deletion."""

import pytest

from lexnlp.utils.map import Map


def test_getattr_missing_key_returns_none() -> None:
    m = Map({"a": 1})
    assert m.a == 1
    assert m.missing is None
    assert m.get("missing") is None
    assert "missing" not in m
    assert "missing" not in m.__dict__


def test_delitem_removes_key_and_attr() -> None:
    m = Map({"a": 1})
    assert "a" in m
    assert "a" in m.__dict__
    del m["a"]
    assert "a" not in m
    assert "a" not in m.__dict__
    assert m.a is None


def test_delitem_missing_key_raises() -> None:
    m = Map({"a": 1})
    with pytest.raises(KeyError):
        del m["nope"]
    assert m["a"] == 1


def test_delattr_removes_key_and_attr() -> None:
    m = Map({"b": 2})
    assert m.b == 2
    del m.b
    assert "b" not in m
    assert "b" not in m.__dict__
    assert m.b is None


def test_delattr_missing_key_raises() -> None:
    m = Map({"b": 2})
    with pytest.raises(KeyError):
        del m.nope
    assert m["b"] == 2


def test_kwargs_init() -> None:
    m = Map(a=1, b=2)
    assert m["a"] == 1
    assert m["b"] == 2
    assert m.a == 1
