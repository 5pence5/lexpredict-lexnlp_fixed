from __future__ import annotations

from ci.check_dist_contents import (
    is_forbidden_wheel_member,
    normalise_package_name,
)


def test_normalise_package_name_rejects_nested_test_data_lexnlp_directory():
    assert normalise_package_name("lexnlp-2.4.0a1/test_data/lexnlp/nlp/en/sota_segmentation/gold.json") is None


def test_normalise_package_name_accepts_wheel_and_sdist_package_members():
    expected = "lexnlp/nlp/en/sota_segmentation/runtime.json"
    assert normalise_package_name(expected) == expected
    assert normalise_package_name(f"lexnlp-2.4.0a1/{expected}") == expected


def test_wheel_policy_excludes_every_tests_directory_component():
    assert is_forbidden_wheel_member("tests/test_gate.py")
    assert is_forbidden_wheel_member("lexnlp/nlp/en/tests/test_sota_hierarchy_quality.py")
    assert is_forbidden_wheel_member("lexnlp/nlp/en/tests/fixture.json")
    assert not is_forbidden_wheel_member("lexnlp/nlp/en/segments/hierarchy.py")
