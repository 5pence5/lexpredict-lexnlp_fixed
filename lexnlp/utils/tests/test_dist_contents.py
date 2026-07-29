from ci.check_dist_contents import (
    find_duplicate_members,
    find_duplicate_resources,
    find_resource_failures,
)


def test_resource_parity_rejects_missing_and_unexpected_files():
    expected = {
        "lexnlp/data/expected.json": b"expected",
        "lexnlp/data/missing.json": b"missing",
    }
    actual = {
        "lexnlp/data/expected.json": b"expected",
        "lexnlp/data/unexpected.json": b"unexpected",
    }

    failures = find_resource_failures(expected, actual, "archive")

    assert failures == [
        "archive: missing runtime resource: lexnlp/data/missing.json",
        "archive: unexpected runtime resource: lexnlp/data/unexpected.json",
    ]


def test_duplicate_archive_members_and_resources_are_rejected():
    names = [
        "lexnlp-2.3.0/lexnlp/data/model.json",
        "lexnlp-2.3.0/lexnlp/data/model.json",
        "other-root/lexnlp/data/model.json",
    ]

    assert find_duplicate_members(names) == [
        "lexnlp-2.3.0/lexnlp/data/model.json",
    ]
    assert find_duplicate_resources(names) == [
        "lexnlp/data/model.json",
    ]
