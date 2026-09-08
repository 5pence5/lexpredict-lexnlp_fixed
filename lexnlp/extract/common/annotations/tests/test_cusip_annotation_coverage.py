"""Coverage tests for lexnlp.extract.common.annotations.cusip_annotation."""

from lexnlp.extract.common.annotations.cusip_annotation import CusipAnnotation


def test_get_dictionary_values_minimal_has_only_code_and_internal():
    ann = CusipAnnotation(coords=(0, 9))
    df = ann.get_dictionary_values()
    assert df["tags"]["Extracted Entity Code"] is None
    assert df["tags"]["Extracted Entity Internal"] is None
    assert set(df["tags"].keys()) == {"Extracted Entity Code", "Extracted Entity Internal"}


def test_get_dictionary_values_with_code_and_internal():
    ann = CusipAnnotation(coords=(0, 9), code="03783310", internal=True)
    df = ann.get_dictionary_values()
    assert df["tags"]["Extracted Entity Code"] == "03783310"
    assert df["tags"]["Extracted Entity Internal"] is True


def test_get_dictionary_values_with_tba():
    ann = CusipAnnotation(coords=(0, 9), tba={"details": True})
    df = ann.get_dictionary_values()
    assert df["tags"]["Extracted Entity TBA"] == {"details": True}


def test_get_dictionary_values_with_string_ppn():
    ann = CusipAnnotation(coords=(0, 9), ppn="PPN123")
    df = ann.get_dictionary_values()
    assert df["tags"]["Extracted Entity PPN"] == "PPN123"


def test_get_dictionary_values_with_bool_ppn():
    ann = CusipAnnotation(coords=(0, 9), ppn=True)
    df = ann.get_dictionary_values()
    assert df["tags"]["Extracted Entity PPN"] is True


def test_get_dictionary_values_without_ppn():
    ann = CusipAnnotation(coords=(0, 9))
    assert "Extracted Entity PPN" not in ann.get_dictionary_values()["tags"]


def test_get_dictionary_values_with_zero_checksum():
    # A zero check digit is legitimate and must not be dropped.
    ann = CusipAnnotation(coords=(0, 9), checksum=0)
    df = ann.get_dictionary_values()
    assert "Extracted Entity Checksum" in df["tags"]
    assert df["tags"]["Extracted Entity Checksum"] == 0


def test_get_dictionary_values_with_string_checksum():
    ann = CusipAnnotation(coords=(0, 9), checksum="5")
    assert ann.get_dictionary_values()["tags"]["Extracted Entity Checksum"] == "5"


def test_get_dictionary_values_without_checksum():
    ann = CusipAnnotation(coords=(0, 9))
    assert "Extracted Entity Checksum" not in ann.get_dictionary_values()["tags"]


def test_get_dictionary_values_with_issuer_and_issue_ids():
    ann = CusipAnnotation(coords=(0, 9), issuer_id="ISR1", issue_id="ISS2")
    df = ann.get_dictionary_values()
    assert df["tags"]["Extracted Entity Issuer ID"] == "ISR1"
    assert df["tags"]["Extracted Entity Issue ID"] == "ISS2"


def test_get_dictionary_values_without_ids():
    ann = CusipAnnotation(coords=(0, 9))
    df = ann.get_dictionary_values()
    assert "Extracted Entity Issuer ID" not in df["tags"]
    assert "Extracted Entity Issue ID" not in df["tags"]


def test_get_dictionary_values_all_fields():
    ann = CusipAnnotation(
        coords=(0, 9),
        code="03783310",
        internal=False,
        ppn="PPN123",
        tba={"details": True},
        checksum=7,
        issuer_id="ISR1",
        issue_id="ISS2",
    )
    tags = ann.get_dictionary_values()["tags"]
    assert tags == {
        "Extracted Entity Code": "03783310",
        "Extracted Entity Internal": False,
        "Extracted Entity TBA": {"details": True},
        "Extracted Entity PPN": "PPN123",
        "Extracted Entity Checksum": 7,
        "Extracted Entity Issuer ID": "ISR1",
        "Extracted Entity Issue ID": "ISS2",
    }


def test_get_cite_value_parts_coerces_bool_ppn_to_empty():
    ann = CusipAnnotation(coords=(0, 9), code="03783310", ppn=True, issue_id="ISS2", issuer_id="ISR1")
    assert ann.get_cite_value_parts() == ["03783310", "", "ISS2", "ISR1"]


def test_get_cite_value_parts_keeps_string_ppn():
    ann = CusipAnnotation(coords=(0, 9), code="03783310", ppn="PPN123")
    assert ann.get_cite_value_parts() == ["03783310", "PPN123", "", ""]
