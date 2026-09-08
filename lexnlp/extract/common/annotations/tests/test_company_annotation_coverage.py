"""Coverage tests for lexnlp.extract.common.annotations.company_annotation."""

from lexnlp.extract.common.annotations.company_annotation import CompanyAnnotation


def test_repr_with_name_and_full_company_type():
    ann = CompanyAnnotation(coords=(0, 100), name="Acme", company_type_full="Corporation")
    assert repr(ann) == "Acme Corporation, (0, 100)"


def test_repr_with_abbreviated_company_type():
    ann = CompanyAnnotation(coords=(5, 15), name="Acme", company_type_abbr="LLC")
    assert repr(ann) == "Acme LLC, (5, 15)"


def test_repr_with_label_company_type():
    ann = CompanyAnnotation(coords=(5, 15), name="Acme", company_type_label="Corp")
    assert repr(ann) == "Acme Corp, (5, 15)"


def test_repr_without_company_type():
    ann = CompanyAnnotation(coords=(0, 100), name="Acme")
    assert repr(ann) == "Acme, (0, 100)"


def test_repr_falls_back_to_name_abbr():
    ann = CompanyAnnotation(coords=(0, 10), name_abbr="ABC")
    assert repr(ann) == "ABC, (0, 10)"


def test_repr_falls_back_to_text():
    ann = CompanyAnnotation(coords=(0, 10), text="some text here")
    assert repr(ann) == "some text here, (0, 10)"


def test_repr_with_nothing_falls_back_to_empty():
    ann = CompanyAnnotation(coords=(0, 10))
    assert repr(ann) == ", (0, 10)"


def test_company_type_prefers_full_over_abbr_and_label():
    ann = CompanyAnnotation(
        coords=(0, 10),
        name="Acme",
        company_type_full="Corporation",
        company_type_abbr="LLC",
        company_type_label="Corp",
    )
    assert ann.company_type == "Corporation"


def test_company_type_empty_when_all_missing():
    ann = CompanyAnnotation(coords=(0, 10), name="Acme")
    assert ann.company_type == ""


def test_get_cite_value_parts():
    ann = CompanyAnnotation(coords=(0, 100), name="Acme", company_type_full="Corporation")
    assert ann.get_cite_value_parts() == ["Acme", "Corporation"]


def test_get_cite_value_parts_empty_type():
    ann = CompanyAnnotation(coords=(0, 100), name="Acme")
    assert ann.get_cite_value_parts() == ["Acme", ""]


def test_get_cite_uses_name_and_type():
    ann = CompanyAnnotation(coords=(0, 4), name="Acme", company_type_abbr="LLC")
    assert ann.get_cite() == "/en/company/Acme/LLC"


def test_get_dictionary_values_full():
    ann = CompanyAnnotation(
        coords=(0, 100),
        name="Acme",
        company_type_full="Corporation",
        text="Acme Corp text",
    )
    df = ann.get_dictionary_values()
    assert df["tags"]["Extracted Entity Name"] == "Acme"
    assert df["tags"]["Extracted Entity Text"] == "Acme Corp text"
    assert df["tags"]["Extracted Entity Company"] == "Acme"
    assert df["tags"]["Extracted Entity Company Type"] == "Corporation"


def test_get_dictionary_values_text_falls_back_to_name():
    ann = CompanyAnnotation(coords=(0, 100), name="Acme")
    df = ann.get_dictionary_values()
    assert df["tags"]["Extracted Entity Text"] == "Acme"
    assert df["tags"]["Extracted Entity Company"] == "Acme"


def test_get_dictionary_values_without_name_or_type():
    ann = CompanyAnnotation(coords=(0, 100), text="just text")
    df = ann.get_dictionary_values()
    assert df["tags"]["Extracted Entity Name"] is None
    assert df["tags"]["Extracted Entity Text"] == "just text"
    assert "Extracted Entity Company" not in df["tags"]
    assert "Extracted Entity Company Type" not in df["tags"]
