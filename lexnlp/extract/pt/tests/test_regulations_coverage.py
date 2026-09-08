"""Coverage tests for :mod:`lexnlp.extract.pt.regulations` missing lines."""

from lexnlp.extract.common.annotations.regulation_annotation import RegulationAnnotation
from lexnlp.extract.pt.regulations import (
    get_regulation_annotation_list,
    get_regulation_annotations,
    get_regulation_list,
    get_regulations,
)

TEXT = "Lei nº 12.527, de 18 de novembro de 2011"


def test_get_regulation_annotation_list() -> None:
    result = get_regulation_annotation_list(TEXT)
    assert isinstance(result, list)
    # Trigger phrase first, then the formal citation spanning the full date.
    assert len(result) == 2
    trigger, formal = result
    assert isinstance(trigger, RegulationAnnotation)
    assert (trigger.name, trigger.text, trigger.coords) == (
        "Lei nº 12.527",
        "Lei nº 12.527",
        (0, 13),
    )
    assert (formal.name, formal.text, formal.coords) == (
        "Lei nº 12.527, de 18 de novembro de 2011",
        "Lei nº 12.527, de 18 de novembro de 2011",
        (0, 40),
    )
    for ant in result:
        assert ant.locale == "pt"
        assert ant.country == "Brazil"
    assert [a.text for a in result] == [a.text for a in get_regulation_annotations(TEXT)]
    assert get_regulation_annotation_list("Sem normas aqui.") == []


def test_get_regulations_yields_dictionaries() -> None:
    result = list(get_regulations(TEXT))
    assert len(result) == 2
    row = result[0]
    assert row["attrs"] == {"start": 0, "end": 13}
    assert row["tags"]["Extracted Entity Type"] == "regulation"
    assert row["tags"]["External Reference Issuing Country"] == "Brazil"
    assert row["tags"]["Extracted Entity Text"] == "Lei nº 12.527"
    assert result[1]["attrs"] == {"start": 0, "end": 40}
    assert list(get_regulations("Sem normas aqui.")) == []


def test_get_regulation_list_defaults_language_to_pt() -> None:
    explicit = get_regulation_list(TEXT, "pt")
    defaulted = get_regulation_list(TEXT, None)
    implicit = get_regulation_list(TEXT)
    assert len(explicit) == 2
    assert defaulted == explicit
    assert implicit == explicit
    assert get_regulation_list("Sem normas aqui.") == []
