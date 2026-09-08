"""Coverage tests for lexnlp.extract.es.regulations line 231."""

from lexnlp.extract.common.annotations.regulation_annotation import RegulationAnnotation
from lexnlp.extract.es.regulations import get_regulation_annotation_list, get_regulation_annotations


def test_get_regulation_annotation_list_formal_citation():
    text = "Conforme a la Ley Orgánica 4/2015, de 30 de marzo, de protección."
    result = get_regulation_annotation_list(text)
    assert isinstance(result, list)
    assert len(result) == 1
    ann = result[0]
    assert isinstance(ann, RegulationAnnotation)
    assert ann.name == "Ley Orgánica 4/2015, de 30 de marzo"
    assert ann.text == "Ley Orgánica 4/2015, de 30 de marzo"
    assert ann.coords == (14, 49)
    assert ann.locale == "es"
    assert ann.country == "Spain"


def test_get_regulation_annotation_list_matches_generator():
    text = "Vid. art. 5 de la norma."
    result = get_regulation_annotation_list(text)
    expected = list(get_regulation_annotations(text))
    assert [(a.name, a.coords, a.text, a.locale) for a in result] == [
        (a.name, a.coords, a.text, a.locale) for a in expected
    ]
    assert len(result) == 1
    assert result[0].name == "art. 5"


def test_get_regulation_annotation_list_empty():
    assert get_regulation_annotation_list("Hola mundo sin referencias legales.") == []
    assert get_regulation_annotation_list("") == []
