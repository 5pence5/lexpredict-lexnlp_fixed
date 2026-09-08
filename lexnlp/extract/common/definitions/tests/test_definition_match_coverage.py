"""Coverage tests for lexnlp.extract.common.definitions.definition_match."""

from lexnlp.extract.common.definitions import definition_match
from lexnlp.extract.common.definitions.definition_match import DefinitionMatch


def test_module_metadata():
    assert definition_match.__author__ == "ContraxSuite, LLC; LexPredict, LLC"
    assert definition_match.__version__ == "2.3.0"


def test_defaults():
    match = DefinitionMatch()
    assert match.name is None
    assert match.start == 0
    assert match.end == 0
    assert match.probability == 0


def test_attribute_assignment():
    match = DefinitionMatch()
    match.name = "CDF"
    match.start = 0
    match.end = 16
    match.probability = 100
    assert match.name == "CDF"
    assert match.start == 0
    assert match.end == 16
    assert match.probability == 100


def test_instances_are_independent():
    first = DefinitionMatch()
    second = DefinitionMatch()
    first.name = "ABC"
    first.start = 5
    assert second.name is None
    assert second.start == 0


def test_instances_usable_as_parse_results():
    matches = [DefinitionMatch(), DefinitionMatch()]
    matches[0].name = "CDF"
    matches[0].start = 0
    matches[0].end = 16
    matches[0].probability = 100
    assert [(m.name, m.start, m.end, m.probability) for m in matches] == [
        ("CDF", 0, 16, 100),
        (None, 0, 0, 0),
    ]
