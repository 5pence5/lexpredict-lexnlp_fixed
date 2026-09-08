"""Final-coverage tests for :mod:`lexnlp.extract.en.definition_parsing_methods`."""

from lexnlp.extract.en.definition_parsing_methods import (
    DefinitionCaught,
    filter_definitions_for_self_repeating,
)


def test_filter_drops_earlier_definition_consumed_by_later() -> None:
    inner = DefinitionCaught("Obligation", "sentence", (0, 10))
    outer = DefinitionCaught("Obligations", "sentence", (2, 8))
    assert inner.does_consume_target(outer) == -1
    result = filter_definitions_for_self_repeating([inner, outer])
    assert inner.name is None
    assert result == [outer]
