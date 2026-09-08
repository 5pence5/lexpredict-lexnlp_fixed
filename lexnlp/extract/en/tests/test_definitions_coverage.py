__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import os

import pytest

from lexnlp.extract.common.annotation_locator_type import AnnotationLocatorType
from lexnlp.extract.en.definitions import (
    get_definition_annotations,
    get_definitions,
    parser_ml_classifier,
)
from lexnlp.extract.ml.environment import ENV_EN_DATA_DIRECTORY

TEXT = (
    '"Consolidated EBITDA" means, for any period, for the Company and its Subsidiaries '
    "on a consolidated basis, an amount equal to Consolidated Net Income for such period"
)

TRAINED_MODEL_PATH = os.path.join(ENV_EN_DATA_DIRECTORY, "definition_model_layered.skops.zip")


def _ensure_ml_loaded() -> None:
    if not parser_ml_classifier.initialized:
        parser_ml_classifier.load_compressed(TRAINED_MODEL_PATH)


class TestGetDefinitionsReturnSources:
    def test_return_sources_yields_name_and_text(self) -> None:
        defs = list(get_definitions(TEXT, return_sources=True))
        assert len(defs) == 1
        name, source = defs[0]
        assert name == "Consolidated EBITDA"
        assert "means, for any period" in source
        assert name in source or "Consolidated EBITDA" in source

    def test_return_coords_takes_precedence_over_return_sources(self) -> None:
        defs = list(get_definitions(TEXT, return_sources=True, return_coords=True))
        assert len(defs) == 1
        name, source, coords = defs[0]
        assert name == "Consolidated EBITDA"
        assert coords == (1, 20)
        assert source.startswith('"Consolidated EBITDA" means')

    def test_plain_yields_names_only(self) -> None:
        defs = list(get_definitions(TEXT))
        assert defs == ["Consolidated EBITDA"]


class TestGetDefinitionsMlLocator:
    def test_ml_plain_yields_term_names(self) -> None:
        _ensure_ml_loaded()
        defs = list(get_definitions(TEXT, locator_type=AnnotationLocatorType.MlWordVectorBased))
        assert len(defs) >= 1
        assert any("Consolidated EBITDA" in name for name in defs)
        assert defs[0].strip('"') == "Consolidated EBITDA"

    def test_ml_return_sources_yields_name_text_pairs(self) -> None:
        _ensure_ml_loaded()
        defs = list(
            get_definitions(
                TEXT,
                return_sources=True,
                locator_type=AnnotationLocatorType.MlWordVectorBased,
            )
        )
        assert len(defs) >= 1
        name, source = defs[0]
        assert "Consolidated EBITDA" in name
        assert "Consolidated EBITDA" in source

    def test_ml_not_initialized_raises(self) -> None:
        _ensure_ml_loaded()
        original = parser_ml_classifier.initialized
        parser_ml_classifier.initialized = False
        try:
            with pytest.raises(Exception, match="should be initialized"):
                list(get_definitions(TEXT, locator_type=AnnotationLocatorType.MlWordVectorBased))
        finally:
            parser_ml_classifier.initialized = original

    def test_ml_annotations_success(self) -> None:
        _ensure_ml_loaded()
        ants = list(get_definition_annotations(TEXT, locator_type=AnnotationLocatorType.MlWordVectorBased))
        assert len(ants) >= 1
        assert "Consolidated EBITDA" in ants[0].name
        start, end = ants[0].coords
        assert TEXT[start:end] == ants[0].text


class TestGetDefinitionAnnotationsMlNotInitialized:
    def test_raises_when_not_initialized(self) -> None:
        _ensure_ml_loaded()
        original = parser_ml_classifier.initialized
        parser_ml_classifier.initialized = False
        try:
            with pytest.raises(Exception, match="should be initialized"):
                list(get_definition_annotations(TEXT, locator_type=AnnotationLocatorType.MlWordVectorBased))
        finally:
            parser_ml_classifier.initialized = original
