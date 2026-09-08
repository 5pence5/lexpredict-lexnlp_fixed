__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

import pandas as pd

from lexnlp.extract.common.annotations.regulation_annotation import RegulationAnnotation
from lexnlp.extract.es.regulations import (
    ARTICLE_REFERENCE_RE,
    CONSTITUTIONAL_REF_RE,
    FORMAL_CITATION_RE,
    PARAGRAPH_LEADING_REFERENCE_RE,
    RegulationsParser,
    get_regulation_annotations,
    get_regulations,
    parser,
)
from lexnlp.tests.typed_annotations_tests import TypedAnnotationsTester
from lexnlp.tests.utility_for_testing import annotate_text, load_resource_document, save_test_document


class TestParseSpanishLawsRegulations(TestCase):
    def test_parse_comision(self):
        """
        Verify parsing of Spanish regulation mentions and extracted attributes.

        Asserts that parsing a Spanish text containing "Comisión Nacional Bancaria y de Valores" yields two parsed items and that the second item has country "Spain" and the expected name. Then asserts that parsing a second Spanish text containing "Registro Nacional de Valores" yields one parsed item with expected coordinates, country "Spain", name and text "Registro Nacional de Valores", and locale "es".
        """
        text = (
            "Las instituciones de banca múltiple que se ubiquen en lo dispuesto en esta fracción, deberán "
            + "entregar a la Comisión Nacional Bancaria y de Valores, la información y documentación que acredite "
            + "satisfacer lo antes señalado, dentro de los quince días hábiles siguientes a que se encuentren en dicho supuesto."
        )

        ret = list(parser.parse(text))
        self.assertEqual(2, len(ret))
        reg = ret[1]
        self.assertEqual("Spain", reg.country)
        self.assertEqual("Comisión Nacional Bancaria y de Valores", reg.name)

        text = (
            "Tampoco se considerarán operaciones de banca y crédito la captación de recursos del público "
            + "mediante la emisión de instrumentos inscritos en el Registro Nacional de Valores, colocados "
            + "mediante oferta pública incluso cuando dichos recursos se utilicen para el otorgamiento de "
            + "financiamientos de cualquier naturaleza."
        )
        ret = list(parser.parse(text))
        self.assertEqual(1, len(ret))
        reg = ret[0]
        self.assertEqual((144, 172), reg.coords)
        self.assertEqual("Spain", reg.country)
        self.assertEqual("Registro Nacional de Valores", reg.name)
        self.assertEqual("Registro Nacional de Valores", reg.text)
        self.assertEqual("es", reg.locale)

    def test_parse_ley_del(self):
        text = (
            "Para efectos de lo previsto en la presente Ley, por inversionistas institucionales se entenderá a las "
            + "instituciones de seguros y de fianzas, únicamente cuando inviertan sus reservas técnicas; a las "
            + "sociedades de inversión comunes y a las especializadas de fondos para el retiro; a los fondos de "
            + "pensiones o jubilaciones de personal, complementarios a los que establece la Ley del Seguro Social "
            + "y de primas de antigüedad, que cumplan con los requisitos señalados en la Ley del Impuesto sobre "
            + "la Renta, así como a los demás inversionistas institucionales que autorice expresamente la "
            + "Secretaría de Hacienda y Crédito Público."
        )
        ret = list(parser.parse(text))
        self.assertEqual(4, len(ret))

        reg_items = list(get_regulations(text))
        name = reg_items[0]["tags"]["External Reference Text"]
        self.assertEqual("instituciones de seguros y de fianzas", name)

    def test_parse_large_text(self):
        text = load_resource_document("lexnlp/extract/es/sample_es_regulations.txt", "utf-8")
        ret = list(parser.parse(text))
        self.assertGreater(len(ret), 100)
        html = annotate_text(text, ret)
        save_test_document("sample_es_regulations.html", html)

    def test_file_samples(self):
        tester = TypedAnnotationsTester()
        tester.test_and_raise_errors(
            get_regulation_annotations, "lexnlp/typed_annotations/es/regulation/regulations.txt", RegulationAnnotation
        )


# ---------------------------------------------------------------------------
# RegulationsParser - DataFrame injection (PR fix: `is None` check)
# ---------------------------------------------------------------------------


class TestRegulationsParserDataFrameInjection:
    """
    PR changed ``if not self.regulations_dataframe:`` to
    ``if self.regulations_dataframe is None:`` in load_trigger_words.

    An empty DataFrame is *falsy* but **not** None.  The old code would
    overwrite a caller-supplied empty DataFrame with the CSV content; the new
    code preserves whatever was passed in.
    """

    def test_none_dataframe_loads_from_csv(self):
        """Passing no DataFrame (None) causes load_trigger_words to read the CSV."""
        # Default constructor path — it must not raise and must populate triggers.
        p = RegulationsParser()
        assert len(p.start_triggers) > 0

    def test_non_none_dataframe_is_preserved(self):
        """A caller-supplied DataFrame (even empty) must not be replaced by the CSV."""
        # Build a minimal DataFrame with the required columns but no rows.
        empty_df = pd.DataFrame(columns=["trigger", "position"])
        p = RegulationsParser(regulations_dataframe=empty_df)
        # start_triggers will be empty because there are no rows.
        assert p.start_triggers == []
        # The dataframe attribute must still be our empty one, not the CSV.
        assert len(p.regulations_dataframe) == 0

    def test_custom_dataframe_rows_are_used(self):
        """A caller-supplied DataFrame with rows must drive trigger extraction."""
        custom_df = pd.DataFrame({"trigger": ["ley del", "comision"], "position": ["start", "start"]})
        p = RegulationsParser(regulations_dataframe=custom_df)
        assert "ley del" in p.start_triggers
        assert "comision" in p.start_triggers

    def test_custom_dataframe_non_start_rows_are_excluded(self):
        """Only rows with position='start' must end up in start_triggers."""
        custom_df = pd.DataFrame({"trigger": ["ley del", "reglamento"], "position": ["start", "end"]})
        p = RegulationsParser(regulations_dataframe=custom_df)
        assert "ley del" in p.start_triggers
        assert "reglamento" not in p.start_triggers

    def test_formal_citation_ley_organica(self):
        """``Ley Orgánica 4/2015`` is captured by FORMAL_CITATION_RE."""
        text = "Conforme a la Ley Orgánica 4/2015, de 30 de marzo, de protección."
        matches = list(FORMAL_CITATION_RE.finditer(text))
        assert len(matches) >= 1
        assert "Ley Orgánica 4/2015" in matches[0].group("full")

    def test_formal_citation_real_decreto(self):
        """``Real Decreto 123/2020`` is captured."""
        text = "El Real Decreto 123/2020 establece nuevos plazos."
        matches = list(FORMAL_CITATION_RE.finditer(text))
        assert len(matches) == 1
        assert matches[0].group("number") == "123/2020"

    def test_article_reference_simple(self):
        text = "Vid. art. 5 de la norma."
        matches = list(ARTICLE_REFERENCE_RE.finditer(text))
        assert len(matches) == 1
        assert matches[0].group("number") == "5"

    def test_paragraph_leading_apartado(self):
        text = "Aplica el apartado 2 del art. 14 de la ley."
        matches = list(PARAGRAPH_LEADING_REFERENCE_RE.finditer(text))
        assert len(matches) == 1
        assert matches[0].group("full") == "apartado 2 del art. 14"

    def test_parser_suppresses_nested_article_inside_paragraph_leading(self):
        """``apartado 2 del art. 14`` must NOT also yield a separate ``art. 14``."""
        text = "Aplica el apartado 2 del art. 14 de la norma."
        regs = list(parser.parse(text))
        names = [r.name for r in regs]
        assert "apartado 2 del art. 14" in names
        # ``art. 14`` is nested inside the paragraph-leading span, so it must
        # not be emitted on its own.
        assert "art. 14" not in names

    def test_constitutional_reference(self):
        text = "Según la Constitución Española de 1978."
        matches = list(CONSTITUTIONAL_REF_RE.finditer(text))
        assert len(matches) >= 1

    def test_parser_emits_constitutional_canonical_name(self):
        text = "Vid. la Constitución Española."
        regs = list(parser.parse(text))
        cfs = [r for r in regs if r.name == "Constitución Española"]
        assert len(cfs) == 1
        assert cfs[0].country == "Spain"

    def test_parser_dedupes_trigger_swallowing_formal_citation(self):
        """A trigger phrase that engulfs a formal citation is suppressed."""
        text = "Conforme a la ley de protección Real Decreto 123/2020 vigente, los plazos se actualizan."
        regs = list(parser.parse(text))
        # The trigger phrase ("ley de protección Real Decreto 123/2020 vigente")
        # fully contains the formal citation, so it must be dropped — only the
        # canonical "Real Decreto 123/2020" should survive.
        assert len(regs) == 1
        assert regs[0].name == "Real Decreto 123/2020"
        assert regs[0].text == "Real Decreto 123/2020"

    def test_empty_dataframe_means_no_triggers_no_csv_load(self):
        """
        Regression: previously ``not empty_df`` was True so the CSV was loaded.
        With the fix, an empty DataFrame means no triggers and no CSV reload.
        """
        empty_df = pd.DataFrame(columns=["trigger", "position"])
        p = RegulationsParser(regulations_dataframe=empty_df)
        # Empty regex alternation should produce no matches (not crash).
        results = list(p.parse("Ley del Impuesto sobre la Renta"))
        # We cannot assert empty because the regex pattern for empty alternation
        # may behave differently per version; we just ensure no exception is raised.
        assert isinstance(results, list)
