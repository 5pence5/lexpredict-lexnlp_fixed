"""Coverage tests for uncovered lines in lexnlp.extract.pt.identifiers."""

from unittest import TestCase

from lexnlp.extract.pt.identifiers import (
    _cnpj_is_valid,
    _cpf_is_valid,
    get_cnpj_annotations,
    get_cpf_annotations,
    get_oab_annotations,
)


class TestFirstCheckDigitBranches(TestCase):
    def test_cpf_first_check_digit_failure(self):
        # "52998224725" is valid; mutating the first check digit (index 9:
        # 2 -> 3) fails the first check (line 136), unlike the existing
        # second-digit test ("...724").
        self.assertTrue(_cpf_is_valid("52998224725"))
        self.assertFalse(_cpf_is_valid("52998224735"))
        # The extractor filters it out even though the regex matches.
        self.assertEqual([], list(get_cpf_annotations("CPF 529.982.247-35")))
        ret = list(get_cpf_annotations("CPF 529.982.247-25"))
        self.assertEqual(1, len(ret))
        self.assertEqual("52998224725", ret[0].value)

    def test_cnpj_first_check_digit_failure(self):
        # "11222333000181" is valid; mutating the first check digit
        # (index 12: 8 -> 7) fails the first check (line 171).
        self.assertTrue(_cnpj_is_valid("11222333000181"))
        self.assertFalse(_cnpj_is_valid("11222333000171"))
        self.assertEqual([], list(get_cnpj_annotations("CNPJ 11.222.333/0001-71")))
        ret = list(get_cnpj_annotations("CNPJ 11.222.333/0001-81"))
        self.assertEqual(1, len(ret))
        self.assertEqual("11222333000181", ret[0].value)


class TestToDictionary(TestCase):
    def test_cpf_to_dictionary_exact(self):
        ret = list(get_cpf_annotations("CPF 529.982.247-25"))
        self.assertEqual(1, len(ret))
        self.assertEqual(
            {
                "record_type": "cpf",
                "coords": (4, 18),
                "text": "529.982.247-25",
                "value": "52998224725",
                "locale": "pt",
            },
            ret[0].to_dictionary(),
        )

    def test_oab_to_dictionary_exact(self):
        text = "Representado pelo Dr. Joao Silva, OAB/SP 123.456."
        ret = list(get_oab_annotations(text))
        self.assertEqual(1, len(ret))
        self.assertEqual(
            {
                "record_type": "oab",
                "coords": (34, 48),
                "text": "OAB/SP 123.456",
                "value": "SP/123456",
                "locale": "pt",
            },
            ret[0].to_dictionary(),
        )
