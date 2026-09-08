"""Coverage tests for uncovered lines in lexnlp.extract.pt.pii."""

from unittest import TestCase

from lexnlp.extract.pt.pii import (
    PHONE_PTN_RE,
    _digits_only,
    get_phone_annotation_list,
    get_phone_annotations,
    get_phone_list,
    get_phones,
    get_pii_annotations,
)


class TestPtPiiCoverage(TestCase):
    def test_overlong_match_is_rejected(self):
        # "+55 011 91234-5678" matches the regex (14 digits) but the
        # length filter (line 100, `continue`) rejects it.
        text = "Ligar +55 011 91234-5678 agora."
        match = PHONE_PTN_RE.search(text)
        self.assertIsNotNone(match)
        self.assertEqual(14, len(_digits_only(match.group("phone"))))
        self.assertEqual([], list(get_phone_annotations(text)))
        self.assertEqual([], get_phone_list(text))

    def test_operator_prefix_mobile_is_rejected(self):
        # "032 11 91234-5678" (explicit carrier code + mobile) is also
        # 14 digits: regex hit, annotation rejected.
        text = "Discar 032 11 91234-5678 hoje."
        match = PHONE_PTN_RE.search(text)
        self.assertIsNotNone(match)
        self.assertEqual(14, len(_digits_only(match.group("phone"))))
        self.assertEqual([], get_phone_annotation_list(text))

    def test_get_phones_yields_canonical(self):
        text = "Contato: (11) 98765-4321 para mais informacoes."
        self.assertEqual(["11987654321"], list(get_phones(text)))
        self.assertEqual(["11987654321"], get_phone_list(text))

    def test_get_pii_annotations_matches_phone_annotations(self):
        text = "Contato: (11) 98765-4321 para mais informacoes."
        pii = list(get_pii_annotations(text))
        ants = get_phone_annotation_list(text)
        self.assertEqual(len(ants), len(pii))
        self.assertEqual(1, len(pii))
        self.assertEqual("11987654321", pii[0].phone)
        self.assertEqual("(11) 98765-4321", pii[0].text)
        self.assertEqual("pt", pii[0].locale)
        self.assertEqual(ants[0].coords, pii[0].coords)
        self.assertEqual([], list(get_pii_annotations("Sem telefones aqui.")))
