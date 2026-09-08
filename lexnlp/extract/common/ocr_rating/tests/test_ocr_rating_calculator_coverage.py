__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import pandas
import pytest

from lexnlp.extract.common.ocr_rating.ocr_rating_calculator import (
    BaseOcrRatingCalculator,
    CosineSimilarityOcrRatingCalculator,
    build_cs_quad_rating_calculator,
    build_cs_rating_calculator,
)

ENGLISH_TEXT = (
    "This Agreement is made and entered into as of the date set forth above by and "
    "between the parties hereto with reference to the following facts and circumstances."
)


class TestBaseOcrRatingCalculator:
    def test_get_rating_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            BaseOcrRatingCalculator().get_rating("some text", "en")


class TestBreakChars:
    def test_c_maps_to_o(self) -> None:
        assert BaseOcrRatingCalculator.break_chars("c") == "o"

    def test_d_maps_to_o(self) -> None:
        assert BaseOcrRatingCalculator.break_chars("d") == "o"

    def test_t_maps_to_plus(self) -> None:
        assert BaseOcrRatingCalculator.break_chars("t") == "+"

    def test_f_maps_to_plus(self) -> None:
        assert BaseOcrRatingCalculator.break_chars("f") == "+"

    def test_other_chars_pass_through(self) -> None:
        assert BaseOcrRatingCalculator.break_chars("x") == "x"
        assert BaseOcrRatingCalculator.break_chars("A") == "A"


class TestInitLanguageData:
    def test_loads_explicit_language_file_paths(self, tmp_path) -> None:
        frame = pandas.DataFrame({"prob": [0.5, 0.5]}, index=["ab", "cd"])
        pickle_path = str(tmp_path / "en.pickle")
        frame.to_pickle(pickle_path)
        calc = BaseOcrRatingCalculator()
        calc.init_language_data([], [pickle_path])
        assert "en" in calc.distribution_by_lang
        assert calc.distribution_by_lang["en"].equals(frame)

    def test_folder_files_do_not_overwrite_explicit_paths(self, tmp_path) -> None:
        explicit = pandas.DataFrame({"prob": [1.0]}, index=["ab"])
        explicit_path = str(tmp_path / "en.pickle")
        explicit.to_pickle(explicit_path)
        folder = tmp_path / "folder"
        folder.mkdir()
        other = pandas.DataFrame({"prob": [0.0]}, index=["zz"])
        other.to_pickle(str(folder / "en.pickle"))
        calc = BaseOcrRatingCalculator()
        calc.init_language_data([str(folder)], [explicit_path])
        assert calc.distribution_by_lang["en"].equals(explicit)

    def test_loads_languages_from_data_folders(self, tmp_path) -> None:
        frame = pandas.DataFrame({"prob": [0.25]}, index=["xy"])
        frame.to_pickle(str(tmp_path / "fr.pickle"))
        calc = BaseOcrRatingCalculator()
        calc.init_language_data([str(tmp_path)])
        assert calc.distribution_by_lang["fr"].equals(frame)


class TestCosineSimilarityCalculator:
    def test_unknown_language_falls_back_to_default(self) -> None:
        calc = build_cs_rating_calculator()
        assert "en" in calc.distribution_by_lang
        assert calc.get_cs(ENGLISH_TEXT, "xx-unknown") == calc.get_cs(ENGLISH_TEXT, "en")

    def test_english_text_scores_high(self) -> None:
        calc = build_cs_rating_calculator()
        rating = calc.get_rating(ENGLISH_TEXT, "en")
        assert rating >= 5

    def test_empty_text_scores_zero(self) -> None:
        calc = CosineSimilarityOcrRatingCalculator()
        assert calc.get_cs("", "en") == 0

    def test_quadratic_rating_is_bounded(self) -> None:
        calc = build_cs_quad_rating_calculator()
        rating = calc.get_rating(ENGLISH_TEXT, "en")
        assert 0 <= rating <= 10

    def test_quadratic_rating_of_empty_text_is_zero(self) -> None:
        calc = build_cs_quad_rating_calculator()
        assert calc.get_rating("", "en") == 0

    def test_get_file_rating_reads_file(self, tmp_path) -> None:
        calc = build_cs_rating_calculator()
        text_file = tmp_path / "sample.txt"
        text_file.write_text(ENGLISH_TEXT, encoding="utf-8")
        assert calc.get_file_rating(str(text_file), "en") == calc.get_rating(ENGLISH_TEXT, "en")
