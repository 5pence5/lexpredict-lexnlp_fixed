__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import os
import pickle

import pandas

_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
_FN_UNICODE_CHAR_CATEGORIES = os.path.join(_MODULE_PATH, "unicode_character_categories.pickle")
_FN_UNICODE_CHAR_CATEGORY_MAPPING = os.path.join(_MODULE_PATH, "unicode_character_category_mapping.pickle")
_FN_UNICODE_CHAR_TOP_CATEGORY_MAPPING = os.path.join(_MODULE_PATH, "unicode_character_top_category_mapping.pickle")


def _load_table(fn: str, ignore_error: bool):
    try:
        with open(fn, "rb") as f:
            return pickle.load(f)
    except OSError as e:
        print(f"Unable to load unicode lookup table: {fn}")
        if ignore_error:
            return None
        raise e


UNICODE_CHAR_CATEGORIES = _load_table(_FN_UNICODE_CHAR_CATEGORIES, True)
UNICODE_CHAR_CATEGORY_MAPPING = _load_table(_FN_UNICODE_CHAR_CATEGORY_MAPPING, True)
UNICODE_CHAR_TOP_CATEGORY_MAPPING = _load_table(_FN_UNICODE_CHAR_TOP_CATEGORY_MAPPING, True)


def build_lookup_tables(
    fn_char_categories, fn_char_category_mapping, fn_char_top_category_mapping, table_source: str | None = None
):
    """
    Build and save Unicode character lookup tables derived from UnicodeData.txt.

    Reads UnicodeData.txt (from `table_source` or the default Unicode FTP URL), extracts character groupings and category mappings, and writes three pickle files containing:

    - a mapping of category group names to lists of characters,
    - a mapping from each character to its full general category,
    - a mapping from each character to its top-level category letter.

    Parameters:
        fn_char_categories (str): Path to write the pickle file containing the category-group -> list-of-characters mapping.
        fn_char_category_mapping (str): Path to write the pickle file containing the character -> general category mapping (e.g., 'Lu', 'Nd').
        fn_char_top_category_mapping (str): Path to write the pickle file containing the character -> top-level category letter mapping (e.g., 'L', 'N').
        table_source (str | None): Optional URL or local path to UnicodeData.txt. If None, uses
            'ftp://ftp.unicode.org/Public/11.0.0/ucd/UnicodeData.txt'.

    Side effects:
        Writes three pickle files at the provided paths containing the described mappings.
    """
    # https://www.unicode.org/reports/tr44/#UnicodeData.txt

    url = table_source or "ftp://ftp.unicode.org/Public/11.0.0/ucd/UnicodeData.txt"
    df = pandas.read_csv(url, sep=";", header=None)

    df.columns = [
        "value",
        "name",
        "general_category",
        "canonical_combining_class",
        "bidi_class",
        "decomposition_type",
        "decomposition_mapping",
        "numeric_type",
        "numeric_value",
        "bidi_mirrored",
        "unicode_1_name",
        "iso_comment",
        "simple_uppercase_mapping",
        "simple_lowercase_mapping",
        "simple_titlecase_mapping",
    ]

    # Get all punctuation characters
    punctuation_category_list = ["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "P"]
    punctuation_character_set = (
        df.loc[df["general_category"].isin(punctuation_category_list), "value"]
        .apply(lambda x: chr(int(x, 16)))
        .tolist()
    )
    print(punctuation_character_set[0:10])

    # Get all punctuation characters
    punctuation_start_category_list = ["Ps", "Pi"]
    punctuation_start_character_set = (
        df.loc[df["general_category"].isin(punctuation_start_category_list), "value"]
        .apply(lambda x: chr(int(x, 16)))
        .tolist()
    )
    print(punctuation_start_character_set[0:10])

    # Get all punctuation characters
    punctuation_end_category_list = ["Pe", "Pf"]
    punctuation_end_character_set = (
        df.loc[df["general_category"].isin(punctuation_end_category_list), "value"]
        .apply(lambda x: chr(int(x, 16)))
        .tolist()
    )
    print(punctuation_end_character_set[0:10])

    # Get all punctuation characters
    symbol_currency_category_list = ["Sc"]
    symbol_currency_character_set = (
        df.loc[df["general_category"].isin(symbol_currency_category_list), "value"]
        .apply(lambda x: chr(int(x, 16)))
        .tolist()
    )
    print(symbol_currency_character_set[0:10])

    # Get all math characters
    symbol_math_category_list = ["Sm"]
    symbol_math_character_set = (
        df.loc[df["general_category"].isin(symbol_math_category_list), "value"]
        .apply(lambda x: chr(int(x, 16)))
        .tolist()
    )
    print(symbol_math_character_set[0:10])

    # Get all math characters
    symbol_category_list = ["Sk", "So"]
    symbol_character_set = (
        df.loc[df["general_category"].isin(symbol_category_list), "value"].apply(lambda x: chr(int(x, 16))).tolist()
    )
    # symbol_character_set[0:10]

    # Get all math characters
    blankspace_category_list = ["Zs", "Zl", "Zp", "Cc", "Cf", "Cs"]
    blankspace_character_set = (
        df.loc[df["general_category"].isin(blankspace_category_list), "value"].apply(lambda x: chr(int(x, 16))).tolist()
    )
    print(blankspace_character_set[0:10])

    # Get all math characters
    space_category_list = ["Zs"]
    space_character_set = (
        df.loc[df["general_category"].isin(space_category_list), "value"].apply(lambda x: chr(int(x, 16))).tolist()
    )
    print(space_character_set[0:10])

    # Get all math characters
    line_category_list = ["Zl"]
    line_character_set = (
        df.loc[df["general_category"].isin(line_category_list), "value"].apply(lambda x: chr(int(x, 16))).tolist()
    )
    print(line_character_set[0:10])

    # Get all math characters
    para_category_list = ["Zp"]
    para_character_set = (
        df.loc[df["general_category"].isin(para_category_list), "value"].apply(lambda x: chr(int(x, 16))).tolist()
    )
    print(para_character_set[0:10])

    unicode_character_categories = {
        "punctuation": punctuation_character_set,
        "punctuation_start": punctuation_start_character_set,
        "punctuation_end": punctuation_end_character_set,
        "symbol": symbol_character_set,
        "symbol_currency": symbol_currency_character_set,
        "symbol_math": symbol_math_character_set,
        "whitespace": blankspace_character_set,
        "space": space_character_set,
        "line": line_character_set,
    }

    with open(fn_char_categories, "wb") as output_file:
        pickle.dump(unicode_character_categories, output_file)

    # Reverse mapping
    unicode_character_category_mapping = {}

    for _, row in df.iterrows():
        unicode_character_category_mapping[chr(int(row["value"], 16))] = row["general_category"]

    with open(fn_char_category_mapping, "wb") as output_file:
        pickle.dump(unicode_character_category_mapping, output_file)

    # Reverse mapping
    unicode_character_top_category_mapping = {}

    for _, row in df.iterrows():
        unicode_character_top_category_mapping[chr(int(row["value"], 16))] = row["general_category"][0].upper()

    with open(fn_char_top_category_mapping, "wb") as output_file:
        pickle.dump(unicode_character_top_category_mapping, output_file)


if __name__ == "__main__":
    print("Building and saving unicode lookup tables...")
    build_lookup_tables(
        _FN_UNICODE_CHAR_CATEGORIES, _FN_UNICODE_CHAR_CATEGORY_MAPPING, _FN_UNICODE_CHAR_TOP_CATEGORY_MAPPING
    )

    print("Done")
