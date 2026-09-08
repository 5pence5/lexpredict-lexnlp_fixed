"""Coverage tests for NltkTokenizer.convert_parentheses branch."""

from lexnlp.extract.en.entities.nltk_tokenizer import NltkTokenizer


def test_convert_parentheses_replaces_brackets():
    tokenizer = NltkTokenizer()
    assert tokenizer.tokenize("hello (world)", convert_parentheses=True) == ["hello", "-LRB-", "world", "-RRB-"]


def test_convert_parentheses_false_keeps_brackets():
    tokenizer = NltkTokenizer()
    assert tokenizer.tokenize("hello (world)") == ["hello", "(", "world", ")"]
    assert tokenizer.tokenize("hello (world)", convert_parentheses=False) == ["hello", "(", "world", ")"]


def test_convert_parentheses_all_bracket_types_return_str():
    tokenizer = NltkTokenizer()
    result = tokenizer.tokenize("(test) [bracket] {brace}", convert_parentheses=True, return_str=True)
    assert isinstance(result, str)
    assert "-LRB-" in result
    assert "-RRB-" in result
    assert "-LSB-" in result
    assert "-RSB-" in result
    assert "-LCB-" in result
    assert "-RCB-" in result
    assert "(" not in result
