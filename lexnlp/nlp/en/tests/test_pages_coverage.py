"""Coverage tests for page-break feature-name window clamping."""

from lexnlp.nlp.en.segments.pages import get_page_break_feature_names


def test_clamps_pre_window() -> None:
    # lines_count - 1 (2) < line_window_pre (5): pre clamps to 2,
    # so offsets -2..1 are present and -5..-3 are absent.
    names = get_page_break_feature_names(3, 5, 1)
    assert "line_len_-2" in names
    assert "line_len_-1" in names
    assert "line_len_0" in names
    assert "line_len_1" in names
    assert "line_len_-3" not in names
    assert "line_len_-5" not in names
    assert "line_n_alpha_-2" in names
    assert "line_n_alpha_1" in names


def test_clamps_post_window() -> None:
    # line_window_post (10) >= lines_count (10): post clamps to -1,
    # so only offsets -2..-1 survive and 0..10 vanish.
    names = get_page_break_feature_names(10, 2, 10)
    assert "line_len_-2" in names
    assert "line_len_-1" in names
    assert "line_len_0" not in names
    assert "line_len_1" not in names
    assert "line_len_10" not in names
    assert "line_n_alpha_-2" in names
    assert "line_n_alpha_0" not in names


def test_clamps_both_windows_tiny_doc() -> None:
    # Single-line doc: both guards fire, window range is empty.
    names = get_page_break_feature_names(1, 3, 3)
    assert "page" in names
    assert "PAGE" in names
    assert "Page" in names
    assert "sw_page" in names
    assert "sw_pg" in names
    assert [key for key in names if key.startswith("line_len_")] == []
    assert [key for key in names if key.startswith("line_n_alpha_")] == []


def test_unclamped_windows_keep_full_range() -> None:
    names = get_page_break_feature_names(10, 3, 3)
    for offset in [-3, -2, -1, 0, 1, 2, 3]:
        assert f"line_len_{offset}" in names
        assert f"line_n_alpha_{offset}" in names
        assert f"line_n_number_{offset}" in names
        assert f"line_n_punct_{offset}" in names
        assert f"line_n_whitespace_{offset}" in names
    assert "line_len_-4" not in names
    assert "line_len_4" not in names
