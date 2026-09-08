"""Coverage tests for lexnlp.extract.common.entities.entity_banlist."""

from lexnlp.extract.common.entities.entity_banlist import BanListUsage, EntityBanListItem


class TestEntityBanListItemRepr:
    def test_repr_default_flags(self) -> None:
        item = EntityBanListItem("Company")
        assert item.pattern == "company"
        assert repr(item) == '"company" I,T'

    def test_repr_regex_only_flag(self) -> None:
        item = EntityBanListItem("x", ignore_case=False, is_regex=True, trim_phrase=False)
        assert repr(item) == '"x" Re'

    def test_repr_no_flags(self) -> None:
        item = EntityBanListItem("AbC", ignore_case=False, trim_phrase=False)
        assert item.pattern == "AbC"
        assert repr(item) == '"AbC"'

    def test_repr_all_flags(self) -> None:
        item = EntityBanListItem("Acme.*", is_regex=True)
        assert repr(item) == '"Acme.*" I,Re,T'


class TestBanListUsageRepr:
    def test_repr_default(self) -> None:
        usage = BanListUsage()
        assert repr(usage) == "0 items provided, default=Trueappend=False"

    def test_repr_with_items(self) -> None:
        usage = BanListUsage(
            banlist=[EntityBanListItem("a"), EntityBanListItem("b")],
            use_default_banlist=False,
            append_to_default=True,
        )
        assert repr(usage) == "2 items provided, default=Falseappend=True"
