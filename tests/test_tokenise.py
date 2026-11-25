from spam_filter.tokenise import tokenise


def test_lowercases_and_splits():
    assert tokenise("FREE Entry NOW") == ["free", "entry", "now"]


def test_punctuation_deleted_not_split():
    assert tokenise("don't") == ["dont"]


def test_collapses_whitespace():
    assert tokenise("  a\t\tb\n") == ["a", "b"]


def test_empty_and_punctuation_only():
    assert tokenise("") == []
    assert tokenise("!!! ???") == []


def test_idempotent():
    msg = "WIN a £1000 prize!! Txt NOW"
    assert tokenise(" ".join(tokenise(msg))) == tokenise(msg)