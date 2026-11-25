from spam_filter.counts import fit_counts


def test_counts_occurrences_not_presence():
    c = fit_counts(["free free free"], [1])
    assert c.word_counts[1]["free"] == 3        # multinomial, not Bernoulli


def test_vocabulary_is_sorted_union_across_classes():
    c = fit_counts(["ham word", "spam word"], [0, 1])
    assert c.vocabulary == ("ham", "spam", "word")


def test_totals_reconcile():
    c = fit_counts(["a b", "b c c"], [0, 1])
    assert c.doc_counts == {0: 1, 1: 2 - 1}
    assert c.total_tokens[0] == 2
    assert c.total_tokens[1] == 3
    assert sum(c.total_tokens.values()) == 5


def test_unseen_word_absent_from_vocabulary():
    c = fit_counts(["win cash now"], [1])
    assert "lottery" not in c.vocabulary