# multinomial vs bernoulli - multi counts every instance, whereas bernoulli counts presence once per message

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from spam_filter.tokenise import tokenise


@dataclass(frozen=True)
class ClassCounts:
    """Sufficient statistics for multinomial Naive Bayes, from training data only."""

    vocabulary: tuple[str, ...]        # sorted; |V| is the Laplace denominator term
    doc_counts: dict[int, int]         # messages per class -> priors
    word_counts: dict[int, Counter]    # per class: token -> occurrences
    total_tokens: dict[int, int]       # per class: N_c, the denominator

    @property
    def V(self) -> int:
        return len(self.vocabulary)


def fit_counts(texts, labels) -> ClassCounts:
    doc_counts: Counter[int] = Counter()
    word_counts: dict[int, Counter] = {}

    for text, label in zip(texts, labels, strict=True):
        doc_counts[label] += 1
        word_counts.setdefault(label, Counter()).update(tokenise(text))

    vocabulary = tuple(sorted({w for c in word_counts.values() for w in c}))

    return ClassCounts(
        vocabulary=vocabulary,
        doc_counts=dict(doc_counts),
        word_counts=word_counts,
        total_tokens={c: sum(ctr.values()) for c, ctr in word_counts.items()},
    )