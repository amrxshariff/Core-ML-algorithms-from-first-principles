import string

_PUNCT = str.maketrans("", "", string.punctuation)


def tokenise(text: str) -> list[str]:
    """Lowercase, delete ASCII punctuation, split on whitespace.

    Deliberately minimal: every step is reversible by hand, which is the
    point of this project. The costs, measured on the training corpus:

    - Lowercasing merges "FREE" and "free", pooling evidence for a strong
      spam term. But an ALL-CAPS word appears in 74.0% of spam against
      7.1% of ham — a 10:1 signal, destroyed outright by this step.
    - Deleting punctuation merges "don't" -> "dont", keeping contractions
      whole rather than splitting into "don" and "t". It also deletes "!",
      present in 49.0% of spam against 11.6% of ham. Discarded, not neutral.
    - string.punctuation is ASCII-only, so "£" survives. This matters: "£"
      appears in 33.9% of spam against 0.1% of ham, the sharpest single
      signal in the corpus. It survives by accident of the stdlib rather
      than by design — and splitting on whitespace then shatters it across
      75 distinct types ("a£150", "3x£150pw", "cost£150pm")
    - Non-ASCII punctuation also survives, including 29 messages carrying a
      mis-encoded apostrophe (U+0092). "don't" tokenises to "dont" but
      "don\x92t" tokenises to "don\x92t" — the same word, two types.
    - No stemming, no stopword list, no n-grams. Each would help, and each
      would make the hand-worked example in the README impossible to follow.

    Vocabulary: 9,661 types, of which 557 are purely numeric.
    """
    return text.lower().translate(_PUNCT).split()

# whilst stripping punctuation loses information, tokeniser did as it was supposed to and still lost signal.
# Signal being any property of a raw message that shifts the odds of ham against spam
# FREE and free map to free (many-to-one relationship) and so FREE is permanently lost which is not good
# because 74% of spam messages have all caps in them. Many-to-one is known as Collapsing.
# Deletion is a similar concept but applies to a character being present, no longer being present in the output
# Fragmenetation is where for example the £ survives and every instance of it is present in the output, but estimator cannot use it
# collapsing and deleteion destroy information, whereas fragmentation preserves it but starves the estimator
# 


