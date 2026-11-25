"""Measuring what the tokeniser discards"""

from collections import Counter

from spam_filter.data import load_sms_spam
from spam_filter.tokenise import tokenise

texts, labels = load_sms_spam()
spam = [t for t, y in zip(texts, labels) if y == 1]
ham = [t for t, y in zip(texts, labels) if y == 0]

print("SIGNAL LOST TO PUNCTUATION STRIPPING")
for ch in "!£$*":
    s = sum(ch in t for t in spam) / len(spam)
    h = sum(ch in t for t in ham) / len(ham)
    print(f"  {ch!r}: {s:6.1%} of spam, {h:6.1%} of ham")

caps_spam = sum(any(w.isupper() and len(w) > 2 for w in t.split()) for t in spam) / len(spam)
caps_ham = sum(any(w.isupper() and len(w) > 2 for w in t.split()) for t in ham) / len(ham)
print(f"\nSIGNAL LOST TO LOWERCASING\n  has an ALL-CAPS word: {caps_spam:.1%} of spam, {caps_ham:.1%} of ham")

print("\nNON-ASCII CHARACTERS (survive string.punctuation)")
print(" ", Counter(c for t in texts for c in t if ord(c) > 127).most_common(8))

tokens = [tokenise(t) for t in texts]
vocab = {w for toks in tokens for w in toks}
print("\nVOCABULARY")
print(f"  types: {len(vocab):,}")
print(f"  messages reduced to zero tokens: {sum(not toks for toks in tokens)}")
print(f"  purely numeric types: {sum(w.isdigit() for w in vocab):,}")


