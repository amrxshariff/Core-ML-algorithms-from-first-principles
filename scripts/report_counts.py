from spam_filter.counts import fit_counts
from spam_filter.data import load_sms_spam
from spam_filter.split import SEED, TEST_SIZE, train_test
from spam_filter.tokenise import tokenise

texts, labels = load_sms_spam()
X_train, X_test, y_train, y_test = train_test(texts, labels)
counts = fit_counts(X_train, y_train)

names = {0: "ham", 1: "spam"}
n_train = sum(counts.doc_counts.values())

print(f"SPLIT  seed={SEED}  test_size={TEST_SIZE}")
print(f"  train {len(X_train):,} messages, test {len(X_test):,}")

print("\nPRIORS (training split only)")
for c, n in sorted(counts.doc_counts.items()):
    print(f"  P({names[c]:>4}) = {n:>5,} / {n_train:,} = {n / n_train:.4f}")

print("\nVOCABULARY")
print(f"  V (train only)  = {counts.V:,}")
full_V = len({w for t in texts for w in tokenise(t)})
print(f"  V (full corpus) = {full_V:,}   <- must be strictly larger")

print("\nTOKEN TOTALS AND LAPLACE DENOMINATORS (alpha = 1)")
for c, n in sorted(counts.total_tokens.items()):
    print(f"  {names[c]:>4}: N_c = {n:>7,}   N_c + aV = {n + counts.V:>7,}")

print("\nMOST FREQUENT TOKENS")
for c in sorted(counts.word_counts):
    top = ", ".join(w for w, _ in counts.word_counts[c].most_common(8))
    print(f"  {names[c]:>4}: {top}")