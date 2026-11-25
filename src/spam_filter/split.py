"""Stratified train/test split with a fixed seed."""

from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.25


def train_test(texts, labels, test_size=TEST_SIZE, seed=SEED):
    """Stratified split.an unstratified
    split can shift the minority rate by a percentage point or more, which moves
    the prior and makes results non-comparable between runs.
    """
    return train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

