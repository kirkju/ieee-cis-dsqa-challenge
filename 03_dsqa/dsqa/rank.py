"""Chunks ranked for a question: by the words they share with it, and several
rankings fused into one.

An embedder ranks by meaning and misses the exact word: a nuclide, the name
of a method, a number. Matching by word finds those and misses what is said
in other words. The two rankings fail on different questions, so they are
fused, by rank and not by score: their scores are not on one scale.
"""

import re
import unicodedata
from collections import Counter

import numpy as np
from scipy import sparse

# Reciprocal rank fusion: a chunk scores 1 / (RRF_K + its rank) in each ranking.
RRF_K = 60

WORD = re.compile(r"[^\W\d_]+|\d+(?:\.\d+)?")


def words(text: str) -> list[str]:
    """The words and numbers of a text, for matching by word: case folded,
    with letters apart from digits, so that "Be7" and "7 Be" give the same."""
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


class Bm25:
    """Okapi BM25 over a set of texts, with Lucene's defaults.

    A text scores, for each distinct word of the query it has, the rarity of
    the word times a count of its occurrences that saturates and is corrected
    for the length of the text.
    """

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        vocabulary: dict[str, int] = {}
        rows, columns, counts = [], [], []
        lengths = np.zeros(len(texts))
        for row, text in enumerate(texts):
            occurrences = Counter(words(text))
            lengths[row] = sum(occurrences.values())
            for word, count in occurrences.items():
                rows.append(row)
                columns.append(vocabulary.setdefault(word, len(vocabulary)))
                counts.append(count)
        rows, columns = np.asarray(rows, dtype=np.intp), np.asarray(columns, dtype=np.intp)
        counts = np.asarray(counts, dtype=np.float64)
        in_texts = np.bincount(columns, minlength=len(vocabulary))
        rarity = np.log(1 + (len(texts) - in_texts + 0.5) / (in_texts + 0.5))
        mean = lengths.mean() if len(texts) and lengths.any() else 1.0
        length = k1 * (1 - b + b * lengths / mean)
        weights = rarity[columns] * counts * (k1 + 1) / (counts + length[rows])
        shape = (len(texts), len(vocabulary))
        self.weights = sparse.csc_matrix((weights, (rows, columns)), shape=shape)
        self.vocabulary = vocabulary

    def scores(self, query: str) -> np.ndarray:
        known = sorted({self.vocabulary[word] for word in words(query) if word in self.vocabulary})
        if not known:
            return np.zeros(self.weights.shape[0])
        return np.asarray(self.weights[:, known].sum(axis=1)).ravel()


def best_first(scores: np.ndarray) -> np.ndarray:
    """Positions by score, highest first; equal scores keep their order."""
    return np.argsort(-scores, kind="stable")


def fuse(orders: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Several rankings of the same items as one, by reciprocal rank fusion."""
    scores = np.zeros(len(orders[0]))
    for order in orders:
        scores[order] += 1.0 / (k + np.arange(1, len(order) + 1))
    return best_first(scores)
