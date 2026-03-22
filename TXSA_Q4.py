"""
bigram language model (unsmoothed + laplace smoothed)
"""

from nltk.util import bigrams
from collections import Counter


def bigram_prob_unsmoothed(sentence, unigram_counts, bigram_counts):
    """
    compute unsmoothed bigram probability
    """
    tokens = sentence.split()
    prob = 1.0

    for w1, w2 in bigrams(tokens):
        bigram_count = bigram_counts[(w1, w2)]
        unigram_count = unigram_counts[w1]

        if unigram_count == 0 or bigram_count == 0:
            return 0.0

        prob *= bigram_count / unigram_count

    return prob


def bigram_prob_laplace(sentence, unigram_counts, bigram_counts, vocab_size):
    """
    compute laplace smoothed bigram probability
    """
    tokens = sentence.split()
    prob = 1.0

    for w1, w2 in bigrams(tokens):
        bigram_count = bigram_counts[(w1, w2)]
        unigram_count = unigram_counts[w1]

        prob *= (bigram_count + 1) / (unigram_count + vocab_size)

    return prob


def main():
    # data
    corpus = [
        "<s> He read a book </s>",
        "<s> I read a different book </s>",
        "<s> He read a book by Danielle </s>",
    ]
    test_sentence = "<s> I read a different book by Danielle </s>"

    # tokenize
    tokenized = [sentence.split() for sentence in corpus]

    # count bigrams and unigrams
    unigram_counts = Counter()
    bigram_counts = Counter()

    for sentence in tokenized:
        unigram_counts.update(sentence)
        bigram_counts.update(bigrams(sentence))

    # bigram models
    unsmoothed_prob = bigram_prob_unsmoothed(
        test_sentence, unigram_counts, bigram_counts
    )
    # print results
    print(f"Unsmoothed Bigram Probability: {unsmoothed_prob:.8f}")

    # count unique words
    vocab = set()
    for sentence in tokenized:
        vocab.update(sentence)
    vocab_size = len(vocab)

    # smoothed bigram models
    smoothed_prob = bigram_prob_laplace(
        test_sentence, unigram_counts, bigram_counts, vocab_size
    )

    # print results
    print(f"Laplace Smoothed Bigram Probability: {smoothed_prob:.8f}")


if __name__ == "__main__":
    main()
