corpus = """Classification is the task of choosing the correct class label for a given input. In basic
classification tasks, each input is considered in isolation from all other inputs, and the set of labels is defined in advance. The basic classification task has a number of interesting variants. For example, in multiclass classification, each instance may be assigned multiple labels; in open-class classification, the set of labels is not defined in advance; and in sequence classification, a list of inputs are jointly classified."""

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(corpus)

# all tokens
all_tokens = [token.text for token in doc]
print(f"Tokens BEFORE Removal ({len(all_tokens)}):")
print(all_tokens)
 
# stop words and punctuation found
stop_words_found  = [token.text for token in doc if token.is_stop]
punctuation_found = [token.text for token in doc if token.is_punct] 
print(f"\nStop Words Found ({len(stop_words_found)}):")
print(stop_words_found)
print(f"\nPunctuation Found ({len(punctuation_found)}):")
print(punctuation_found)
 
# stop words, punctuation, and whitespace
filtered_tokens = [
    token.text for token in doc
    if not token.is_stop and not token.is_punct and not token.is_space
]
print(f"\nTokens AFTER Removal ({len(filtered_tokens)}):")
print(filtered_tokens)
