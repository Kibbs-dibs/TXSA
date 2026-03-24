import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import time

# Load data
# Sentiment140: labels are 0 (negative) and 4 (positive) — remap to 0/1
df = pd.read_csv("./sentiment140_cleaned.csv")

if df["target"].max() == 4:
    df["target"] = df["target"].replace({0: 0, 4: 1})

X = df["final_text"].astype(str)
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorisation
vectorizer = TfidfVectorizer(
    max_features=None,  # no max limit
    ngram_range=(1, 3),  # unigrams + bigrams + trigrams
    sublinear_tf=True,  # 1 + log(tf) instead of raw count
    min_df=2,  # ignore terms appearing in fewer than 3 docs
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{2,}",  # tokens of at least 2 chars
)

t0 = time.time()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(
    f"Vectorisation done in {time.time() - t0:.1f}s — vocab size: {len(vectorizer.vocabulary_)}"
)

# Train LinearSVC
# clf = LinearSVC(C=0.5, max_iter=1000, dual="auto")
# clf.fit(X_train_tfidf, y_train)

# Train LinearSVC
# Alternative: do a grid search to find best hyperparameters
param_dist = {
    "C": uniform(0.01, 2.0),  # samples C uniformly between 0.01 and 2.01
}

clf = RandomizedSearchCV(
    LinearSVC(max_iter=1000, dual="auto"),
    param_distributions=param_dist,
    n_iter=10,  # number of random combinations to try
    cv=3,  # 3-fold cross validation
    scoring="accuracy",
    verbose=2,
    n_jobs=-1,  # use all cores
)

t0 = time.time()
clf.fit(X_train_tfidf, y_train)
print(f"Grid search done in {time.time() - t0:.1f}s")
print(clf.best_params_)
print(clf.best_score_)

# Evaluate
y_pred = clf.predict(X_test_tfidf)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Inference example
samples = [
    "I absolutely love this, it made my day!",
    "Worst experience ever, completely disappointed.",
    "The package arrived on time.",
]

sample_tfidf = vectorizer.transform(samples)
predictions = clf.predict(sample_tfidf)
labels = {0: "Negative", 1: "Positive"}

print("\nSample predictions:")
for text, pred in zip(samples, predictions):
    print(f"  [{labels[pred]}] {text}")
