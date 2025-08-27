import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """
    Cleans input text by:
    - converting to lowercase
    - removing links
    - keeping only letters
    - normalizing spaces
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove links
    text = re.sub(r"[^a-z\s]", " ", text)       # keep letters only
    text = re.sub(r"\s+", " ", text)            # normalize spaces
    return text.strip()


def vectorize_text(X_train, X_test=None, min_df=5, max_df=0.7, ngram_range=(1,2)):
    """
    Converts text into TF-IDF features.
    Returns: X_train_features, X_test_features (if X_test provided), vectorizer
    """
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True
    )
    X_train_features = vectorizer.fit_transform(X_train)
    if X_test is not None:
        X_test_features = vectorizer.transform(X_test)
        return X_train_features, X_test_features, vectorizer
    return X_train_features, vectorizer
