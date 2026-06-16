# -*- coding: utf-8 -*-
import zipfile, os, glob #zipfile used to extract the files, os &glob used
#navigate and read multiple essay files
import pandas as pd #to organise essays into a DataFramefor analysis
import regex as re  #for regex based cleaning that is punctuations, digits, etc
import nltk as nl #for tokenization, stopword removal, lemmatization
from textblob import TextBlob #for basic sentiment and noun phrase extraction
from langdetect import detect #to detect and filter non-English or irrelevant entries

from sklearn.feature_extraction.text import TfidfVectorizer #for TF-IDF vectorization and cosine similarity
from sklearn. metrics.pairwise import cosine_similarity #for TF-IDF vectorization and cosine similarity
from sentence_transformers import SentenceTransformer #for semantic similarity using BERT embeddings
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from difflib import SequenceMatcher
from collections import Counter

import numpy as np #Handling arrays, matrices, fast operations on large datasets, and numerical computation.
import scipy as sc #Advanced mathematical functions, scientific algorithm, etc
import matplotlib as mp #Data visualisation and plotting


# nltk downloads (run once)
nl.download('punkt')
nl.download('stopwords')
nl.download('wordnet')
nl.download('omw-1.4')
nl.download('averaged_perceptron_tagger_eng')


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

STOPWORDS = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()

#Loading the Essay files and preprocessing tasks.

#Creating a function that extracts each file from the zipped folder and saves them to a dataframe.
def read_zip_to_dataframe(zip_path, pattern="*.txt", encoding='utf-8'):
    """
    Extracts text files from a zip into a DataFrame with columns:
    filename, raw_text
    """
    rows = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if glob.fnmatch.fnmatch(name, pattern):
                with z.open(name) as f:
                    try:
                        raw = f.read().decode(encoding, errors='replace')
                    except Exception:
                        raw = f.read().decode('latin-1', errors='replace')
                rows.append({'filename': name, 'raw_text': raw})
    return pd.DataFrame(rows)

#Function for basic cleaning: lower, remove digits/punctuation, collapse whitespace. Keep this deterministic for exact-duplicate detection.
def simple_clean(text):
 
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r'\d+', ' ', t)                      # remove digits
    t = re.sub(r'[^\p{L}\s]', ' ', t)               # remove punctuation (unicode letters kept)
    t = re.sub(r'\s+', ' ', t).strip()              # normalize whitespace
    return t

#Function for Tokenize, remove stopwords, lemmatize. Returns list of tokens and also a joined string.
def tokenize_and_lemmatize(text):
   
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]           # keep words only
    tokens = [t for t in tokens if t not in STOPWORDS]
    # POS tagging -> map to WordNet POS for lemmatizer
    pos_tags = pos_tag(tokens)
    def wn_pos(tag):
        if tag.startswith('J'): return 'a'
        if tag.startswith('V'): return 'v'
        if tag.startswith('N'): return 'n'
        if tag.startswith('R'): return 'r'
        return 'n'
    lemmas = [LEMMA.lemmatize(tok, wn_pos(pt)) for tok, pt in pos_tags]
    return lemmas, " ".join(lemmas)

def add_common_columns(df):
    """
    Adds columns used by multiple tasks:
      - cleaned: simple_clean(raw_text)
      - lang: language detection (try/except)
      - tokens, lemma_text
      - word_count, unique_token_count
    """
    df = df.copy()
    df['cleaned'] = df['raw_text'].fillna('').astype(str).apply(simple_clean)
    # language detection (wrap in try)
    def safe_lang(s):
        try:
            return detect(s)
        except Exception:
            return 'unknown'
    df['lang'] = df['raw_text'].fillna('').astype(str).apply(lambda s: safe_lang(s[:10000]) if len(s.strip())>0 else 'unknown')
    tokenized = df['cleaned'].apply(tokenize_and_lemmatize)
    df['tokens'] = tokenized.apply(lambda x: x[0])
    df['lemma_text'] = tokenized.apply(lambda x: x[1])
    df['word_count'] = df['tokens'].apply(len)
    df['unique_token_count'] = df['tokens'].apply(lambda t: len(set(t)))
    return df

#Performing the data reading and preprocessing steps from the defined functions abogve to the essay zipped folder.

if __name__ == "__main__":
    zip_path = "EssaysToTextAnalysis.zip"
    try:
        raw_essay_collection = read_zip_to_dataframe(zip_path)
    except FileNotFoundError as e:
        print(e)
        raise

    print(f"Read {len(raw_essay_collection)} files from {zip_path}")
    df = add_common_columns(raw_essay_collection)

    # quick checks
    print("\nColumns in df:", df.columns.tolist())
    print("\nSample rows:")
    print(df[['filename', 'lang', 'word_count']].head(10).to_string(index=False))

    # save processed data for downstream tasks
    df.to_pickle("essays_preprocessed.pkl")
    print("\nSaved essays_preprocessed.pkl")
    
    
#Task 1_a: Finding exact duplicate entry
def find_exact_duplicates(df):
    """
    Returns a DataFrame of groups where identical cleaned text appear >1 time.
    """
    grouped = df.groupby('cleaned').filter(lambda g: len(g) > 1)
    if grouped.empty:
        return pd.DataFrame(columns=['cleaned','filenames'])
    # aggregate filenames for each identical cleaned text
    out = grouped.groupby('cleaned')['filename'].apply(list).reset_index().rename(columns={'filename':'filenames'})
    return out


exact_dups = find_exact_duplicates(df)
print(exact_dups)

#Task 1_b: Near/Empty Text
def find_near_empty(df, word_threshold=30, unique_threshold=8):
    """
    Flags essays where word_count < word_threshold OR unique tokens < unique_threshold.
    Returns DataFrame with filename, word_count, unique_token_count
    """
    mask = (df['word_count'] < word_threshold) | (df['unique_token_count'] < unique_threshold)
    return df.loc[mask, ['filename', 'word_count', 'unique_token_count', 'raw_text']]


empties = find_near_empty(df)
print(empties)

#Task 1_c: Irrelevant Entry

def find_non_english(df):
    """Mark entries not detected as English."""
    return df.loc[~df['lang'].isin(['en']) , ['filename', 'lang', 'raw_text']]

def find_off_topic_entries(df, min_avg_sim=None):
    """
    Use TF-IDF + cosine similarity:
      - Compute TF-IDF vectors for all lemma_text
      - Compute average cosine similarity for each document
      - Flag as off-topic if avg similarity < threshold (auto or manual)
    Returns DataFrame with filename and avg_similarity
    """
    vect = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2), stop_words='english')
    X = vect.fit_transform(df['lemma_text'].fillna(''))
    sims = cosine_similarity(X)
    np.fill_diagonal(sims, 0.0)
    avg_sims = sims.mean(axis=1)

    # Automatically set threshold if not provided
    if min_avg_sim is None:
        min_avg_sim = np.median(avg_sims) * 0.8  # adaptive threshold slightly below median

    out = pd.DataFrame({'filename': df['filename'], 'avg_similarity': avg_sims})
    off_topic = out.loc[out['avg_similarity'] < min_avg_sim].sort_values('avg_similarity')
    return off_topic


# Combine non-English and off-topic checks
non_english = find_non_english(df)
off_topic = find_off_topic_entries(df, min_avg_sim=0.03)  # or leave None for auto-threshold
# Review both to mark irrelevant entries.

#Task 1_d: Near Duplicate 1
def find_near_duplicates_tfidf(df, threshold=0.90):
    """
    Compute TF-IDF vectors and find pairs with cosine similarity >= threshold.
    Returns list of tuples: (i_idx, j_idx, filename_i, filename_j, similarity)
    """
    vect = TfidfVectorizer(max_df=0.95, min_df=1, ngram_range=(1,3))
    X = vect.fit_transform(df['lemma_text'].fillna(''))
    sims = cosine_similarity(X)
    pairs = []
    n = sims.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            s = sims[i,j]
            if s >= threshold:
                pairs.append((i, j, df.at[i,'filename'], df.at[j,'filename'], float(s)))
    pairs = sorted(pairs, key=lambda x: -x[4])  # sort by similarity desc
    return pairs

near_dup_tfidf = find_near_duplicates_tfidf(df, threshold=0.88)
for p in near_dup_tfidf: print(p)

#Task 1_e: Near Duplicate 2
def find_near_duplicates_semantic(df, model_name='all-MiniLM-L6-v2', threshold=0.85):
    """
    Compute sentence-transformer embeddings for lemma_text and find pairs with cosine similarity >= threshold.
    Returns list of pairs (i, j, filename_i, filename_j, sim)
    """
    model = SentenceTransformer(model_name)  # instantiate once
    texts = df['lemma_text'].fillna('').tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    sims = cosine_similarity(embeddings)
    pairs = []
    n = sims.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            s = sims[i,j]
            if s >= threshold:
                pairs.append((i, j, df.at[i,'filename'], df.at[j,'filename'], float(s)))
    pairs = sorted(pairs, key=lambda x: -x[4])
    return pairs

near_dup_semantic = find_near_duplicates_semantic(df, threshold=0.82)
for p in near_dup_semantic: print(p)

#Task 1_f: Plagiarised text 1
from itertools import islice

def get_ngrams(tokens, n=8):
    """Return list of n-gram tuples from token list."""
    return list(zip(*(tokens[i:] for i in range(n))))

def ngram_overlap_ratio(tokens_a, tokens_b, n=8):
    """Compute ratio of shared n-grams (Jaccard-ish): |A∩B| / min(|A|,|B|) to emphasize copying."""
    a = set(get_ngrams(tokens_a, n))
    b = set(get_ngrams(tokens_b, n))
    if not a or not b:
        return 0.0
    inter = a.intersection(b)
    # Ratio relative to the smaller number of n-grams (detects copying from shorter source)
    denom = min(len(a), len(b))
    return len(inter) / denom

def find_ngram_plagiarism(df, n=8, threshold=0.5):
    """
    For each pair compute n-gram overlap ratio and return pairs above threshold.
    """
    pairs = []
    toks = df['tokens'].tolist()
    n_docs = len(toks)
    for i in range(n_docs):
        for j in range(i+1, n_docs):
            ratio = ngram_overlap_ratio(toks[i], toks[j], n=n)
            if ratio >= threshold:
                pairs.append((i, j, df.at[i,'filename'], df.at[j,'filename'], ratio))
    return sorted(pairs, key=lambda x: -x[4])

plag1 = find_ngram_plagiarism(df, n=8, threshold=0.4)
for p in plag1: print(p)

#Task 1_g: Plagiqarised Text 2
def longest_common_substring_ratio(a, b):
    """
    Returns ratio = length_longest_common_substring / min(len(a), len(b))
    Uses difflib.SequenceMatcher as an approximation.
    """
    if not a or not b:
        return 0.0
    s = SequenceMatcher(None, a, b)
    match = s.find_longest_match(0, len(a), 0, len(b))
    lcs_len = match.size
    denom = min(len(a), len(b))
    return lcs_len / denom if denom > 0 else 0.0, lcs_len

def find_long_substring_plagiarism(df, char_threshold_ratio=0.20, min_chars_common=200):
    """
    Flag pairs where a long common substring exists.
    char_threshold_ratio = fraction of min-length consumed by the LCS to consider plagiarism
    """
    pairs = []
    texts = df['cleaned'].tolist()
    n = len(texts)
    for i in range(n):
        for j in range(i+1, n):
            ratio, lcs_len = longest_common_substring_ratio(texts[i], texts[j])
            if ratio >= char_threshold_ratio and lcs_len >= min_chars_common:
                pairs.append((i, j, df.at[i,'filename'], df.at[j,'filename'], ratio, lcs_len))
    return sorted(pairs, key=lambda x: -x[4])

# Sentence-level paraphrase detector using sentence-transformers
def find_sentence_level_plagiarism(df, model_name='all-MiniLM-L6-v2', sent_sim_threshold=0.9, proportion_thresh=0.2):
    """
    Split docs into sentences. For each pair, compute fraction of sentences in doc B that have
    a very similar sentence in doc A (sent_sim_threshold). If fraction >= proportion_thresh, flag.
    Returns list: (i, j, filename_i, filename_j, fraction_similar)
    """
    model = SentenceTransformer(model_name)
    # Pre-split sentences using nltk (work with raw_text to preserve sentence boundaries)
    df_sentences = df['raw_text'].fillna('').apply(nl.tokenize.sent_tokenize).tolist()
    n = len(df_sentences)
    pairs = []
    # build sentence embeddings for each doc's sentences lazily
    embeddings_cache = {}
    for i in range(n):
        sents_i = df_sentences[i]
        if not sents_i:
            embeddings_cache[i] = np.array([])
        else:
            embeddings_cache[i] = model.encode(sents_i, convert_to_numpy=True)
    for i in range(n):
        for j in range(n):
            if i == j: 
                continue
            emb_i = embeddings_cache[i]
            emb_j = embeddings_cache[j]
            if emb_i.size == 0 or emb_j.size == 0:
                continue
            sims = cosine_similarity(emb_j, emb_i)  # shape (len(sents_j), len(sents_i))
            # For each sentence in j, find max sim to any sentence in i
            max_per_sent_j = sims.max(axis=1)
            fraction = (max_per_sent_j >= sent_sim_threshold).mean()
            if fraction >= proportion_thresh:
                pairs.append((i, j, df.at[i,'filename'], df.at[j,'filename'], float(fraction)))
    return sorted(pairs, key=lambda x: -x[4])



long_subs = find_long_substring_plagiarism(df, 0.2, 200)
sent_plag = find_sentence_level_plagiarism(df, sent_sim_threshold=0.92, proportion_thresh=0.15)

#Task 2: Converting all existing ratings to 0-5 scale.
df = pd.read_csv("hotel_reviews.csv")

print(df.columns)
print(df.info())
print(df['reviews.rating'].value_counts(dropna=False))

# Example: if ratings are 1–10, divide by 2
df['rating_scaled'] = df['reviews.rating'] / 2

# Drop missing ratings for training
df_model = df.dropna(subset=['reviews.rating']).copy()


# combine title + text if you want
df_model['cleaned_text'] = (df_model['reviews.title'].fillna('') + " " + 
                            df_model['reviews.text'].fillna('')).apply(simple_clean)


#Task 2_b:
from sklearn.model_selection import train_test_split
X = df_model['cleaned_text']
y = df_model['rating_scaled']  # numeric 0–5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,2))),
    ('reg', LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


df_missing = df[df['reviews.rating'].isna()].copy()
df_missing['cleaned_text'] = (df_missing['reviews.title'].fillna('') + " " + 
                              df_missing['reviews.text'].fillna('')).apply(simple_clean)


# Use regression or classifier depending on your best model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# round ratings to nearest integer
y_train_cls = y_train.round().astype(int)
y_test_cls = y_test.round().astype(int)

model_cls = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000))
])

model_cls.fit(X_train, y_train_cls)
y_pred_cls = model_cls.predict(X_test)

print(classification_report(y_test_cls, y_pred_cls))

#Task 2_c
df_missing['predicted_rating'] = model_cls.predict(df_missing['cleaned_text'])
import matplotlib.pyplot as plt
plt.hist(df_missing['predicted_rating'], bins=6)
plt.title("Predicted ratings for missing reviews")
plt.xlabel("Predicted rating (0–5)")
plt.ylabel("Count")
plt.show()

print("Mean predicted rating:", df_missing['predicted_rating'].mean())

#Task 2_d
# --- Categorize ratings into 'poor', 'ok', 'excellent' ---

def categorize(r):
    if r <= 1:
        return 'poor'
    elif 2 <= r <= 3:
        return 'ok'
    else:
        return 'excellent'

df_model['category'] = df_model['rating_scaled'].apply(categorize)

# Encode categories
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_cat = le.fit_transform(df_model['category'])

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df_model['cleaned_text'], y_cat, test_size=0.2, random_state=42
)

# Build and train the model
from sklearn.pipeline import Pipeline

model_cat = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

model_cat.fit(X_train, y_train)
y_pred_cat = model_cat.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred_cat, target_names=le.classes_))

# ---- CONTINUATION: Compute mean predicted value + plot histogram ----

import matplotlib.pyplot as plt


# Convert predicted encoded labels back to category names
pred_labels = le.inverse_transform(y_pred_cat)

# Map categories to numeric scale for averaging
category_to_score = {'poor': 1, 'ok': 3, 'excellent': 5}
df_pred_cat = pd.DataFrame({'predicted_category': pred_labels})
df_pred_cat['numeric_value'] = df_pred_cat['predicted_category'].map(category_to_score)

# Compute mean predicted numeric value
mean_category_pred = df_pred_cat['numeric_value'].mean()
print(f"\nMean predicted numeric value (based on categories): {mean_category_pred:.2f}")

# Plot histogram of predicted categories
plt.figure(figsize=(6,4))
df_pred_cat['predicted_category'].value_counts().sort_index().plot(
    kind='bar', color='lightgreen', edgecolor='black'
)
plt.title("Predicted Rating Categories (Poor / OK / Excellent)")
plt.xlabel("Predicted Category")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.show()


