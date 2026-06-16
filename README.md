# Hotel Review Rating Prediction & Essay Text Analytics

> *An NLP project in Python: detecting suspicious essay submissions and predicting hotel review ratings from raw review text.*

---

## Project Overview

This project applies **Python text analytics (NLP)** to two distinct tasks:

1. **Essay integrity analysis** — screening a collection of ~50 essays on the "Data Scientist Role" to flag suspicious entries: exact duplicates, (near-)empty text, irrelevant/off-topic submissions, near-duplicates, and plagiarized passages.

2. **Hotel review rating prediction** — building a model that predicts a 0–5 star rating directly from the free text of a hotel review (~36,000 reviews), then using it to impute ratings for the 862 reviews that have text but no rating.

The work was completed as a university text-analytics assignment using the NLTK ecosystem supported by scikit-learn, sentence-transformers, and the standard scientific Python stack.

---

## Assignment Background

This project fulfils the **Python Text Analysis** assignment, which specifies two parts:

**Part 1 — Essay screening.** Read and pre-process ~50 essays from `EssaysToTextAnalysis.zip` and use word count, similarity, and methods of choice to identify: (a) an exact duplicate, (b) near-empty text, (c) an irrelevant entry, (d–e) two near-duplicates, and (f–g) two plagiarized texts.

**Part 2 — Rating prediction.** Using `hotel_reviews.csv` (~36,000 rows, 862 without a rating), pre-process the data, convert all ratings to a 0–5 scale, and:
- (a) build a model that suggests a rating from review text (`reviews.text`, optionally `reviews.title`), leaving out the 862 unrated rows;
- (b) split into train/test and report accuracy, recall, and precision for the best model;
- (c) predict ratings for the 862 unrated rows, plot a histogram, and report the mean;
- (d) collapse the scale into categories — *poor* (0–1), *ok* (2–3), *excellent* (4–5) — re-run, and discuss the practical effect of categorization.

---

## Methodology

### Part 1 — Essay Text Analytics

**Pre-processing.** Each essay is extracted from the zip into a pandas DataFrame, then cleaned (lowercasing, digit/punctuation removal, whitespace normalization), language-detected, tokenized, stop-word-filtered, and lemmatized with POS-aware WordNet lemmatization.

**Detection methods**, one per suspicious-entry type:

| Task | Suspicious entry | Method |
|------|------------------|--------|
| 1a | Exact duplicate | Group by cleaned text; flag identical groups |
| 1b | Near-empty | Threshold on word count and unique-token count |
| 1c | Irrelevant / off-topic | Language detection + low average TF-IDF cosine similarity to the corpus |
| 1d | Near-duplicate (lexical) | TF-IDF (1–3 grams) cosine similarity above threshold |
| 1e | Near-duplicate (semantic) | Sentence-Transformer (`all-MiniLM-L6-v2`) embedding similarity |
| 1f | Plagiarism (copying) | Shared 8-gram overlap ratio |
| 1g | Plagiarism (substring/paraphrase) | Longest-common-substring ratio + sentence-level embedding match |

### Part 2 — Rating Prediction

- **Pre-processing:** combine `reviews.title` + `reviews.text`, clean, and scale ratings to 0–5. Unrated rows (n = 862) are held out for later prediction.
- **Regression baseline:** TF-IDF → Linear Regression, evaluated with MAE and R².
- **Classification model:** TF-IDF → Logistic Regression on rounded integer ratings, evaluated with a full classification report (accuracy, precision, recall, F1).
- **Imputation:** the classifier predicts ratings for the 862 unrated reviews; a histogram and mean are produced.
- **Categorization:** ratings are collapsed into *poor / ok / excellent*, and a class-balanced Logistic Regression is re-trained and re-evaluated to assess the practical trade-off of coarser labels.

---

## Technologies Used

- **Language:** Python 3
- **NLP:** `nltk` (tokenization, stopwords, lemmatization, POS tagging), `textblob`, `langdetect`, `sentence-transformers`
- **ML / vectorization:** `scikit-learn` (TF-IDF, Linear/Logistic Regression, train/test split, metrics, pipelines)
- **Data & numerics:** `pandas`, `numpy`, `scipy`, `regex`
- **Visualization:** `matplotlib`
- **Utilities:** `zipfile`, `glob`, `difflib`, `collections`, `itertools`

---

## Repository Structure

```
hotel-review-text-analytics/
├── src/
│   └── python_text_analysis.py      # Main analysis script (both tasks)
├── data/
│   ├── EssaysToTextAnalysis.zip     # ~50 essays (Part 1 input)
│   └── hotel_reviews.zip            # hotel_reviews.csv (Part 2 input)
├── outputs/
│   ├── predicted_rating_for_missing_reviews.png
│   └── predicted_rating_categories.png
├── docs/
│   └── Assignment_PythonTextAnalytics.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** This structure is a recommendation — adjust paths to match how you arrange the files. The script currently expects the data files in the working directory, so update the paths in the script if you move them into `data/` (see Limitations).

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Tatiana-Tongwa/hotel-review-text-analytics.git
   cd hotel-review-text-analytics
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. The script downloads required NLTK corpora on first run (`punkt`, `stopwords`, `wordnet`, `omw-1.4`, `averaged_perceptron_tagger_eng`).

---

## Usage

Unzip `hotel_reviews.zip` so that `hotel_reviews.csv` is available, then run:

```bash
python src/python_text_analysis.py
```

The script reads the essay zip and the hotel CSV, runs both tasks in sequence, prints flagged entries and model metrics to the console, and displays the prediction histograms.

> **Heads-up:** the first run downloads a Sentence-Transformer model (`all-MiniLM-L6-v2`), and the essay similarity/plagiarism checks compare every pair of documents, so Part 1 takes a little time on first execution.

---

## Results and Findings

**Part 2 — Rating prediction (imputation of missing ratings).** The classifier's predictions for the 862 unrated reviews are heavily concentrated at the top of the scale, with the large majority predicted around a 2 (on the displayed axis) and a small cluster near 0 — consistent with a review set skewed toward positive sentiment.

![Predicted ratings for missing reviews](outputs/predicted_rating_for_missing_reviews.png)

**Part 2d — Categorized model (poor / ok / excellent).** After collapsing the scale into three categories and re-training with class balancing, the predicted categories distribute across *excellent*, *ok*, and *poor*, with **ok** the most common predicted label:

![Predicted rating categories](outputs/predicted_rating_categories.png)

**Practical effect of categorization.** Collapsing a 0–5 scale into three bands makes the model easier to act on (a clear poor/ok/excellent signal) and can improve robustness to small rating differences, at the cost of resolution — it can no longer distinguish, say, a 4 from a 5. For an operational dashboard flagging which reviews need attention, the coarser labels are often more usable; for fine-grained ranking, the numeric model retains more information.

> Exact accuracy/precision/recall figures are produced at runtime via scikit-learn's `classification_report`. They are intentionally not hard-coded here so the README stays accurate if the pipeline or split changes.

---

## Limitations

- **Hard-coded file paths.** The script reads `EssaysToTextAnalysis.zip` and `hotel_reviews.csv` from the working directory. If you adopt the `data/` structure above, update the paths in the script accordingly.
- **Rating-scale conversion.** The script scales ratings by dividing by 2 (`reviews.rating / 2`), which assumes the source ratings are on a 1–10 scale. If the raw data is already 0–5, this would halve them — worth verifying against the actual `reviews.rating` distribution before trusting the numbers.
- **Pairwise comparisons are O(n²).** The essay duplicate/plagiarism checks compare all document pairs; fine for ~50 essays, but would not scale to large corpora without optimization.
- **`scipy`, `textblob`, and some imports are present but lightly used**; the core similarity work relies on TF-IDF and sentence-transformers.
- **Linear Regression on review text** is a simple baseline; predicted values can fall outside 0–5 and are not clipped.
- The script mixes a `__main__` guard (Part 1 setup) with top-level execution further down, so it runs as a single linear script rather than a fully modular pipeline.

---

## Future Improvements

- Parameterize file paths via command-line arguments or a small config block instead of hard-coding them.
- Convert the script into a Jupyter notebook (or numbered modules) so each task's output is visible inline — well suited to a portfolio reviewer skimming on GitHub.
- Verify the true rating scale and replace the divide-by-2 assumption with an explicit, validated mapping.
- Try stronger models for Part 2 (e.g. linear SVM, gradient boosting, or a fine-tuned transformer) and report a confusion matrix.
- Cache Sentence-Transformer embeddings and use approximate nearest-neighbor search to make the essay checks scale.
- Save flagged-essay results and metrics to CSV/JSON for reproducibility.

---

## Contributors

- **Tatiana Nanette Tongwa**

---

## License

MIT License
