# 💬 Sentiment Classification - Product Reviews

This project performs sentiment analysis on food product reviews from the **Amazon Fine Food Reviews** dataset.  
It classifies each review as **positive**, **neutral**, or **negative** using NLP preprocessing and a supervised machine learning model.

---

## Goal

The goal is to:
- Clean and preprocess raw text reviews
- Transform text into numerical features using TF-IDF
- Train a **Logistic Regression classifier**
- Evaluate the model with performance metrics and visualization
- Test the model in an **interactive mode** where a user can input a custom English sentence

---

## How it works

### Steps in the script:

1. **Preprocessing**  
   - Lowercasing, punctuation & digit removal  
   - Tokenization  
   - Lemmatization  
   - Stopwords filtering

2. **TF-IDF Vectorization**  
   - Limiting to `max_features=5000` or `10000` (depending on test phase)  
   - Using unigrams and bigrams (`ngram_range=(1,2)`)

3. **Train/Test Split**  
   - Stratified 80/20 split

4. **Model**  
   - `LogisticRegression(max_iter=200)`  
   - Improved version uses `class_weight='balanced'` and `max_iter=1000`

5. **Evaluation**  
   - Classification report  
   - Confusion matrix (displayed with seaborn)  
   - F1 macro score via `cross_val_score`

6. **User Input Test (at end)**  
   - Asks the user to type an English sentence  
   - Outputs the predicted sentiment  
   - Can quit typing `exit` or `quit`

---

## Example Results

Classification report before tuning:

| Class     | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| Negative  | 0.72      | 0.65   | 0.68     |
| Neutral   | 0.44      | 0.14   | 0.21     |
| Positive  | 0.89      | 0.97   | 0.93     |

After tuning (`class_weight='balanced'`, better vectorizer):

| Class     | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| Negative  | 0.65      | 0.72   | 0.68     |
| Neutral   | 0.32      | 0.33   | 0.32     |
| Positive  | 0.93      | 0.91   | 0.92     |

**Macro F1 score (cross-validation)**: `0.6321`

Author
Émeline Medan
GitHub: @emelinemedan
