#!/usr/bin/env python3
"""
embedding.py

Train a Skip-gram Word2Vec model (gensim) on tokenized news and vectorize the news rows.

Outputs:
 - CSV with schema: date,symbol,news_vector,impact_score  (news_vector is JSON list of floats)
 - Numpy .npz file with arrays 'X' and 'y' for convenience

Usage example:
    python embedding.py \
      --in aggregate_news_with_impact.csv \
      --out vectors_skipgram.csv \
      --npz vectors_skipgram.npz \
      --vector-size 100 --window 5 --min-count 5 --workers 4 --epochs 5 --evaluate

Key parameters you should describe in your report:
 - vector_size: embedding dimensionality (num floats per token vector).
 - window: context window size (how many surrounding words to predict).
 - min_count: ignore words with total frequency lower than this.
 - sg=1 in Word2Vec -> skip-gram; sg=0 would be CBOW.
"""
import argparse
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def preprocess(text):
    # simple_preprocess lowercases, tokenizes, removes punctuation; remove STOPWORDS
    tokens = [t for t in simple_preprocess(text) if t not in STOPWORDS]
    return tokens

def vectorize(tokens, model, vector_size):
    # Use model.wv.key_to_index (gensim 4.x) to check membership
    vectors = [model.wv[word] for word in tokens if word in model.wv.key_to_index]
    if len(vectors) == 0:
        return np.zeros(vector_size, dtype=float)
    return np.mean(vectors, axis=0)

def run():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', default='./datasets/aggregate_news_with_impact.csv', help='Input CSV with columns date,symbol,news,impact_score')
    p.add_argument('--out', default='./datasets/vectorized_news_skip-gram_embeddings.csv', help='Output CSV')
    p.add_argument('--npz', default='./datasets/vectorized_news_skip-gram_embeddings.npz', help='Output .npz file with X and y arrays')
    p.add_argument('--vector-size', type=int, default=100, help='Word2Vec vector_dim (vector_size)')
    p.add_argument('--window', type=int, default=5, help='Word2Vec window size')
    p.add_argument('--min-count', type=int, default=5, help='Word2Vec min_count')
    p.add_argument('--workers', type=int, default=4, help='Word2Vec workers')
    p.add_argument('--epochs', type=int, default=5, help='Training epochs')
    p.add_argument('--evaluate', action='store_true', help='Run simple logistic regression evaluation on impact_score')
    args = p.parse_args()

    print("Loading input CSV...")
    df = pd.read_csv(args.infile, dtype=str)
    # Ensure expected columns exist
    for col in ['date', 'symbol', 'news']:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain '{col}' column")

    # Preprocess & tokenization
    print("Tokenizing and removing stopwords...")
    df['tokens'] = df['news'].fillna('').astype(str).apply(preprocess)

    # Optionally drop empty documents for training Word2Vec, but keep them for vectorization (will become zero vector)
    tokenized_docs = [t for t in df['tokens'] if len(t) > 0]
    print("Number of documents with tokens:", len(tokenized_docs))

    # Train Word2Vec with skip-gram (sg=1)
    print("Training Word2Vec (skip-gram)...")
    w2v = Word2Vec(
        sentences=tokenized_docs,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=1,               # skip-gram
        workers=args.workers,
        epochs=args.epochs
    )

    print("Vectorizing documents (average of token vectors)...")
    X = np.vstack([vectorize(tokens, w2v, args.vector_size) for tokens in df['tokens']])
    # Try to get impact_score as integers if present, else use NaN
    if 'impact_score' in df.columns:
        y_raw = df['impact_score'].fillna('').astype(str)
        # try convert to int where possible, else set to NaN
        def conv(v):
            try:
                return int(v)
            except:
                return np.nan
        y = np.array([conv(v) for v in y_raw])
    else:
        y = np.array([np.nan]*len(df))

    # Save CSV: date,symbol,news_vector,impact_score
    out_rows = []
    print("Preparing output CSV...")
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('date,symbol,news_vector,impact_score\n')
        for i, row in df.iterrows():
            vec = X[i].tolist()
            # JSON-encode vector to put in CSV cell
            vec_str = json.dumps(vec)
            impact = df.loc[i, 'impact_score'] if 'impact_score' in df.columns else ''
            # Escape commas in symbol or date not necessary but we write simple CSV line with quoting for safety
            date = df.loc[i, 'date']
            symbol = df.loc[i, 'symbol']
            # Ensure any quotes inside vec_str are escaped by using double quotes around the field
            # We'll use simple CSV-style quoting
            line = f'"{date}","{symbol}","{vec_str}","{impact}"\n'
            f.write(line)
    print(f"Wrote vectorized CSV to {args.out}")

    # Save binary npz for quick loading later (only numeric arrays)
    print(f"Saving numeric arrays to {args.npz} (X shape {X.shape})...")
    np.savez_compressed(args.npz, X=X, y=y)
    print("Saved .npz file.")

    # Optional evaluation: train/test split + logistic regression
    if args.evaluate:
        # Only keep rows where y is not NaN
        mask = ~np.isnan(y)
        X_eval = X[mask]
        y_eval = y[mask].astype(int)
        if len(y_eval) == 0:
            print("No labelled data available for evaluation (impact_score missing).")
            return
        print("Running quick evaluation (LogisticRegression) to mimic professor's sample script...")
        X_train, X_test, y_train, y_test = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval if len(set(y_eval))>1 else None)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification report:")
        print(classification_report(y_test, y_pred))

    print("Done.")

if __name__ == '__main__':
    run()
