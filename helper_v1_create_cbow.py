#!/usr/bin/env python3
"""
cbow_from_scratch.py

Train a CBOW Word2Vec model from scratch (full softmax) and produce a CSV:
  date,symbol,news_vector,impact_score

Usage (example):
  python cbow_from_scratch.py \
    --in aggregate_news_with_impact.csv \
    --out vectorized_news_cbow_embeddings.csv \
    --vector-size 100 --window 4 --min-count 5 --epochs 3 --lr 0.05 \
    --max-vocab-size 20000 --max-examples 200000

Notes:
 - window: number of words to take on EACH side of the target (like gensim). Actual context size C
   is the number of available context words (<= 2*window).
 - For large corpora, training with full softmax can be VERY slow. Use --max-examples to limit
   the number of training target-context pairs (good for debugging).
 - The script uses simple tokenization via regex; will keep tokens of word characters.
"""
import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ---------- utilities ----------
WORD_RE = re.compile(r"\w+")  # simple tokenization: letters, digits, underscore

def tokenize(text):
    if not isinstance(text, str):
        return []
    return [w.lower() for w in WORD_RE.findall(text)]

def softmax_stable(x):
    # x: 1d numpy
    x_max = np.max(x)
    e = np.exp(x - x_max)
    return e / e.sum()

# ---------- main ----------
def build_vocab(docs_tokens, min_count=5, max_vocab_size=None):
    cnt = Counter()
    for tokens in docs_tokens:
        cnt.update(tokens)
    # filter by min_count
    filtered = {w: c for w, c in cnt.items() if c >= min_count}
    # sort by frequency desc
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    if max_vocab_size is not None:
        sorted_items = sorted_items[:max_vocab_size]
    vocab = {w: idx for idx, (w, _) in enumerate(sorted_items)}
    inv_vocab = [w for w, _ in sorted_items]
    freqs = np.array([c for _, c in sorted_items], dtype=np.int64)
    return vocab, inv_vocab, freqs

def prepare_training_examples(tokenized_docs, vocab, window, max_examples=None):
    examples = []  # list of (context_word_indices_list, target_index)
    for tokens in tokenized_docs:
        # map tokens to indices if in vocab
        idxs = [vocab[w] for w in tokens if w in vocab]
        n = len(idxs)
        if n == 0:
            continue
        for pos in range(n):
            target = idxs[pos]
            # context: all words within window on both sides
            left = max(0, pos - window)
            right = min(n, pos + window + 1)
            context = [idxs[i] for i in range(left, right) if i != pos]
            if len(context) == 0:
                continue
            examples.append((context, target))
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    return examples

def train_cbow(examples, vocab_size, vector_size, epochs, lr, print_every=100000):
    # Initialize input (V) and output (U) embeddings
    rng = np.random.RandomState(42)
    V = (rng.randn(vocab_size, vector_size) * 0.01).astype(np.float32)  # input embeddings
    U = (rng.randn(vocab_size, vector_size) * 0.01).astype(np.float32)  # output embeddings

    step = 0
    for epoch in range(1, epochs + 1):
        random.shuffle(examples)
        total_loss = 0.0
        for (context_idxs, target_idx) in examples:
            step += 1
            C = len(context_idxs)
            # compute hidden vector h = mean of input embeddings for context
            h = np.mean(V[context_idxs], axis=0)  # shape (vector_size,)

            # compute logits = U @ h  -> shape (vocab_size,)
            logits = U.dot(h)  # (V,)
            y_hat = softmax_stable(logits)  # (V,)

            # accumulate loss: -log P(target)
            loss = -math.log(max(1e-12, y_hat[target_idx]))
            total_loss += loss

            # error vector: y_hat - y (where y is one-hot for target)
            err = y_hat
            err[target_idx] -= 1.0  # shape (vocab_size,)

            # update output embeddings U: U_j <- U_j - lr * err_j * h
            # vectorized: U -= lr * outer(err, h)
            # but to avoid allocating giant outer for very large vocab we do vectorized row update:
            U -= (lr * err)[:, np.newaxis] * h[np.newaxis, :]

            # gradient wrt h: grad_h = U^T @ err
            grad_h = U.T.dot(err)  # shape (vector_size,)

            # update input embeddings for each context word:
            grad_each = (lr * (1.0 / C)) * grad_h  # we subtract grad_each from each context embedding
            for ci in context_idxs:
                V[ci] -= grad_each

            if (step % print_every) == 0:
                avg_loss = total_loss / print_every
                print(f"[epoch {epoch}] step {step} avg_loss (last {print_every}) = {avg_loss:.4f}")
                total_loss = 0.0

        # epoch end print
        if total_loss > 0:
            avg_loss_epoch = total_loss / max(1, (len(examples) // (epoch if epoch>0 else 1)))
        else:
            avg_loss_epoch = 0.0
        print(f"Epoch {epoch}/{epochs} finished. (examples seen: {len(examples)}).")
    return V, U

def vectorize_documents(tokens_list, V, vocab, vector_size):
    vectors = []
    for tokens in tokens_list:
        idxs = [vocab[t] for t in tokens if t in vocab]
        if len(idxs) == 0:
            vectors.append(np.zeros(vector_size, dtype=float))
        else:
            vec = np.mean(V[idxs], axis=0)
            vectors.append(vec)
    return np.vstack(vectors)

def main():
    p = argparse.ArgumentParser(description="Train CBOW from scratch and vectorize news rows.")
    p.add_argument('--in', dest='infile', default='./datasets/corpus.txt', help='Input CSV (with news column) OR corpus.txt')
    p.add_argument('--out', default='./datasets/vectorized_news_cbow_embeddings.csv', help='Output CSV path')
    p.add_argument('--vector-size', type=int, default=100, help='Dimensionality of embeddings')
    p.add_argument('--window', type=int, default=5, help='Context window size (each side). Total C <= 2*window')
    p.add_argument('--min-count', type=int, default=5, help='Minimum frequency for a word to be in vocab')
    p.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    p.add_argument('--lr', type=float, default=0.05, help='Learning rate (eta)')
    p.add_argument('--max-vocab-size', type=int, default=20000, help='Limit vocab to top-N frequent words (optional)')
    p.add_argument('--max-examples', type=int, default=None, help='Limit number of training examples (target/context pairs)')
    p.add_argument('--news-col', default='news', help='Column name containing news text (if input is CSV)')
    p.add_argument('--date-col', default='date', help='Date column name (for output CSV)')
    p.add_argument('--symbol-col', default='symbol', help='Symbol column name (for output CSV)')
    p.add_argument('--impact-col', default='impact_score', help='Impact score column')
    p.add_argument('--save-dir', default='cbow_model', help='Directory to save embeddings & vocab')
    p.add_argument('--sample-rows', type=int, default=None, help='Optional: only read first N rows from CSV (for debug)')
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # load corpus
    print("Loading input...")
    docs_text = []
    meta = []  # list of tuples (date, symbol, impact_score) if CSV
    is_csv = args.infile.lower().endswith('.csv')
    if is_csv:
        df = pd.read_csv(args.infile, dtype=str, nrows=args.sample_rows)
        if args.news_col not in df.columns:
            raise ValueError(f"Input CSV must contain '{args.news_col}' column")
        if args.date_col not in df.columns:
            raise ValueError(f"Input CSV must contain '{args.date_col}' column")
        if args.symbol_col not in df.columns:
            raise ValueError(f"Input CSV must contain '{args.symbol_col}' column")
        # impact column may be missing; we still include blank if missing
        for _, row in df.iterrows():
            text = row.get(args.news_col, "") or ""
            docs_text.append(text)
            meta.append((row.get(args.date_col, ""), row.get(args.symbol_col, ""), row.get(args.impact_col, "")))
    else:
        # treat infile as plain text corpus (one document per line)
        with open(args.infile, 'r', encoding='utf-8') as f:
            for line in f:
                docs_text.append(line.strip())
                meta.append(("", "", ""))

    print(f"Documents loaded: {len(docs_text)}")

    # tokenize
    print("Tokenizing documents...")
    tokenized_docs = [tokenize(txt) for txt in docs_text]

    # build vocab
    print("Building vocabulary (min_count=%d, max_vocab_size=%s)..." % (args.min_count, str(args.max_vocab_size)))
    vocab, inv_vocab, freqs = build_vocab(tokenized_docs, min_count=args.min_count, max_vocab_size=args.max_vocab_size)
    vocab_size = len(vocab)
    print(f"Vocab size after filtering: {vocab_size}")
    if vocab_size == 0:
        raise RuntimeError("Vocabulary is empty after applying min_count/max_vocab_size filters.")

    # prepare training examples (context,target) pairs
    print("Preparing training examples (context,target) pairs with window=%d ..." % args.window)
    examples = prepare_training_examples(tokenized_docs, vocab, args.window, max_examples=args.max_examples)
    print(f"Prepared {len(examples)} training examples (target-context pairs).")

    if len(examples) == 0:
        raise RuntimeError("No training examples found. Try lowering min_count or increasing corpus size.")

    # train CBOW
    print("Training CBOW (full softmax) with epochs=%d lr=%g ..." % (args.epochs, args.lr))
    V, U = train_cbow(examples, vocab_size, args.vector_size, args.epochs, args.lr, print_every=50000)

    # save embeddings & vocab
    np.save(os.path.join(args.save_dir, 'cbow_input_embeddings.npy'), V)
    np.save(os.path.join(args.save_dir, 'cbow_output_embeddings.npy'), U)
    with open(os.path.join(args.save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump({'word_to_idx': vocab, 'idx_to_word': inv_vocab}, f)
    print(f"Saved embeddings and vocab to {args.save_dir}")

    # vectorize each document as mean of input embeddings of its tokens (consistent with h)
    print("Vectorizing documents (averaging input embeddings for each news row)...")
    doc_vecs = vectorize_documents(tokenized_docs, V, vocab, args.vector_size)

    # write output CSV with date,symbol,news_vector,impact_score
    print(f"Writing vectorized CSV to {args.out} ...")
    with open(args.out, 'w', encoding='utf-8') as fout:
        fout.write('date,symbol,news_vector,impact_score\n')
        for i, vec in enumerate(doc_vecs):
            date, sym, impact = meta[i]
            vec_str = json.dumps(vec.tolist())
            # write quoted fields for safety
            fout.write(f'"{date}","{sym}","{vec_str}","{impact}"\n')
    print("Done. Output written.")

if __name__ == '__main__':
    main()
