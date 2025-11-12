#!/usr/bin/env python3
"""
cbow_fast.py

Streaming, memory-efficient CBOW trainer using negative sampling (numpy only).

Key features:
 - Two-pass approach: (1) build vocab counts streaming, (2) train streaming per epoch,
   (3) vectorize docs streaming.
 - Negative sampling default (fast). Full softmax is optional but VERY slow.
 - Subsampling of frequent words to reduce training cost.
 - No ML framework required (numpy + pandas only).

Usage example:
  python cbow_fast.py --in aggregate_news_with_impact.csv --out vectorized_news_cbow_embeddings.csv \
    --vector-size 100 --window 5 --min-count 5 --epochs 1 --neg-samples 5 --chunksize 2000 --max-docs 200000
"""
import argparse
import json
import math
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd

WORD_RE = re.compile(r"\w+")


def tokenize(text):
    if not isinstance(text, str):
        return []
    return [w.lower() for w in WORD_RE.findall(text)]


def sigmoid(x):
    # numerically stable sigmoid for numpy arrays/scalars
    # uses np.where to avoid overflow
    x = np.array(x, dtype=np.float64)
    pos_mask = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[~pos_mask])
    out[~pos_mask] = exp_x / (1.0 + exp_x)
    return out


def build_vocab_stream(csv_path, news_col='news', chunksize=2000, min_count=5, max_vocab_size=None, sample_rows=None):
    """One-pass: count tokens, return vocab (word->idx), inv_vocab, counts, total_tokens."""
    counts = Counter()
    total = 0
    it = pd.read_csv(csv_path, usecols=[news_col], dtype=str, chunksize=chunksize)
    rows_seen = 0
    for chunk in it:
        for text in chunk[news_col].fillna('').astype(str):
            tokens = tokenize(text)
            counts.update(tokens)
            total += len(tokens)
            rows_seen += 1
            if sample_rows is not None and rows_seen >= sample_rows:
                break
        if sample_rows is not None and rows_seen >= sample_rows:
            break

    # filter by min_count
    items = [(w, c) for w, c in counts.items() if c >= min_count]
    # sort by freq desc
    items.sort(key=lambda x: x[1], reverse=True)
    if max_vocab_size is not None:
        items = items[:max_vocab_size]
    vocab = {w: i for i, (w, _) in enumerate(items)}
    inv_vocab = [w for w, _ in items]
    freqs = np.array([c for _, c in items], dtype=np.int64)
    return vocab, inv_vocab, freqs, total


def build_negative_sampling_cdf(freqs, power=0.75):
    """Return cumulative distribution array for sampling negatives with P ~ freq^power."""
    # freqs: numpy array of counts (length V)
    p = freqs.astype(np.float64) ** power
    p /= p.sum()
    cdf = np.cumsum(p)
    return cdf


def sample_negatives(cdf, K, forbidden=None):
    """Sample K negative indices from cdf (1D np array). Avoid 'forbidden' int if provided."""
    V = len(cdf)
    # draw extra to filter out forbidden; draw factor times K
    out = []
    while len(out) < K:
        # sample batch
        r = np.random.rand(K * 2)
        ids = np.searchsorted(cdf, r)
        for iid in ids:
            if forbidden is not None and iid == forbidden:
                continue
            out.append(iid)
            if len(out) >= K:
                break
    return out[:K]


def prepare_context(indices, pos, window):
    """Given list of indices for a tokenized document and a position, return list of context indices (exclude pos)."""
    n = len(indices)
    left = max(0, pos - window)
    right = min(n, pos + window + 1)
    context = [indices[i] for i in range(left, right) if i != pos]
    return context


def train_streaming_negative_sampling(csv_path,
                                      news_col,
                                      vocab,
                                      inv_vocab,
                                      freqs,
                                      total_tokens,
                                      vector_size=100,
                                      window=5,
                                      epochs=1,
                                      lr=0.025,
                                      neg_samples=5,
                                      chunksize=2000,
                                      subsample_t=1e-5,
                                      max_docs=None,
                                      sample_rows_for_build=None,
                                      use_full_softmax=False,
                                      print_every=100000):
    """
    Train CBOW using online updates with negative sampling.
    Returns input embeddings V (vocab_size x d) and output embeddings U (vocab_size x d).
    """
    V_size = len(vocab)
    rng = np.random.RandomState(42)
    V = (rng.randn(V_size, vector_size) * 0.01).astype(np.float32)  # input embeddings
    U = (rng.randn(V_size, vector_size) * 0.01).astype(np.float32)  # output embeddings

    # Precompute keep probabilities (subsampling) per word index
    # gensim-like keep_prob: p = (sqrt(freq/ t_total) + 1) * (t_total / freq)
    # where freq = count / total_tokens
    keep_prob = np.ones(V_size, dtype=np.float64)
    if subsample_t > 0.0:
        for w_idx, cnt in enumerate(freqs):
            f = cnt / max(1, total_tokens)
            # gensim formula (commonly used)
            prob = (math.sqrt(f / subsample_t) + 1) * (subsample_t / f) if f > 0 else 1.0
            if prob > 1.0:
                prob = 1.0
            keep_prob[w_idx] = prob

    # negative sampling cdf
    if not use_full_softmax:
        cdf = build_negative_sampling_cdf(freqs, power=0.75)
    else:
        cdf = None

    # Training: iterate epochs, stream CSV rows in chunks; for each doc produce CBOW updates for each position
    steps = 0
    total_examples = 0
    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}/{epochs}")
        doc_count = 0
        chunk_iter = pd.read_csv(csv_path, usecols=[news_col], dtype=str, chunksize=chunksize)
        for chunk in chunk_iter:
            for text in chunk[news_col].fillna('').astype(str):
                doc_count += 1
                if max_docs is not None and doc_count > max_docs:
                    break

                tokens = tokenize(text)
                # Map tokens to vocab indices, filter OOV
                idxs = [vocab[w] for w in tokens if w in vocab]
                if len(idxs) == 0:
                    continue

                # Apply subsampling (random drop of very frequent words)
                if subsample_t > 0.0:
                    idxs = [i for i in idxs if random.random() <= keep_prob[i]]
                    if len(idxs) == 0:
                        continue

                # For each position in doc produce a CBOW update
                n = len(idxs)
                for pos in range(n):
                    target = idxs[pos]
                    context = prepare_context(idxs, pos, window)
                    if len(context) == 0:
                        continue
                    C = len(context)

                    # Compute hidden vector h = mean of input embeddings for context
                    # shape (d,)
                    h = np.mean(V[context], axis=0)  # float32

                    if use_full_softmax:
                        # Full softmax (VERY slow). implemented for completeness.
                        logits = U.dot(h)  # (V,)
                        # stable softmax
                        mx = np.max(logits)
                        ex = np.exp(logits - mx)
                        probs = ex / ex.sum()
                        # gradient on output embeddings: probs - y
                        err = probs
                        err[target] -= 1.0
                        # update U
                        # U_j <- U_j - lr * err_j * h
                        U -= (lr * err)[:, np.newaxis] * h[np.newaxis, :]
                        # grad wrt h
                        grad_h = U.T.dot(err)
                        # update context input vectors
                        grad_each = (lr * (1.0 / C)) * grad_h
                        for ci in context:
                            V[ci] -= grad_each.astype(np.float32)
                    else:
                        # Negative sampling:
                        # Positive sample update
                        ut = U[target]  # (d,)
                        score_pos = sigmoid(np.dot(ut, h))  # scalar
                        # gradient for ut: (sigma(u·h) - 1) * h
                        g_ut = (score_pos - 1.0)  # scalar
                        # update U[target] -= lr * g_ut * h
                        U[target] -= (lr * g_ut) * h

                        # sample negatives (avoid target)
                        negs = sample_negatives(cdf, neg_samples, forbidden=target)
                        # For negatives, update each un with label 0: gradient = sigma(un·h) * h
                        # vectorized:
                        un = U[negs]  # (K, d)
                        scores_neg = sigmoid(un.dot(h))  # (K,)
                        # update U[negs] -= lr * (scores_neg)[:,None] * h[None,:]
                        U[negs] -= (lr * scores_neg)[:, np.newaxis] * h[np.newaxis, :]

                        # grad wrt h = (sigma(ut·h)-1)*ut + sum_k sigma(un_k·h)*un_k
                        grad_h = (score_pos - 1.0) * ut + (scores_neg[:, None] * un).sum(axis=0)

                        # update input embeddings for each context word:
                        grad_each = (lr * (1.0 / C)) * grad_h
                        for ci in context:
                            V[ci] -= grad_each.astype(np.float32)

                    steps += 1
                    total_examples += 1
                    if steps % print_every == 0:
                        print(f"Epoch {epoch} steps {steps} examples {total_examples}")
                # end for pos
            # end for text
            if max_docs is not None and doc_count > max_docs:
                break
        # end chunk_iter
        print(f"Epoch {epoch} done. docs seen this epoch: {doc_count}, total examples so far: {total_examples}")
    print("Training finished.")
    return V, U, total_examples


def vectorize_docs_stream(csv_path, out_csv, news_col, date_col, symbol_col, impact_col, vocab, V,
                          chunksize=2000):
    """Stream corpus and compute doc vectors (mean of input embeddings per doc), write out CSV."""
    d = V.shape[1]
    header = 'date,symbol,news_vector,impact_score\n'
    with open(out_csv, 'w', encoding='utf-8') as fout:
        fout.write(header)
        it = pd.read_csv(csv_path, dtype=str, chunksize=chunksize)
        for chunk in it:
            for _, row in chunk.iterrows():
                text = row.get(news_col, '') or ''
                date = row.get(date_col, '') or ''
                symbol = row.get(symbol_col, '') or ''
                impact = row.get(impact_col, '') or ''
                tokens = tokenize(text)
                idxs = [vocab[w] for w in tokens if w in vocab]
                if len(idxs) == 0:
                    vec = np.zeros(d, dtype=float)
                else:
                    vec = np.mean(V[idxs], axis=0)
                vec_str = json.dumps(vec.tolist())
                fout.write(f'"{date}","{symbol}","{vec_str}","{impact}"\n')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', default='./datasets/aggregate_news_with_impact.csv', help='Input CSV (with news column)')
    parser.add_argument('--out', dest='out', default='./datasets/vectorized_news_cbow_embeddings.csv', help='Output CSV path')
    parser.add_argument('--vector-size', type=int, default=100)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--min-count', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--neg-samples', type=int, default=5, help='Number of negative samples (K). If 0 and --use-full-softmax not set, fallback to K=5.')
    parser.add_argument('--chunksize', type=int, default=2000)
    parser.add_argument('--max-vocab-size', type=int, default=20000)
    parser.add_argument('--max-docs', type=int, default=None, help='Limit number of documents to train on (for quick experiments).')
    parser.add_argument('--max-examples', type=int, default=None, help='Deprecated. Kept for compatibility.')
    parser.add_argument('--subsample-t', type=float, default=1e-5)
    parser.add_argument('--save-dir', default='datasets')
    parser.add_argument('--news-col', default='news')
    parser.add_argument('--date-col', default='date')
    parser.add_argument('--symbol-col', default='symbol')
    parser.add_argument('--impact-col', default='impact_score')
    parser.add_argument('--use-full-softmax', action='store_true', help='Force full softmax (very slow!).')
    parser.add_argument('--sample-rows-for-build', type=int, default=None, help='Use only first N rows to build vocab (optional).')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Building vocabulary (streaming)...")
    vocab, inv_vocab, freqs, total_tokens = build_vocab_stream(
        args.infile, news_col=args.news_col, chunksize=args.chunksize,
        min_count=args.min_count, max_vocab_size=args.max_vocab_size, sample_rows=args.sample_rows_for_build)
    vocab_size = len(vocab)
    print(f"Vocabulary size = {vocab_size}, total tokens (approx) = {total_tokens}")
    if vocab_size == 0:
        raise RuntimeError("Vocabulary empty. Lower min_count or provide more data.")

    # Train
    use_full = args.use_full_softmax
    neg_k = args.neg_samples if args.neg_samples > 0 else (0 if use_full else 5)
    print("Training parameters:")
    print(f" vector_size={args.vector_size} window={args.window} epochs={args.epochs} lr={args.lr}")
    print(f" neg_samples={neg_k} chunksize={args.chunksize} subsample_t={args.subsample_t} max_docs={args.max_docs}")
    V, U, total_examples = train_streaming_negative_sampling(
        args.infile, args.news_col, vocab, inv_vocab, freqs, total_tokens,
        vector_size=args.vector_size, window=args.window,
        epochs=args.epochs, lr=args.lr, neg_samples=neg_k,
        chunksize=args.chunksize, subsample_t=args.subsample_t,
        max_docs=args.max_docs, sample_rows_for_build=args.sample_rows_for_build,
        use_full_softmax=use_full, print_every=50000
    )

    # Save embeddings & vocab
    np.save(os.path.join(args.save_dir, 'cbow_input_embeddings.npy'), V)
    np.save(os.path.join(args.save_dir, 'cbow_output_embeddings.npy'), U)
    with open(os.path.join(args.save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump({'word_to_idx': vocab, 'idx_to_word': inv_vocab}, f)
    print(f"Saved embeddings to {args.save_dir} (V/U and vocab.json)")

    # Vectorize docs (streaming)
    print("Vectorizing documents (streaming) and writing output CSV...")
    vectorize_docs_stream(args.infile, args.out, args.news_col, args.date_col, args.symbol_col, args.impact_col, vocab, V, chunksize=args.chunksize)
    print("Wrote:", args.out)
    print("Done.")


if __name__ == '__main__':
    run()
