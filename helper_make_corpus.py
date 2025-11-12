#!/usr/bin/env python3
"""
make_corpus.py

Concatenate the 'news' column into a corpus text file (one document per line).
Useful for training Word2Vec or for quick inspection.

Usage:
    python make_corpus.py --in aggregate_news_with_impact.csv --out corpus.txt

If you want to read directly from aggregate_news.csv (without impact), pass that file.
"""
import argparse
import pandas as pd
import sys

def run():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', default='./datasets/aggregate_news_with_impact.csv', help='Input CSV with "news" column')
    p.add_argument('--out', default='./datasets/corpus.txt', help='Output corpus file (one doc per line)')
    p.add_argument('--news-col', default='news', help='Column name containing news text')
    args = p.parse_args()

    df = pd.read_csv(args.infile, dtype=str)
    if args.news_col not in df.columns:
        print(f"ERROR: Column '{args.news_col}' not found in {args.infile}", file=sys.stderr)
        sys.exit(1)

    # Replace newlines inside news with spaces so each row maps to a single line in corpus
    docs = df[args.news_col].fillna('').astype(str).apply(lambda s: s.replace('\r',' ').replace('\n',' '))
    print(f"Writing {len(docs)} documents to {args.out} ...")
    with open(args.out, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(doc.strip() + '\n')
    print("Done.")

if __name__ == '__main__':
    run()
