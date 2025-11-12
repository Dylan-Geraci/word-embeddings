#!/usr/bin/env python3
"""
merge_impact.py

Merge impact_score from historical_prices_impact.csv into aggregate_news.csv by (date, symbol).

Usage:
    python merge_impact.py \
        --news aggregate_news.csv \
        --prices historical_prices_impact.csv \
        --out aggregate_news_with_impact.csv

Notes:
- Dates are normalized with pandas.to_datetime then formatted as YYYY-MM-DD before merging.
- If no matching (date,symbol) is found for a news row, impact_score remains NaN (so you can decide how to handle).
"""
import argparse
import pandas as pd

def run():
    p = argparse.ArgumentParser()
    p.add_argument('--news', default='./datasets/aggregated_news.csv', help='Path to aggregate_news.csv')
    p.add_argument('--prices', default='./datasets/historical_prices_impact.csv', help='Path to historical_prices_impact.csv')
    p.add_argument('--out', default='./datasets/aggregate_news_with_impact.csv', help='Output CSV path')
    p.add_argument('--news-date-col', default='date')
    p.add_argument('--prices-date-col', default='date')
    p.add_argument('--symbol-col', default='symbol')
    args = p.parse_args()

    print("Loading news...")
    news = pd.read_csv(args.news, dtype=str)
    print("Loading prices (impact)...")
    prices = pd.read_csv(args.prices, dtype=str)

    # Normalize date columns to YYYY-MM-DD
    print("Normalizing date columns...")
    news[args.news_date_col] = pd.to_datetime(news[args.news_date_col], errors='coerce').dt.date.astype('str')
    prices[args.prices_date_col] = pd.to_datetime(prices[args.prices_date_col], errors='coerce').dt.date.astype('str')

    # Keep only date, symbol, impact_score from prices
    if 'impact_score' not in prices.columns:
        raise ValueError("prices file must contain 'impact_score' column")
    prices_small = prices[[args.prices_date_col, args.symbol_col, 'impact_score']].copy()

    # Merge (left join to keep all news rows)
    print("Merging...")
    merged = pd.merge(news, prices_small,
                      left_on=[args.news_date_col, args.symbol_col],
                      right_on=[args.prices_date_col, args.symbol_col],
                      how='left', suffixes=('', '_price'))

    # If both date columns still exist, drop the duplicate price date column
    if 'date_price' in merged.columns:
        merged.drop(columns=['date_price'], inplace=True)

    # Show statistics
    total = len(merged)
    matched = merged['impact_score'].notna().sum()
    print(f"Total news rows: {total}")
    print(f"Rows with matched impact_score: {matched}")
    print(f"Rows without matched impact_score: {total - matched}")

    # Save
    merged.to_csv(args.out, index=False)
    print(f"Wrote merged file to {args.out}")

if __name__ == '__main__':
    run()
