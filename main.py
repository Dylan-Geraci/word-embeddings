# main.py
import os
import argparse
from train import train_and_evaluate

# Import the files for creating the skip-gram and cbow embedding matrices
import embedding
import helper_v2_create_cbow

def run_for_file(input_csv, out_model, model_name):
    print("="*80)
    print(f"Training & evaluating model for {model_name}")
    print("Input CSV:", input_csv)
    print("Output checkpoint:", out_model)
    # default hyperparams (you can tune)
    result = train_and_evaluate(input_csv=input_csv,
                                out_model_path=out_model,
                                hidden_dims=(128,64),
                                batch_size=256,
                                lr=1e-3,
                                epochs=8,               # default modest number; increase for real training
                                test_size=0.2,
                                dropout=0.2)
    print(f"Result for {model_name}: test_acc={result['test_accuracy']:.4f}")
    print("="*80)
    return result

def main():
    # default file names produced by your earlier scripts:
    skipgram_csv = './datasets/vectorized_news_skip-gram_embeddings.csv'
    cbow_csv = './datasets/vectorized_news_cbow_embeddings.csv'  # matches the naming in earlier script

    # output models
    skipgram_pth = './datasets/skipgram.pth'
    cbow_pth = './datasets/cbow.pth'

    # First, call the embedding script to merge the impact data, create a text corpus and create a skip-gram model
    embedding.run()

    # Second, call the helper script to create a cbow model
    helper_v2_create_cbow.run()

    # Third, call the train script to train the cbow and skipgram models separately
    # (sanity checks)
    if not os.path.exists(skipgram_csv):
        print(f"Warning: {skipgram_csv} not found. Please generate skipgram vectors first or change the filename in main.py")
    else:
        run_for_file(skipgram_csv, skipgram_pth, 'Skip-gram')

    if not os.path.exists(cbow_csv):
        print(f"Warning: {cbow_csv} not found. Please generate CBOW vectors first or change the filename in main.py")
    else:
        run_for_file(cbow_csv, cbow_pth, 'CBOW')

    print("All done.")

if __name__ == '__main__':
    main()
