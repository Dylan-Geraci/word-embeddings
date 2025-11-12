import helper_merge_impact
import helper_make_corpus
import helper_create_skipgram

def run():

    # Step 1: Merge impact data
    helper_merge_impact.run()

    # Step 2: Create text corpus
    helper_make_corpus.run()

    # Step 3: Create skip-gram model
    helper_create_skipgram.run()


if __name__ == "__main__":
    run()