"""main.py - This applies the techniques generated in the other files to
generate text using the ngram method and graph method. The accuracy and times
of these methods are then compared.

Each approach generate 500 sentences that will be evaluated.
"""

import pickle
import text_segmentation as ts
import ngram
import data_graph as dg
from time import time
import author_classification as ac
import tqdm

NUM_SENTENCES = 500


def print_results(
    approach: str, author: str, average_sentence_time: float, percent_correct: float
) -> None:
    """Generates the results for each of the given approaches and prints them
    to the console

    Args:
        approach (str):
            The name of the approach taken.
        author (str):
            The name of the author used to generate the sentences
        average_sentence_time (float):
            The average amount of time needed to generate the sentences
        percent_correct (float):
            The percent of sentences generated using this approach that were
            correctly attributed to the given author
    """
    print(
        f"Approach: {approach}\n"
        f"Author: {author}\n"
        f"Average Sentence Time: {average_sentence_time:.2f}\n"
        f"Percent Correct: {percent_correct:.2f}"
    )


def determine_accuracy(author: str, sentences: list[str], model) -> float:
    """Determines the the accuracy of the given sentences when put into the
    author classification model.

    Args:
        author (str):
            The name of the author being classified
        sentences (list[str]):
            The sentences that act as the dataset.
        model:
            The classification model

    Returns:
        float: The percent accuracy of the sentences
    """
    correct = 0
    for sentence in tqdm.tqdm(sentences):
        if ac.classify_text(model, sentence) == author:
            correct += 1
    return float(correct) / float(len(sentences))


if __name__ == "__main__":
    N = 2
    LEN = 10
    with open("model.p", "rb") as file:
        model = pickle.load(file)

    # Doyle ngram approach
    with open("sherlock_complete_texts.txt", "r", encoding="utf-8") as file:
        text = file.read()
        text = ts.preprocess_text(text)
        text = ts.segment_by_sentence(text)
    doyle_gram = ngram.create_model_based_on_text(N, text)
    start_time = time()
    doyle_gram_sentences = [doyle_gram.generate_text(LEN) for _ in range(NUM_SENTENCES)]
    doyle_gram_average_time = (time() - start_time) / NUM_SENTENCES
    doyle_gram_accuracy = determine_accuracy(
        "Arthur Conan Doyle", doyle_gram_sentences, model
    )
    print_results(
        "ngram",
        "Arthur Conan Doyle",
        doyle_gram_average_time,
        doyle_gram_accuracy,
    )

    # Shakespeare ngram approach
    with open("shakespeare.txt", "r", encoding="utf-8") as file:
        text = file.read()
        text = ts.preprocess_text(text)
        text = ts.segment_by_sentence(text)
    shakespeare_gram = ngram.create_model_based_on_text(N, text)
    start_time = time()
    shakespeare_gram_sentences = [
        shakespeare_gram.generate_text(LEN) for _ in range(NUM_SENTENCES)
    ]
    shakespeare_gram_average_time = (time() - start_time) / NUM_SENTENCES
    shakespeare_gram_accuracy = determine_accuracy(
        "William Shakespeare", shakespeare_gram_sentences, model
    )
    print_results(
        "ngram",
        "William Shakespeare",
        shakespeare_gram_average_time,
        shakespeare_gram_accuracy,
    )

    # graph setup
    app = dg.App()

    # Doyle graph approach
    start_time = time()
    doyle_graph_sentences = [
        app.generate_sentence(LEN, "Arthur Conan Doyle", n=N)
        for _ in range(NUM_SENTENCES)
    ]
    doyle_graph_average_time = (time() - start_time) / NUM_SENTENCES
    doyle_graph_accuracy = determine_accuracy(
        "Arthur Conan Doyle", doyle_gram_sentences, model
    )
    print_results(
        "graph", "Arthur Conan Doyle", doyle_graph_average_time, doyle_graph_accuracy
    )

    # Shakespeare graph approach
    start_time = time()
    shakespeare_graph_sentences = [
        app.generate_sentence(LEN, "William Shakespeare", n=N)
        for _ in range(NUM_SENTENCES)
    ]
    shakespeare_graph_average_time = (time() - start_time) / NUM_SENTENCES
    shakespeare_graph_accuracy = determine_accuracy(
        "William Shakespeare", shakespeare_gram_sentences, model
    )
    print_results(
        "graph",
        "William Shakespeare",
        shakespeare_graph_average_time,
        shakespeare_graph_accuracy,
    )
