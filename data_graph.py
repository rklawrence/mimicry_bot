from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable
import text_segmentation as ts
import tqdm
import random


def weighted_word_choice(results: list, use_weights: bool = True) -> str:
    """Processes a list of results from the word database, and chooses one of
    the words randomly with respect to the weights if use_weights is True

    Args:
        results (list):
            The results from a query to the graph db
        use_weights (bool):
            If True, the randomly chosen word will be based off of the weights
            represented in the connectors.

    Returns:
        str: The randomly chosen word.
    """


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    def associate_words(self, word1, word2, author):
        with self.driver.session() as session:
            result = session.execute_write(self._create_follow, word1, word2, author)
        return result

    @staticmethod
    def _create_follow(tx, word1, word2, author):
        query = (
            f"MERGE (w1:Word {{text: '{word1}', author: '{author}'}})"
            f"MERGE (w2:Word {{text: '{word2}', author: '{author}'}})"
            f"MERGE (w1)-[r:FOLLOWS]-(w2) "
            f"ON MATCH SET r.count = r.count + 1 "
            f"ON CREATE SET r.count = 1 "
            f"RETURN w1, r, w2 "
        )
        result = tx.run(query)
        return list(result)

    @staticmethod
    def _get_next_word(tx, word, num_words, author, use_weights=True):
        query = (
            f"MATCH"
            f"(s:Word{{text:'{word}', author: '{author}'}})-"
            f"[r:FOLLOWS]-"
            f"(w:Word)"
            f"MATCH (e:Word{{text: '<end>'}})"
            f"WHERE EXISTS("
            f"(w)-[*{num_words}]-(e:Word{{text: '<end>', author: '{author}'}})"
            f")"
            f"RETURN r, w"
        )
        result = tx.run(query)
        # process results and get a weighted random choice based on r
        word = weighted_word_choice(result, use_weights)

        return word

    def generate_sentence(self, length, author):
        current_word = "<start>"
        sentence = ""
        with self.driver.session() as session:
            for i in range(length):
                sentence += session.execute_read(
                    self._get_next_word,
                    current_word,
                    length - i,
                    author,
                    current_word == "<start>",
                )
        return sentence


def populate_graphs(app):
    with open(r"george_eliot.txt", "r", encoding="utf-8") as file:
        text = file.read()
    text = ts.preprocess_text(text)
    sentences = ts.segment_by_sentence(text)
    for index, sentence in enumerate(sentences):
        if index >= 3000:
            break
        print(index)
        words = ts.segment_by_word(sentence)
        for bigram in words:
            word1, word2 = bigram
            app.associate_words(word1, word2, "George Eliot")
    with open(r"sherlock_complete_texts.txt", "r", encoding="utf-8") as file:
        text = file.read()
    text = ts.preprocess_text(text)
    sentences = ts.segment_by_sentence(text)
    for index, sentence in enumerate(sentences):
        if index >= 3000:
            break
        print(index)
        words = ts.segment_by_word(sentence)
        for bigram in words:
            word1, word2 = bigram
            app.associate_words(word1, word2, "George Eliot")


if __name__ == "__main__":
    # Aura queries use an encrypted connection using the "neo4j+s" URI scheme
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "plagiarism"
    app = App(uri, user, password)

    app.close()
