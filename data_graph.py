from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable
import text_segmentation as ts
import tqdm
import random

URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "plagiarism"


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
    words = list()
    weights = list()
    for result in results:
        weights.append(result.get("count"))
        words.append(result.get("text"))
    if use_weights:
        word = random.choices(words, weights=weights, k=1)[0]
    else:
        word = random.choice(words)
    return word


class App:
    def __init__(self, uri=URI, user=USERNAME, password=PASSWORD):
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
    def _get_next_word(tx, words, num_words, author, use_weights=True):
        query = "MATCH"
        for i, word in enumerate(words):
            query += f"(:Word{{text:'{word}', author: '{author}'}})-"
            if i == len(words) - 1:
                query += "[r:FOLLOWS]->"
            else:
                query += "[:FOLLOWS]->"
        query += (
            f"(w:Word)"
            f"MATCH (e:Word{{text: '<end>'}})"
            f"WHERE EXISTS("
            f"(w)-[*{num_words}]->(e:Word{{text: '<end>', author: '{author}'}})"
            f")"
            f"RETURN r.count as count, w.text as text"
        )
        result = tx.run(query)
        # process results and get a weighted random choice based on r
        word = weighted_word_choice(list(result), use_weights)

        return word

    def generate_sentence(self, length, author, n=1):
        current_word = "<start>"
        sentence = ""
        with self.driver.session() as session:
            words = [current_word]
            for i in range(length):
                if len(words) > n:
                    input_list = words[-n:]
                else:
                    input_list = words
                word = session.execute_read(
                    self._get_next_word,
                    input_list,
                    length - i,
                    author,
                    current_word != "<start>",
                )
                words.append(word)
                sentence += word + " "
                current_word = word
        sentence = sentence.strip() + "."
        return sentence


def populate_graphs(app: App):
    with open(r"shakespeare.txt", "r", encoding="utf-8") as file:
        text = file.read()
    text = ts.preprocess_text(text)
    sentences = ts.segment_by_sentence(text)

    i = 1
    for sentence in tqdm.tqdm(sentences):
        if i >= 10000:
            break
        words = ts.segment_by_word(sentence)
        for bigram in words:
            word1, word2 = bigram
            app.associate_words(word1, word2, "William Shakespeare")
        i += 1

    with open(r"sherlock_complete_texts.txt", "r", encoding="utf-8") as file:
        text = file.read()
    text = ts.preprocess_text(text)
    sentences = ts.segment_by_sentence(text)

    i = 0
    for sentence in tqdm.tqdm(sentences):
        if i >= 10000:
            break
        words = ts.segment_by_word(sentence)
        for bigram in words:
            word1, word2 = bigram
            app.associate_words(word1, word2, "Arthur Conan Doyle")
        i += 1


if __name__ == "__main__":
    # Aura queries use an encrypted connection using the "neo4j+s" URI scheme

    app = App()
    # populate_graphs(app)
    sentences = list()
    for i in range(20):
        sentence = app.generate_sentence(10, "Arthur Conan Doyle", n=6)
        print(sentence)
        sentences.append(sentence)
    print(sentences)
    app.close()
