import pathlib
import sys

sys.path.append(pathlib.Path(__file__).parent.__str__())

import re


def preprocess_text(text: str) -> str:
    """Applies preprocessing to the text to remove uneccesary whitespace and
    forms of punctiation that do not end a sentence.

    Args:
        text (str):
            The text that should be processed.

    Returns:
        str: The text after being processed.
    """
    text = re.sub(r"[\";\-\&/:'\(\),\[\]]+", "", text)
    text = re.sub(r"[1-9]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def segment_by_sentence(text: str) -> list[str]:
    """Segments the given text into a list of strings. For simplicity, the
    ending punctuation will be removed.
    NOTE: Single word sentences will be ignored.

    Args:
        text (str):
            The text to be segmented.

    Returns:
        list[str]:
            A list of sentences made from the text.
    """
    punctuation = r"[\.!\?]"
    punctuation_location = [index.start() for index in re.finditer(punctuation, text)]
    previous_index = 0
    sentences = list()
    for location in punctuation_location:
        sentence = text[previous_index:location].strip()
        previous_index = location + 1
        if len(sentence.split(" ")) <= 2:
            continue
        sentences.append(sentence)
    return sentences


def segment_by_word(sentence: str) -> list[tuple[str, str]]:
    """This segments a setence into a list of tuples where each tuple is a pair
    of words. The first word in the tuple is the preceding word while the
    second word is the current word.
    NOTE: <start> will be used when there is not preceding word.
    NOTE: <end> will be used as the current word that the last word precedes.

    Args:
        sentence (str):
            The sentence to be segmented.

    Returns:
        list[tuple[str, str]]:
            A pair of words where the first word is the word that came before
            the second word.
    """
    previous_word = "<start>"
    pairs = list()
    for word in sentence.split(" "):
        pairs.append((previous_word, word))
        previous_word = word
    pairs.append((word, "<end>"))
    return pairs


if __name__ == "__main__":
    with open(r"sherlock_complete_texts.txt", "r", encoding="utf-8") as file:
        text = file.read()
    text = preprocess_text(text)
    sentences = segment_by_sentence(text)
    print(len(sentences))
    sentence = sentences[8]
    print(sentence)
    print(segment_by_word(sentence))
