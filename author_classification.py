"""An off the shelf implementation of a binary text classification model.
The article that was followed to implement this can be found at:
https://iq.opengenus.org/binary-text-classification-bert/
"""

import pathlib
import sys

sys.path.append(pathlib.Path(__file__).parent.__str__())

import pickle

import numpy as np
import torch
import torch.utils.data as data
import transformers
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

import text_segmentation as ts

MAX_LENGTH = 128
CLASSES = {0: "Arthur Conan Doyle", 1: "George Eliot"}


def preprocess_sentence(sentence: list[str]) -> list[str]:
    """Preprocesses the sentence to ensure the highest accuracy possible when
    it is used with the classfication model.

    Args:
        sentence (list[str]): An English sentence broken up by words.

    Returns:
        list[str]: The sentence after it has been preprocessed.
    """
    # stop_words = set(stopwords.words("english"))
    # return [w for w in sentence if w not in stop_words]
    return sentence


def tokenize_sentences(sentences: list[str]) -> tuple[list[int], list[int]]:
    """Uses the BertTokenizer to tokenize a sentence for input into the
    classification model.

    Args:
        sentence (str): An English sentence.

    Returns:
        [0]: list[int]:
            The tokens created from the sentence.
        [1]: list[int]:
            The attention mask to inform the model which tokens are important
            and which tokens are padding.
    """
    tokenizer = transformers.BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )
    tokenized_sentences = list()
    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)
        tokenized_sentences.append(tokenized_sentence)
    input_ids = pad_sequences(
        tokenized_sentences, maxlen=MAX_LENGTH, truncating="post", padding="post"
    )
    attention_masks = list()
    for sentence in input_ids:
        attention_mask = [int(token_id > 0) for token_id in sentence]
        attention_masks.append(attention_mask)
    return input_ids, attention_masks


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_model(
    data_set: list[list[int]],
    attention_masks: list[list[int]],
    labels: list[int],
    batch_size: int = 32,
    epochs: int = 4,
):
    """Trains a binary classification model based on the inputs.

    Args:
        data_set (list[list[int]]):
            The dataset represented as tokenized and padded words.
        attention_masks (list[list[int]]):
            The attention mask for the dataset
        labels (list[int]):
            The labels for the dataset.
            NOTE: These will either be 0 or 1
        batch_size (int):
            The size of the training batches.
            (Optional) Defaults to: 32
        epochs (int):
            The number of epochs in the training
            (Optional) Defaults to: 4
    """

    (
        train_inputs,
        validation_inputs,
        train_labels,
        validation_labels,
    ) = train_test_split(data_set, labels, test_size=0.2)
    train_masks, validation_masks, _, _ = train_test_split(
        attention_masks, labels, test_size=0.2
    )

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    train_data = data.TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = data.RandomSampler(train_data)
    train_dataloader = data.DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )

    validation_data = data.TensorDataset(
        validation_inputs, validation_masks, validation_labels
    )
    validation_sampler = data.SequentialSampler(validation_data)
    validation_dataloader = data.DataLoader(
        validation_data, sampler=validation_sampler, batch_size=batch_size
    )

    model = transformers.BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.cuda()

    device = torch.device("cuda")
    total_steps = len(train_dataloader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    loss_values = list()
    print("Starting training")
    for epoch in range(epochs):
        print(f"Starting epoch: {epoch}")
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(avg_train_loss)
        loss_values.append(avg_train_loss)

        model.eval()
        eval_accuracy = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))

    return model


def classify_text(model: transformers.BertPreTrainedModel, text: str) -> str:
    """Takes in a sentence and classifies it using the given model.

    Args:
        model (transformers.BertPreTrainedModel):
            A model trained to classify text.
        text (str):
            The sentence to be classified.
            NOTE: For best accuracy, it should have less than MAX_LENGTH - 2
            -2 words in it.

    Returns:
        str:
            The label associated with the classification in the model.
    """
    model.eval()
    text = ts.preprocess_text(text)
    words = ts.segment_by_word(text)
    tokens, attention_mask = tokenize_sentences(words)
    device = torch.device("cuda")
    tokens = torch.tensor(tokens).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    output = model(tokens, token_type_ids=None, attention_mask=attention_mask)
    result = np.argmax(output[0].detach().cpu().numpy(), axis=1).flatten()[0]
    return CLASSES[result]


def model_creation_pipeline():
    # Warning: This function might take more than 2 hours to finish.
    with open(r"sherlock_complete_texts.txt", encoding="utf-8") as file:
        text = file.read()
    text = ts.preprocess_text(text)
    doyle_sentences_raw = ts.segment_by_sentence(text)
    doyle_sentences = list()
    for sentence in doyle_sentences_raw:
        sentence = preprocess_sentence(sentence)
        doyle_sentences.append(sentence)

    with open(r"george_eliot.txt", encoding="utf-8") as file:
        text = file.read()
    text = ts.preprocess_text(text)
    eliot_sentences_raw = ts.segment_by_sentence(text)
    eliot_sentences = list()
    for sentence in eliot_sentences_raw:
        sentence = preprocess_sentence(sentence)
        eliot_sentences.append(sentence)

    doyle_sentences, doyle_masks = tokenize_sentences(doyle_sentences)
    doyle_labels = [0 for _ in range(len(doyle_sentences))]
    eliot_sentences, eliot_masks = tokenize_sentences(eliot_sentences)
    eliot_labels = [1 for _ in range(len(eliot_sentences))]

    doyle_sentences = doyle_sentences[0 : len(eliot_sentences)]
    doyle_masks = doyle_masks[0 : len(eliot_sentences)]
    doyle_labels = doyle_labels[0 : len(eliot_sentences)]

    sentences = np.concatenate((doyle_sentences, eliot_sentences), axis=0)
    labels = doyle_labels + eliot_labels
    attention_masks = np.concatenate((doyle_masks, eliot_masks), axis=0)
    print("starting training")

    model = train_model(sentences, attention_masks, labels)
    with open("model.p", "wb+") as file:
        pickle.dump(model, file)

    model


if __name__ == "__main__":
    # model_creation_pipeline()
    with open("model.p", "rb") as file:
        model = pickle.load(file)
    test_sentence = "In every varied posture, place, and hour,How widow'd every thought of every joy!"
    print(classify_text(model, test_sentence))
